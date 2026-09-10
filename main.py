"""iCourse Subscriber — top-level orchestration."""

import datetime
import time
import traceback

from src.runtime import config
from src.data.database import Database
from src.data.transcript_store import load_proofread
from src.api.emailer import Emailer
from src.api.icourse import ICourseClient
from src.ai.blackboard_vision import course_requires_blackboard
from src.pipeline.blackboard_lecture_runner import BlackboardLectureRunner as LectureRunner
from src.pipeline.lecture_runner import TRACEABLE_SUMMARY_PREFIX
from src.runtime.reporter import Reporter
from src.runtime.scheduler import Scheduler
from src.ai.summarizer import Summarizer
from src.ai.transcriber import Transcriber
from src.api.webvpn import WebVPNSession


BLACKBOARD_NOTES_MODEL_PREFIX = "blackboard-llm-editor-v9/"


def _has_webvpn_ticket(vpn: WebVPNSession) -> bool:
    return any(
        "wengine_vpn_ticket" in cookie.name
        for cookie in vpn.session.cookies
    )


def login_with_retry(
    max_attempts: int = 3,
    icourse_attempts_per_session: int = 4,
) -> WebVPNSession:
    last_error: Exception | None = None
    for vpn_attempt in range(max_attempts):
        vpn = WebVPNSession()
        try:
            print(
                f"\n[Login] WebVPN (session {vpn_attempt + 1}/{max_attempts})..."
            )
            vpn.login()
        except Exception as exc:
            last_error = exc
            if vpn_attempt < max_attempts - 1:
                wait = min(5 * (2 ** vpn_attempt), 20)
                print(
                    f"  WebVPN failed: {type(exc).__name__}: {exc}; "
                    f"retrying fresh session in {wait}s..."
                )
                time.sleep(wait)
                continue
            raise

        for cas_attempt in range(icourse_attempts_per_session):
            try:
                print(
                    f"[Login] iCourse CAS (same VPN, attempt {cas_attempt + 1}/"
                    f"{icourse_attempts_per_session})..."
                )
                vpn.authenticate_icourse()
                return vpn
            except Exception as exc:
                last_error = exc
                if not _has_webvpn_ticket(vpn):
                    print(
                        "  iCourse CAS failed and the WebVPN ticket cookie "
                        "is missing; rebuilding WebVPN session."
                    )
                    break
                if cas_attempt < icourse_attempts_per_session - 1:
                    wait = min(3 * (2 ** cas_attempt), 18)
                    print(
                        f"  iCourse CAS failed: {type(exc).__name__}: {exc}"
                    )
                    print(
                        "  Keeping the current WebVPN session; "
                        f"retrying iCourse CAS in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"  iCourse CAS failed {icourse_attempts_per_session} "
                        "times on the same WebVPN session."
                    )

        if vpn_attempt < max_attempts - 1:
            wait = min(5 * (2 ** vpn_attempt), 20)
            print(
                f"  Rebuilding WebVPN session in {wait}s "
                f"(next session {vpn_attempt + 2}/{max_attempts})..."
            )
            time.sleep(wait)

    if last_error is not None:
        raise last_error
    raise RuntimeError("WebVPN/iCourse login failed without a captured error")


def _check_session(client: ICourseClient) -> None:
    if client.check_alive():
        return
    print("[Session] WebVPN session expired, re-logging in...")
    client.vpn = login_with_retry()
    client._userinfo = None


def _summary_is_current(row: dict, course_title: str) -> bool:
    model = str(row.get("summary_model") or "")
    if course_requires_blackboard(course_title):
        # v9 can legitimately reuse a previous final board section when a
        # historical schema bug destroyed only the raw blackboard cache.  Once
        # that combined v9 output is generated, do not force another vision run
        # merely because ``blackboard_latex`` is absent.
        return bool(row.get("summary")) and model.startswith(
            BLACKBOARD_NOTES_MODEL_PREFIX
        )
    return model.startswith(TRACEABLE_SUMMARY_PREFIX)


def _enumerate_lectures(
    client: ICourseClient,
    db: Database,
    course_id: str,
    course_title: str,
    reporter: Reporter,
) -> list[dict]:
    """Register remote lectures and return rows that still need processing."""
    # Keep the original implementation below this point unchanged.
    _check_session(client)
    lectures = client.get_sub_list(course_id)
    if not lectures:
        reporter.info(f"  No lectures found for course {course_id}.")
        return []

    for lecture in lectures:
        sub_id = str(lecture.get("sub_id") or lecture.get("id") or "")
        if not sub_id:
            continue
        sub_title = str(
            lecture.get("sub_title")
            or lecture.get("title")
            or lecture.get("name")
            or sub_id
        )
        date = str(
            lecture.get("date")
            or lecture.get("start_time")
            or lecture.get("created_at")
            or ""
        )
        db.insert_lecture(sub_id, course_id, sub_title, date)

    pending: list[dict] = []
    for lecture in lectures:
        sub_id = str(lecture.get("sub_id") or lecture.get("id") or "")
        if not sub_id:
            continue
        row = db.get_lecture(sub_id)
        if not row:
            continue
        if _summary_is_current(row, course_title):
            continue
        if int(row.get("error_count") or 0) >= 3:
            continue
        pending.append(dict(row))
    return pending


def _drive_lectures(
    client: ICourseClient,
    db: Database,
    reporter: Reporter,
    summarizer: Summarizer,
    transcriber: Transcriber,
) -> list[dict]:
    """Process configured courses and return newly-completed email items."""
    completed: list[dict] = []
    for course_id in config.COURSE_IDS:
        _check_session(client)
        try:
            info = client.get_course_info(course_id)
            course_title = str(info.get("title") or info.get("course_name") or course_id)
            teacher = str(info.get("teacher") or info.get("teacher_name") or "")
        except Exception:
            course_title = str(course_id)
            teacher = ""
        db.upsert_course(str(course_id), course_title, teacher)
        reporter.info(f"\n=== Course {course_id}: {course_title} ===")

        pending = _enumerate_lectures(
            client, db, str(course_id), course_title, reporter
        )
        if not pending:
            reporter.info("  No lectures need processing.")
            continue

        for index, lecture in enumerate(pending):
            next_info = None
            if index + 1 < len(pending):
                nxt = pending[index + 1]
                next_info = (
                    str(nxt.get("sub_id") or ""),
                    str(nxt.get("sub_title") or ""),
                )
            runner = LectureRunner(
                client,
                db,
                reporter,
                summarizer,
                transcriber,
            )
            try:
                summary = runner.run(
                    str(course_id),
                    course_title,
                    lecture,
                    next_info=next_info,
                )
                if summary:
                    row = db.get_lecture(str(lecture["sub_id"]))
                    if row:
                        completed.append(dict(row))
            except Exception:
                reporter.info(
                    f"ERROR processing {lecture.get('sub_id')}:\n{traceback.format_exc()}"
                )

    return completed


def main() -> None:
    print("iCourse Subscriber starting...")
    db = Database()
    reporter = Reporter()
    summarizer = Summarizer()
    transcriber = Transcriber()

    try:
        vpn = login_with_retry()
        client = ICourseClient(vpn)
        completed = _drive_lectures(
            client, db, reporter, summarizer, transcriber
        )

        # Include processed-but-not-yet-emailed rows from previous runs.
        unsent = db.get_unsent_lectures()
        by_id = {str(item["sub_id"]): item for item in completed}
        for row in unsent:
            by_id.setdefault(str(row["sub_id"]), row)
        email_items = list(by_id.values())

        if email_items and config.RECEIVER_EMAIL:
            attachments = []
            for item in email_items:
                proofread = load_proofread(db, str(item["sub_id"]))
                if proofread:
                    markdown, _, _ = proofread
                    title = str(item.get("sub_title") or item["sub_id"])
                    safe = "".join(
                        ch if ch not in '\\/:*?\"<>|' else "_"
                        for ch in title
                    ).strip() or str(item["sub_id"])
                    attachments.append(
                        (f"{safe}-AI校订语音转写.md", markdown.encode("utf-8"), "text/markdown")
                    )

            emailer = Emailer()
            emailer.send_summary(email_items, attachments=attachments)
            db.mark_emailed_batch([str(item["sub_id"]) for item in email_items])
            print(f"Sent {len(email_items)} lecture(s) in one email.")
        else:
            print("No email to send.")
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("Run complete.")


if __name__ == "__main__":
    main()
