"""iCourse Subscriber — top-level orchestration."""

import datetime
import time
import traceback

from src.runtime import config
from src.data.database import Database
from src.api.emailer import Emailer
from src.api.icourse import ICourseClient
from src.ai.blackboard_vision import course_requires_blackboard
from src.pipeline.blackboard_lecture_runner import BlackboardLectureRunner as LectureRunner
from src.runtime.reporter import Reporter
from src.runtime.scheduler import Scheduler
from src.ai.summarizer import Summarizer
from src.ai.transcriber import Transcriber
from src.api.webvpn import WebVPNSession


BLACKBOARD_NOTES_MODEL_PREFIX = "blackboard-latex-notes-v2/"


def _has_webvpn_ticket(vpn: WebVPNSession) -> bool:
    """Whether the current requests session still carries a WebVPN ticket."""
    return any(
        "wengine_vpn_ticket" in cookie.name
        for cookie in vpn.session.cookies
    )


def login_with_retry(
    max_attempts: int = 3,
    icourse_attempts_per_session: int = 4,
) -> WebVPNSession:
    """Login to WebVPN, retrying flaky iCourse CAS on the same VPN session.

    WebVPN login can take about a minute, whereas the iCourse CAS redirect
    chain is the part that intermittently bounces back to /login. Therefore
    we keep a successfully established WebVPN session and retry only iCourse
    CAS with exponential backoff. A fresh WebVPN session is created only when
    the ticket cookie disappears or same-session CAS retries are exhausted.
    """
    last_error: Exception | None = None

    for vpn_attempt in range(max_attempts):
        vpn = WebVPNSession()
        try:
            print(
                f"\n[Login] WebVPN "
                f"(session {vpn_attempt + 1}/{max_attempts})..."
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
                    f"[Login] iCourse CAS "
                    f"(same VPN, attempt {cas_attempt + 1}/"
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
                        f"  iCourse CAS failed "
                        f"{icourse_attempts_per_session} times on the same "
                        "WebVPN session."
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
    """Verify WebVPN session; re-login in place if expired."""
    if client.check_alive():
        return
    print("[Session] WebVPN session expired, re-logging in...")
    client.vpn = login_with_retry()
    client._userinfo = None


def _enumerate_lectures(
    client: ICourseClient,
    db: Database,
    reporter: Reporter,
) -> list[tuple[str, str, dict]]:
    """List every lecture that should be processed in this run."""
    out: list[tuple[str, str, dict]] = []

    for course_id in config.COURSE_IDS:
        try:
            _check_session(client)
            detail = client.get_course_detail(course_id)
            course_title = detail["title"]
            teacher = detail["teacher"]
            lectures = detail["lectures"]
            playback_count = sum(1 for lec in lectures if lec.get("has_playback"))

            reporter.course_header(
                course_id,
                course_title,
                teacher,
                total=len(lectures),
                playback=playback_count,
            )
            db.upsert_course(course_id, course_title, teacher)

            seen_sub_titles: set[str] = set()
            deduped = []
            for lec in lectures:
                title = lec.get("sub_title", "")
                if title and title in seen_sub_titles:
                    reporter.course_dedup_skip(title, lec["sub_id"])
                    continue
                if title:
                    seen_sub_titles.add(title)
                deduped.append(lec)
            lectures = deduped

            known_processed = db.get_processed_sub_ids(course_id)
            if course_requires_blackboard(course_title):
                current_processed: set[str] = set()
                for sid in known_processed:
                    row = db.get_lecture(sid) or {}
                    has_board = bool(row.get("blackboard_latex"))
                    has_current_notes = str(
                        row.get("summary_model") or ""
                    ).startswith(BLACKBOARD_NOTES_MODEL_PREFIX)
                    if has_board and has_current_notes:
                        current_processed.add(sid)
                known_processed = current_processed

            new_lectures = [
                lec
                for lec in lectures
                if lec.get("has_playback")
                and str(lec["sub_id"]) not in known_processed
            ]

            unprocessed = db.get_unprocessed_lectures(course_id)
            new_ids = {str(lec["sub_id"]) for lec in new_lectures}
            retry_only = [
                {
                    "sub_id": row["sub_id"],
                    "sub_title": row["sub_title"],
                    "date": row["date"],
                }
                for row in unprocessed
                if row["sub_id"] not in new_ids
            ]
            new_lectures.extend(retry_only)

            reporter.course_new_count(len(new_lectures))
            if not new_lectures:
                continue

            for lecture in new_lectures:
                sub_id = str(lecture["sub_id"])
                db.insert_lecture(
                    sub_id,
                    course_id,
                    lecture.get("sub_title", ""),
                    lecture.get("date", ""),
                )
                out.append((course_id, course_title, lecture))

        except Exception:
            reporter.course_enumeration_error(course_id)
            traceback.print_exc()

    return out


def _drive_lectures(
    client: ICourseClient,
    db: Database,
    scheduler: Scheduler,
    transcriber: Transcriber,
    summarizer: Summarizer,
    reporter: Reporter,
    all_lectures: list[tuple[str, str, dict]],
    email_items: list,
) -> None:
    """Run each queued lecture through LectureRunner."""
    if not all_lectures:
        return

    first_course, _, first_lec = all_lectures[0]
    scheduler.prefetch_lecture(
        client,
        first_course,
        str(first_lec["sub_id"]),
    )

    runner = LectureRunner(
        client,
        db,
        scheduler,
        transcriber,
        summarizer,
        reporter,
    )

    for index, (course_id, course_title, lecture) in enumerate(all_lectures):
        sub_id = str(lecture["sub_id"])
        next_info: tuple[str, str] | None = None

        if index + 1 < len(all_lectures):
            next_course, _, next_lec = all_lectures[index + 1]
            next_info = (next_course, str(next_lec["sub_id"]))

        _check_session(client)

        try:
            summary = runner.run(
                course_id,
                course_title,
                lecture,
                next_info=next_info,
            )
            if summary:
                email_items.append(
                    {
                        "sub_id": sub_id,
                        "course_title": course_title,
                        "sub_title": lecture.get("sub_title", sub_id),
                        "date": lecture.get("date", ""),
                        "summary": summary,
                    }
                )
        except Exception:
            reporter.lecture_error(sub_id)
            traceback.print_exc()
        finally:
            scheduler.image_cache.discard(sub_id)
            scheduler.audio_downloader.release(sub_id)


def _send_email(
    emailer: Emailer | None,
    db: Database,
    reporter: Reporter,
    email_items: list,
) -> None:
    """Append unsent processed lectures, then send."""
    unsent = db.get_unsent_lectures()
    if unsent:
        seen_sub_ids = {item["sub_id"] for item in email_items}
        for row in unsent:
            if row["sub_id"] not in seen_sub_ids:
                email_items.append(
                    {
                        "sub_id": row["sub_id"],
                        "course_title": row["course_title"],
                        "sub_title": row["sub_title"],
                        "date": row["date"],
                        "summary": row["summary"],
                    }
                )
        reporter.email_recovered_unsent(len(unsent))

    if not (emailer and email_items):
        return

    try:
        reporter.email_summary(len(email_items))
        if emailer.send(email_items):
            db.mark_emailed_batch([item["sub_id"] for item in email_items])
        else:
            reporter.email_failed()
    except Exception:
        reporter.info("[Email] Failed to send:")
        traceback.print_exc()


def _crawl_semester_catalog(
    client: ICourseClient,
    db: Database,
    reporter: Reporter,
) -> None:
    """Auto-discover semesters and refresh all_courses."""
    reporter.info("Discovering available semesters from API...")

    try:
        _check_session(client)
        terms = client.discover_terms()
    except Exception as exc:
        reporter.crawl_courses_failed("discovery", exc)
        return

    if not terms:
        reporter.info("No semesters found via API discovery.")
        return

    reporter.info(
        f"Found {len(terms)} semester(s): "
        f"{', '.join(term['name'] for term in terms)}"
    )

    for term_info in terms:
        code = term_info["code"]
        name = term_info["name"]
        expected = term_info["count"]
        reporter.crawl_courses_start(name)
        t0 = time.time()

        try:
            _check_session(client)
            rows = client.list_semester_courses(code)
            if not rows:
                reporter.info(
                    f"  Term {name}: API returned 0 courses, skipping."
                )
                continue

            deleted, upserted = db.upsert_all_courses_for_term(name, rows)
            reporter.crawl_courses_done(
                name,
                len(rows),
                deleted,
                upserted,
                time.time() - t0,
            )
        except Exception as exc:
            reporter.crawl_courses_failed(name, exc)
            continue

        reporter.info(
            f"  ({code}) → {expected} API courses, {len(rows)} fetched"
        )

    reporter.info("Semester catalog crawl complete.")


def run():
    """Single execution of the full pipeline."""
    reporter = Reporter()
    reporter.run_header()

    if not config.COURSE_IDS and not config.CRAWL_TERM:
        reporter.info(
            "No COURSE_IDS configured. Set COURSE_IDS to process lectures "
            "or leave empty for crawl-only mode."
        )

    db = Database()
    corrected = db.sync_dates_from_sub()
    if corrected:
        print(
            f"  [Date] Synced {corrected} lecture date(s) from sub_title",
            flush=True,
        )

    transcriber = Transcriber()
    summarizer = Summarizer() if config.COURSE_IDS else None
    emailer = (
        Emailer()
        if config.SMTP_EMAIL and config.SMTP_PASSWORD
        else None
    )

    vpn = login_with_retry()
    client = ICourseClient(vpn)
    email_items: list = []

    has_catalog = db.has_all_courses()
    today = datetime.datetime.now().day
    if not has_catalog or today in (5, 25):
        _crawl_semester_catalog(client, db, reporter)
    else:
        reporter.info(
            "Skipping catalog crawl (has data, not the 5th or 25th)."
        )

    if not config.COURSE_IDS:
        reporter.info(
            "\n[Crawl-only mode] No COURSE_IDS — skipping lectures."
        )
        reporter.run_footer()
        return

    scheduler = Scheduler(reporter=reporter)

    try:
        all_lectures = _enumerate_lectures(client, db, reporter)
        _drive_lectures(
            client,
            db,
            scheduler,
            transcriber,
            summarizer,
            reporter,
            all_lectures,
            email_items,
        )
    finally:
        scheduler.shutdown()

    _send_email(emailer, db, reporter, email_items)
    reporter.run_footer()


if __name__ == "__main__":
    run()
