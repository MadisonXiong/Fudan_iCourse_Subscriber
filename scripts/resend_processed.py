"""Resend already-processed lecture emails without invoking ASR/OCR/LLMs.

This script is deliberately read-only with respect to lecture processing state.
It loads stored summaries and durable proofread transcript attachments from the
SQLite database and sends them again to the current RECEIVER_EMAIL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Running ``python scripts/resend_processed.py`` makes ``scripts/`` the first
# import root, so the repository-level ``src`` package is otherwise invisible.
# Add the repository root explicitly before importing project modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api.emailer import Emailer
from src.data.database import Database
from src.data.transcript_store import load_proofread


def _selected_course_ids() -> set[str]:
    raw = os.environ.get("RESEND_COURSE_IDS", "").strip()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _rows(db: Database, course_ids: set[str]) -> list[dict]:
    query = """
        SELECT
            l.sub_id,
            l.course_id,
            l.sub_title,
            l.date,
            l.summary,
            c.title AS course_title
        FROM lectures AS l
        LEFT JOIN courses AS c ON c.course_id = l.course_id
        WHERE l.processed_at IS NOT NULL
          AND l.summary IS NOT NULL
          AND TRIM(l.summary) <> ''
    """
    params: list[str] = []
    if course_ids:
        placeholders = ",".join("?" for _ in course_ids)
        query += f" AND l.course_id IN ({placeholders})"
        params.extend(sorted(course_ids))
    query += " ORDER BY l.date, l.course_id, l.sub_id"
    return [dict(row) for row in db.conn.execute(query, params).fetchall()]


def _email_item(db: Database, row: dict) -> dict:
    course_title = str(row.get("course_title") or row.get("course_id") or "课程")
    sub_title = str(row.get("sub_title") or row.get("sub_id") or "课堂")
    item = {
        "sub_id": str(row["sub_id"]),
        "course_title": course_title,
        "sub_title": sub_title,
        "date": str(row.get("date") or ""),
        "summary": str(row.get("summary") or ""),
    }

    proofread = load_proofread(db, str(row["sub_id"]))
    if proofread:
        transcript_md, timed_chunks, _ = proofread
        if transcript_md and timed_chunks:
            item["transcript_attachment"] = transcript_md
            item["transcript_filename"] = (
                f"{course_title}-{sub_title}-AI校订语音转写.md"
            )
    return item


def main() -> int:
    db = Database()
    course_ids = _selected_course_ids()
    rows = _rows(db, course_ids)

    scope = ",".join(sorted(course_ids)) if course_ids else "ALL processed courses"
    print(f"[Resend] Scope: {scope}")
    print(f"[Resend] Found {len(rows)} processed lecture(s) with stored summaries.")
    if not rows:
        return 0

    emailer = Emailer()
    failures = 0
    attachment_count = 0

    # Send one lecture per message.  Large math-heavy lectures can contain
    # hundreds of inline CID images; batching every historical lecture into a
    # single MIME message can exceed provider/client size limits.
    for index, row in enumerate(rows, start=1):
        item = _email_item(db, row)
        if item.get("transcript_attachment"):
            attachment_count += 1
        print(
            f"[Resend] {index}/{len(rows)}: "
            f"{item['course_title']} — {item['sub_title']}"
        )
        if not emailer.send([item]):
            failures += 1
            print(f"[Resend] FAILED: {item['sub_id']}", file=sys.stderr)

    print(
        f"[Resend] Complete: sent={len(rows) - failures}, "
        f"failed={failures}, transcript_attachments={attachment_count}."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
