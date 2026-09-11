"""Resend processed lectures with a mandatory complete transcript appendix.

Stored proofread transcripts are reused. Historical rows that predate timed
proofreading are rebuilt from their persisted ASR text before PDF generation;
the original transcript is retained as a lossless fallback if the LLM is down.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

# Running ``python scripts/resend_processed.py`` makes ``scripts/`` the first
# import root, so the repository-level ``src`` package is otherwise invisible.
# Add the repository root explicitly before importing project modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api.emailer import Emailer
from src.ai.transcript_proofreader import TranscriptProofreader
from src.data.blackboard_store import get_blackboard
from src.data.database import Database
from src.data.transcript_store import (
    load_proofread,
    load_transcript_segments,
    save_proofread,
)


_TIME_RE = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")
_SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]+|\n+|$)", re.S)
_LEGACY_SEGMENT_CHARS = 160


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
            l.transcript,
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


def _time_seconds(match: re.Match) -> int:
    hours = int(match.group(1) or 0)
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def _estimated_duration(
    transcript: str,
    summary: str,
    ppt_pages: list[dict],
    raw_blackboard: str,
) -> int:
    """Estimate historical lecture duration from all persisted time evidence."""
    evidence = [
        _time_seconds(match)
        for text in (summary, raw_blackboard)
        for match in _TIME_RE.finditer(str(text or ""))
    ]
    evidence.extend(int(page.get("created_sec", 0) or 0) for page in ppt_pages)
    # Chinese classroom speech is normally several characters per second. This
    # lower bound prevents a long old transcript from being squeezed into the
    # first few visual windows when the historical duration itself was not saved.
    speech_floor = math.ceil(len(str(transcript or "")) / 3.2)
    return max(180, speech_floor, max(evidence, default=0))


def _reconstruct_timed_segments(transcript: str, duration_sec: int) -> list[dict]:
    """Build a lossless coarse timeline for legacy flattened ASR text."""
    sentences = [
        part.strip()
        for part in _SENTENCE_RE.findall(str(transcript or ""))
        if part.strip()
    ]
    parts = [
        sentence[offset:offset + _LEGACY_SEGMENT_CHARS].strip()
        for sentence in sentences
        for offset in range(0, len(sentence), _LEGACY_SEGMENT_CHARS)
        if sentence[offset:offset + _LEGACY_SEGMENT_CHARS].strip()
    ]
    if not parts:
        return []
    total = sum(len(part) for part in parts)
    elapsed = 0
    out: list[dict] = []
    for index, part in enumerate(parts):
        start_ms = round(duration_sec * 1000 * elapsed / total)
        elapsed += len(part)
        end_ms = (
            duration_sec * 1000
            if index == len(parts) - 1
            else round(duration_sec * 1000 * elapsed / total) - 1
        )
        out.append(
            {
                "start_ms": start_ms,
                "end_ms": max(start_ms + 1, end_ms),
                "text": part,
            }
        )
    return out


def _fmt_seconds(sec: int) -> str:
    hours, rem = divmod(max(0, int(sec)), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _raw_transcript_markdown(segments: list[dict]) -> str:
    """Produce a complete emergency appendix when LLM proofreading is unavailable."""
    lines = [
        "# 完整课堂语音转写（原始 ASR 回退）",
        "",
        "> 大模型校订本次不可用，以下完整保留数据库中的原始语音转写，未作删减。",
        "",
    ]
    for segment in segments:
        start = int(segment.get("start_ms", 0) or 0) // 1000
        end = int(segment.get("end_ms", 0) or 0) // 1000
        lines.extend(
            [
                f"### {_fmt_seconds(start)}–{_fmt_seconds(end)}",
                "",
                str(segment.get("text") or "").strip(),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _ensure_proofread(
    db: Database,
    row: dict,
    proofreader: TranscriptProofreader | None,
) -> tuple[str, list[dict], TranscriptProofreader | None]:
    """Load or rebuild a complete proofread transcript for historical rows."""
    sub_id = str(row["sub_id"])
    cached = load_proofread(db, sub_id)
    if cached and cached[0] and cached[1]:
        return str(cached[0]), list(cached[1]), proofreader

    raw_transcript = str(row.get("transcript") or "").strip()
    segments = load_transcript_segments(db, sub_id) or []
    pages = db.get_done_ppt_pages(sub_id)
    board_cached = get_blackboard(db, sub_id)
    raw_blackboard = board_cached[0] if board_cached else ""
    if not segments and raw_transcript:
        duration = _estimated_duration(
            raw_transcript,
            str(row.get("summary") or ""),
            pages,
            raw_blackboard,
        )
        segments = _reconstruct_timed_segments(raw_transcript, duration)
        print(
            f"[Resend] Reconstructed {len(segments)} coarse ASR segment(s) "
            f"across {duration}s for historical lecture {sub_id}.",
            flush=True,
        )
    if not segments:
        raise RuntimeError(
            f"historical lecture {sub_id} has no persisted ASR text or timing segments; "
            "refusing to send a PDF without the required transcript appendix"
        )

    fallback_markdown = _raw_transcript_markdown(segments)
    try:
        if proofreader is None:
            proofreader = TranscriptProofreader()
        result = proofreader.proofread(
            segments,
            pages,
            raw_blackboard=raw_blackboard,
        )
        save_proofread(
            db,
            sub_id,
            result.markdown,
            result.segments,
            result.model_label,
        )
        print(
            f"[Resend] Rebuilt AI-proofread transcript for {sub_id}: "
            f"{len(result.segments)} timed chunk(s).",
            flush=True,
        )
        return result.markdown, result.segments, proofreader
    except Exception as exc:
        print(
            f"[Resend] Proofreading unavailable for {sub_id}; preserving full raw ASR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return fallback_markdown, segments, proofreader


def _email_item(
    db: Database,
    row: dict,
    proofreader: TranscriptProofreader | None,
) -> tuple[dict, TranscriptProofreader | None]:
    course_title = str(row.get("course_title") or row.get("course_id") or "课程")
    sub_title = str(row.get("sub_title") or row.get("sub_id") or "课堂")
    item = {
        "sub_id": str(row["sub_id"]),
        "course_title": course_title,
        "sub_title": sub_title,
        "date": str(row.get("date") or ""),
        "summary": str(row.get("summary") or ""),
    }

    transcript_md, timed_chunks, proofreader = _ensure_proofread(
        db, row, proofreader
    )
    if transcript_md and timed_chunks:
        item["transcript_attachment"] = transcript_md
        item["transcript_filename"] = (
            f"{course_title}-{sub_title}-AI校订语音转写.md"
        )
    return item, proofreader


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
    proofreader: TranscriptProofreader | None = None

    # One lecture per message keeps each PDF/transcript pair independently
    # deliverable and avoids recreating the oversized historical MIME messages.
    for index, row in enumerate(rows, start=1):
        item, proofreader = _email_item(db, row, proofreader)
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
