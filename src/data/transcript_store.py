"""Persistence helpers for timestamped and AI-proofread lecture transcripts.

The original schema stored only one flattened ASR string. Traceable summaries
need the ASR time segments on later runs, and email transcript attachments need
a durable corrected transcript that can be recovered if email sending is retried.

These helpers intentionally use ``Database.conn`` directly so the large legacy
Database wrapper does not need a second set of near-duplicate accessors.
"""

from __future__ import annotations

import json
from datetime import datetime


# v2 aligns proofreading chunks to the same 3-minute windows used by summary
# provenance. Bumping the prefix prevents reuse of any earlier 4-minute cache.
PROOFREAD_MODEL_PREFIX = "proofread-transcript-v2/"
PROOFREAD_BOARD_MODEL_PREFIX = "proofread-transcript-v2-board/"


def _loads_segments(value: str | None) -> list[dict] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, list):
        return None

    cleaned: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start_ms = int(item.get("start_ms", 0))
            end_ms = int(item.get("end_ms", start_ms))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        cleaned.append(
            {"start_ms": start_ms, "end_ms": end_ms, "text": text}
        )
    return cleaned or None


def load_transcript_segments(db, sub_id: str) -> list[dict] | None:
    row = db.conn.execute(
        "SELECT transcript_segments_json FROM lectures WHERE sub_id = ?",
        (sub_id,),
    ).fetchone()
    if not row:
        return None
    return _loads_segments(row["transcript_segments_json"])


def save_transcript_segments(db, sub_id: str, segments: list[dict] | None) -> None:
    payload = json.dumps(segments or [], ensure_ascii=False, separators=(",", ":"))
    with db.conn:
        db.conn.execute(
            "UPDATE lectures SET transcript_segments_json = ? WHERE sub_id = ?",
            (payload, sub_id),
        )


def load_proofread(db, sub_id: str) -> tuple[str, list[dict] | None, str] | None:
    row = db.conn.execute(
        """SELECT proofread_transcript, proofread_segments_json, proofread_model
           FROM lectures WHERE sub_id = ?""",
        (sub_id,),
    ).fetchone()
    if not row or not row["proofread_transcript"]:
        return None
    return (
        str(row["proofread_transcript"]),
        _loads_segments(row["proofread_segments_json"]),
        str(row["proofread_model"] or ""),
    )


def save_proofread(
    db,
    sub_id: str,
    markdown: str,
    corrected_segments: list[dict],
    model: str,
) -> None:
    payload = json.dumps(
        corrected_segments or [], ensure_ascii=False, separators=(",", ":")
    )
    with db.conn:
        db.conn.execute(
            """UPDATE lectures
               SET proofread_transcript = ?,
                   proofread_segments_json = ?,
                   proofread_model = ?,
                   proofread_at = ?
               WHERE sub_id = ?""",
            (
                markdown,
                payload,
                model,
                datetime.now().isoformat(),
                sub_id,
            ),
        )


def proofread_is_current(model: str, *, uses_blackboard: bool) -> bool:
    expected = (
        PROOFREAD_BOARD_MODEL_PREFIX if uses_blackboard else PROOFREAD_MODEL_PREFIX
    )
    return str(model or "").startswith(expected)
