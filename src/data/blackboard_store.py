"""Persistence helpers for blackboard transcriptions and resumable checkpoints."""

from __future__ import annotations

import json
from datetime import datetime


# Increment this whenever extraction/selection/transcription semantics change.
# Old cached board text is then ignored and regenerated automatically.
BLACKBOARD_CACHE_VERSION = 2
_CHECKPOINT_KEY_PREFIX = "blackboard_checkpoint:"
_CACHE_VERSION_KEY_PREFIX = "blackboard_cache_version:"


def _checkpoint_key(sub_id: str) -> str:
    return f"{_CHECKPOINT_KEY_PREFIX}{sub_id}"


def _cache_version_key(sub_id: str) -> str:
    return f"{_CACHE_VERSION_KEY_PREFIX}{sub_id}"


def get_blackboard(db, sub_id: str) -> tuple[str, str] | None:
    """Return cached board text only when produced by the current algorithm."""
    version_row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        (_cache_version_key(str(sub_id)),),
    ).fetchone()
    try:
        cached_version = int(version_row["value"]) if version_row else 0
    except (TypeError, ValueError):
        cached_version = 0
    if cached_version != BLACKBOARD_CACHE_VERSION:
        return None

    row = db.conn.execute(
        "SELECT blackboard_latex, blackboard_model FROM lectures WHERE sub_id = ?",
        (str(sub_id),),
    ).fetchone()
    if not row or not row["blackboard_latex"]:
        return None
    return row["blackboard_latex"], row["blackboard_model"] or ""


def save_blackboard(db, sub_id: str, markdown: str, model: str) -> None:
    """Persist the final chronological Markdown+LaTeX transcription."""
    with db._lock, db.conn:
        db.conn.execute(
            """UPDATE lectures
               SET blackboard_latex = ?, blackboard_model = ?, blackboard_at = ?
               WHERE sub_id = ?""",
            (markdown, model, datetime.now().isoformat(), str(sub_id)),
        )
        db.conn.execute(
            """INSERT INTO meta(key, value) VALUES(?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (_cache_version_key(str(sub_id)), str(BLACKBOARD_CACHE_VERSION)),
        )
        db.conn.execute(
            "DELETE FROM meta WHERE key = ?",
            (_checkpoint_key(str(sub_id)),),
        )


def load_blackboard_checkpoint(db, sub_id: str, sample_sec: int) -> list[dict]:
    """Load successful per-frame vision results from a previous interrupted run."""
    row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        (_checkpoint_key(str(sub_id)),),
    ).fetchone()
    if not row or not row["value"]:
        return []
    try:
        payload = json.loads(row["value"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if int(payload.get("version", -1)) != BLACKBOARD_CACHE_VERSION:
        return []
    if int(payload.get("sample_sec", -1)) != int(sample_sec):
        return []
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def save_blackboard_checkpoint(db, sub_id: str, sample_sec: int, results: list[dict]) -> None:
    """Persist successful frame results after each vision batch."""
    payload = json.dumps(
        {
            "version": BLACKBOARD_CACHE_VERSION,
            "sample_sec": int(sample_sec),
            "updated_at": datetime.now().isoformat(),
            "results": results,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with db._lock, db.conn:
        db.conn.execute(
            """INSERT INTO meta(key, value) VALUES(?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (_checkpoint_key(str(sub_id)), payload),
        )


def clear_blackboard_checkpoint(db, sub_id: str) -> None:
    with db._lock, db.conn:
        db.conn.execute(
            "DELETE FROM meta WHERE key = ?",
            (_checkpoint_key(str(sub_id)),),
        )
