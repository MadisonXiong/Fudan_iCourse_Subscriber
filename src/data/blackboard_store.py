"""Persistence helpers for blackboard transcriptions and resumable checkpoints."""

from __future__ import annotations

import json
from datetime import datetime


BLACKBOARD_CACHE_VERSION = 2
_CHECKPOINT_KEY_PREFIX = "blackboard_checkpoint:"
_CACHE_VERSION_KEY_PREFIX = "blackboard_cache_version:"
_CACHE_BLOB_KEY_PREFIX = "blackboard_cache_blob:"
_BLACKBOARD_MARKER = "### 黑板板书 LaTeX 转写"


def _checkpoint_key(sub_id: str) -> str:
    return f"{_CHECKPOINT_KEY_PREFIX}{sub_id}"


def _cache_version_key(sub_id: str) -> str:
    return f"{_CACHE_VERSION_KEY_PREFIX}{sub_id}"


def _cache_blob_key(sub_id: str) -> str:
    return f"{_CACHE_BLOB_KEY_PREFIX}{sub_id}"


def _save_blob(db, sid: str, markdown: str, model: str, updated_at: str) -> None:
    payload = json.dumps(
        {
            "version": BLACKBOARD_CACHE_VERSION,
            "markdown": markdown,
            "model": model or "",
            "updated_at": updated_at,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    db.conn.execute(
        """INSERT INTO meta(key, value) VALUES(?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        (_cache_blob_key(sid), payload),
    )


def _load_blob(db, sid: str) -> tuple[str, str] | None:
    row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        (_cache_blob_key(sid),),
    ).fetchone()
    if not row or not row["value"]:
        return None
    try:
        payload = json.loads(row["value"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    try:
        version = int(payload.get("version", 0))
    except (TypeError, ValueError):
        version = 0
    markdown = str(payload.get("markdown") or "")
    if version != BLACKBOARD_CACHE_VERSION or not markdown.startswith(_BLACKBOARD_MARKER):
        return None
    return markdown, str(payload.get("model") or "")


def _backfill_blob_from_lecture(db, sid: str, markdown: str, model: str, updated_at: str) -> None:
    """Mirror a valid legacy lecture-column cache into durable meta storage."""
    with db._lock, db.conn:
        _save_blob(db, sid, markdown, model, updated_at or datetime.now().isoformat())
        db.conn.execute(
            """INSERT INTO meta(key, value) VALUES(?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (_cache_version_key(sid), str(BLACKBOARD_CACHE_VERSION)),
        )
    print(
        f"[Blackboard] mirrored legacy raw cache for {sid} into durable meta storage",
        flush=True,
    )


def _recover_shifted_blackboard_row(db, sub_id: str) -> tuple[str, str] | None:
    sid = str(sub_id)
    row = db.conn.execute(
        """SELECT blackboard_latex, blackboard_model, blackboard_at,
                  emailed_at, error_msg, error_count
           FROM lectures WHERE sub_id = ?""",
        (sid,),
    ).fetchone()
    if not row or row["blackboard_latex"]:
        return None

    misplaced = str(row["emailed_at"] or "")
    if not misplaced.startswith(_BLACKBOARD_MARKER) or len(misplaced) < 10000:
        return None

    recovered_model = str(row["error_msg"] or "")
    recovered_at = str(row["error_count"] or "")
    if "T" not in recovered_at and "-" not in recovered_at:
        recovered_at = datetime.now().isoformat()

    with db._lock, db.conn:
        db.conn.execute(
            """UPDATE lectures
               SET blackboard_latex = ?,
                   blackboard_model = ?,
                   blackboard_at = ?,
                   emailed_at = NULL
               WHERE sub_id = ?""",
            (misplaced, recovered_model, recovered_at, sid),
        )
        db.conn.execute(
            """INSERT INTO meta(key, value) VALUES(?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (_cache_version_key(sid), str(BLACKBOARD_CACHE_VERSION)),
        )
        _save_blob(db, sid, misplaced, recovered_model, recovered_at)

    print(
        f"[Blackboard] recovered shifted raw cache for {sid}: "
        f"{len(misplaced)} chars; no vision regeneration required",
        flush=True,
    )
    return misplaced, recovered_model


def get_blackboard(db, sub_id: str) -> tuple[str, str] | None:
    """Return cached board text, preferring the schema-independent meta blob.

    ``blackboard_cache_blob:<sub_id>`` is authoritative for durable reuse.
    Legacy lecture-column caches remain supported and are automatically mirrored
    into meta on first successful read.  This protects the expensive raw vision
    transcript from future ``lectures`` schema migrations and shard rebuilds.
    """
    sid = str(sub_id)

    durable = _load_blob(db, sid)
    if durable:
        return durable

    row = db.conn.execute(
        "SELECT blackboard_latex, blackboard_model, blackboard_at "
        "FROM lectures WHERE sub_id = ?",
        (sid,),
    ).fetchone()
    if not row or not row["blackboard_latex"]:
        recovered = _recover_shifted_blackboard_row(db, sid)
        if recovered:
            return recovered
        return None

    markdown = str(row["blackboard_latex"] or "")
    model = str(row["blackboard_model"] or "")
    updated_at = str(row["blackboard_at"] or "")
    if not markdown.startswith(_BLACKBOARD_MARKER):
        return None

    version_row = db.conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        (_cache_version_key(sid),),
    ).fetchone()
    try:
        cached_version = int(version_row["value"]) if version_row else 0
    except (TypeError, ValueError):
        cached_version = 0

    if cached_version == BLACKBOARD_CACHE_VERSION:
        _backfill_blob_from_lecture(db, sid, markdown, model, updated_at)
        return markdown, model

    if cached_version == 0 and updated_at:
        _backfill_blob_from_lecture(db, sid, markdown, model, updated_at)
        print(
            f"[Blackboard] recovered existing cache for {sid}; "
            "restored missing cache-version metadata",
            flush=True,
        )
        return markdown, model

    return None


def save_blackboard(db, sub_id: str, markdown: str, model: str) -> None:
    """Persist raw board text twice: lecture columns + durable meta blob."""
    sid = str(sub_id)
    now = datetime.now().isoformat()
    with db._lock, db.conn:
        db.conn.execute(
            """UPDATE lectures
               SET blackboard_latex = ?, blackboard_model = ?, blackboard_at = ?
               WHERE sub_id = ?""",
            (markdown, model, now, sid),
        )
        db.conn.execute(
            """INSERT INTO meta(key, value) VALUES(?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (_cache_version_key(sid), str(BLACKBOARD_CACHE_VERSION)),
        )
        _save_blob(db, sid, markdown, model, now)
        db.conn.execute(
            "DELETE FROM meta WHERE key = ?",
            (_checkpoint_key(sid),),
        )


def load_blackboard_checkpoint(db, sub_id: str, sample_sec: int) -> list[dict]:
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
