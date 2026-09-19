"""Durable cache for evidence-constrained math-enhanced transcripts.

The project currently preserves selected SQLite ``meta`` prefixes when concurrent
workflow databases are merged. To avoid a schema migration while keeping the
new transcript durable, this cache intentionally uses the already-preserved
``blackboard_cache_blob:`` namespace with a more specific sub-prefix.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime


MATH_TRANSCRIPT_VERSION = 9
_KEY_PREFIX = "blackboard_cache_blob:math-transcript-v9:"
EDITORIAL_CHECKPOINT_VERSION = 1
_EDITORIAL_CHECKPOINT_KEY_PREFIX = (
    "blackboard_cache_blob:math-transcript-editorial-checkpoint-v1:"
)
FIRST_PASS_CHECKPOINT_VERSION = 1
_FIRST_PASS_CHECKPOINT_KEY_PREFIX = (
    "blackboard_cache_blob:math-transcript-first-pass-checkpoint-v1:"
)


def source_fingerprint(
    course_title: str,
    summary_reference: str,
    proofread_markdown: str,
    proofread_segments: list[dict] | None,
    ppt_pages: list[dict] | None,
    raw_blackboard: str,
) -> str:
    """Fingerprint exactly the evidence used to produce an enhanced transcript."""
    digest = hashlib.sha256()
    digest.update(str(course_title or "").encode("utf-8"))
    digest.update(b"\0summary-reference\0")
    digest.update(str(summary_reference or "").encode("utf-8"))
    digest.update(b"\0proofread\0")
    digest.update(str(proofread_markdown or "").encode("utf-8"))
    digest.update(b"\0segments\0")
    digest.update(
        json.dumps(
            proofread_segments or [],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0ppt\0")
    compact_pages = [
        {"created_sec": page.get("created_sec"), "text": page.get("text")}
        for page in (ppt_pages or [])
    ]
    digest.update(
        json.dumps(
            compact_pages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0board\0")
    digest.update(str(raw_blackboard or "").encode("utf-8"))
    return digest.hexdigest()


def _key(sub_id: str) -> str:
    return f"{_KEY_PREFIX}{sub_id}"


def _editorial_checkpoint_key(sub_id: str) -> str:
    return f"{_EDITORIAL_CHECKPOINT_KEY_PREFIX}{sub_id}"


def _first_pass_checkpoint_key(sub_id: str) -> str:
    return f"{_FIRST_PASS_CHECKPOINT_KEY_PREFIX}{sub_id}"


def load_first_pass_checkpoint(db, sub_id: str) -> dict | None:
    """Load resumable evidence-pass progress for one lecture."""
    raw = db.read_meta(_first_pass_checkpoint_key(str(sub_id)))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version") or 0) != FIRST_PASS_CHECKPOINT_VERSION:
        return None
    return payload


def save_first_pass_checkpoint(db, sub_id: str, payload: dict) -> None:
    """Atomically persist completed evidence-pass chunks in SQLite meta."""
    value = dict(payload)
    value["version"] = FIRST_PASS_CHECKPOINT_VERSION
    value["updated_at"] = datetime.now().isoformat()
    db.write_meta(
        _first_pass_checkpoint_key(str(sub_id)),
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    )


def clear_first_pass_checkpoint(db, sub_id: str) -> None:
    """Remove the evidence-pass checkpoint after the final cache succeeds."""
    db.write_meta(_first_pass_checkpoint_key(str(sub_id)), "")


def load_editorial_checkpoint(db, sub_id: str) -> dict | None:
    """Load resumable final-review progress for one lecture."""
    raw = db.read_meta(_editorial_checkpoint_key(str(sub_id)))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version") or 0) != EDITORIAL_CHECKPOINT_VERSION:
        return None
    return payload


def save_editorial_checkpoint(db, sub_id: str, payload: dict) -> None:
    """Atomically persist accepted final-review chunks in SQLite meta."""
    value = dict(payload)
    value["version"] = EDITORIAL_CHECKPOINT_VERSION
    value["updated_at"] = datetime.now().isoformat()
    db.write_meta(
        _editorial_checkpoint_key(str(sub_id)),
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    )


def clear_editorial_checkpoint(db, sub_id: str) -> None:
    """Remove progress only after the complete transcript has succeeded."""
    # Writing an empty value works with Database as well as lightweight test
    # doubles and makes the next load behave exactly like a missing key.
    db.write_meta(_editorial_checkpoint_key(str(sub_id)), "")


def _load_version(db, sub_id: str, version: int) -> dict | None:
    raw = db.read_meta(f"blackboard_cache_blob:math-transcript-v{version}:{sub_id}")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version") or 0) != version:
        return None
    if not str(payload.get("markdown") or "").strip():
        return None
    if not isinstance(payload.get("segments"), list):
        return None
    return payload


def load_math_transcript(db, sub_id: str) -> dict | None:
    return _load_version(db, str(sub_id), MATH_TRANSCRIPT_VERSION)


def load_first_pass_seed(db, sub_id: str) -> dict | None:
    """Reuse v7's fully generated evidence-aware transcript as v9 input.

    v8 may contain API fallbacks because its run exhausted the provider balance;
    v7 is the last successfully completed first-pass/visual cache.
    """
    return _load_version(db, str(sub_id), 7)


def save_math_transcript(
    db,
    sub_id: str,
    *,
    markdown: str,
    segments: list[dict],
    model: str,
    source_sha256: str,
) -> None:
    payload = {
        "version": MATH_TRANSCRIPT_VERSION,
        "markdown": markdown,
        "segments": segments,
        "model": model,
        "source_sha256": source_sha256,
        "updated_at": datetime.now().isoformat(),
    }
    db.write_meta(
        _key(str(sub_id)),
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def cache_matches(payload: dict | None, source_sha256: str) -> bool:
    return bool(
        payload
        and int(payload.get("version") or 0) == MATH_TRANSCRIPT_VERSION
        and str(payload.get("source_sha256") or "") == str(source_sha256)
        and str(payload.get("markdown") or "").strip()
        and isinstance(payload.get("segments"), list)
    )
