"""Persistence helpers for required blackboard transcriptions."""

from __future__ import annotations

from datetime import datetime


def get_blackboard(db, sub_id: str) -> tuple[str, str] | None:
    row = db.conn.execute(
        "SELECT blackboard_latex, blackboard_model FROM lectures WHERE sub_id = ?",
        (str(sub_id),),
    ).fetchone()
    if not row or not row["blackboard_latex"]:
        return None
    return row["blackboard_latex"], row["blackboard_model"] or ""


def save_blackboard(db, sub_id: str, markdown: str, model: str) -> None:
    """Persist the raw chronological Markdown+LaTeX transcription."""
    with db._lock, db.conn:
        db.conn.execute(
            """UPDATE lectures
               SET blackboard_latex = ?, blackboard_model = ?, blackboard_at = ?
               WHERE sub_id = ?""",
            (markdown, model, datetime.now().isoformat(), str(sub_id)),
        )
