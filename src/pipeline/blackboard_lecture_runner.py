"""LectureRunner extension for proof-preserving blackboard LaTeX notes.

For whitelisted mathematics courses, the final deliverable is a faithful
transcription-derived LaTeX notebook, not an LLM summary.  The vision pipeline
transcribes the board frame by frame; a deterministic compiler removes only
nearby repeated lines while preserving changed/new proof steps verbatim.
"""

from __future__ import annotations

from typing import Optional

from src.ai.blackboard_notes import compile_blackboard_notes
from src.ai.blackboard_vision import course_requires_blackboard
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner


_NOTES_MODEL_PREFIX = "blackboard-latex-notes-v1/"


class BlackboardLectureRunner(BaseLectureRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_course_id = ""
        self._active_course_title = ""

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        self._active_course_id = str(course_id)
        self._active_course_title = course_title or ""
        return super().run(course_id, course_title, lecture, next_info=next_info)

    def _has_summary(self, existing: dict | None) -> bool:
        if not existing or not existing.get("summary"):
            return False
        if not course_requires_blackboard(self._active_course_title):
            return True

        sub_id = str(existing.get("sub_id") or "")
        if not sub_id or get_blackboard(self._db, sub_id) is None:
            return False

        # Old AI-generated summaries are not valid outputs for a blackboard
        # course. Only a summary field explicitly produced by the deterministic
        # LaTeX-note compiler counts as complete.
        return str(existing.get("summary_model") or "").startswith(_NOTES_MODEL_PREFIX)

    def _ensure_blackboard(self, sub_id: str, course_title: str) -> tuple[str, str]:
        cached = get_blackboard(self._db, sub_id)
        if cached:
            markdown, model = cached
            self._reporter.info(
                f"    [Blackboard] current cache exists ({len(markdown)} chars), reusing."
            )
            return markdown, model

        self._reporter.info(
            "    [Blackboard] cache missing/stale; regenerating proof-preserving board timeline."
        )
        pipeline = BlackboardPipeline(self._client, self._reporter)
        markdown, model = pipeline.run(
            self._active_course_id, course_title, sub_id
        )
        if markdown.strip():
            save_blackboard(self._db, sub_id, markdown, model)
        return markdown, model

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        """Create final output.

        For blackboard courses this method intentionally does not call the
        general-purpose summarizer.  It converts cached frame transcriptions
        directly into chronological LaTeX notes.  Other courses retain the
        repository's original summary behavior.
        """
        if not course_requires_blackboard(course_title):
            return super()._summarize(
                sub_id, course_title, transcript, transcript_segments
            )

        try:
            blackboard_latex, blackboard_model = self._ensure_blackboard(
                sub_id, course_title
            )
            if not blackboard_latex.strip():
                self._reporter.info(
                    "    [FAIL] Blackboard transcription is empty; refusing to invent notes."
                )
                self._db.update_error(
                    sub_id,
                    "blackboard",
                    "blackboard transcription empty; no LaTeX notes generated",
                )
                return None

            notes = compile_blackboard_notes(blackboard_latex)
            if not notes.strip():
                self._reporter.info(
                    "    [FAIL] Blackboard note compiler produced empty output."
                )
                self._db.update_error(
                    sub_id,
                    "blackboard-notes",
                    "deterministic LaTeX note compiler produced empty output",
                )
                return None

            model_used = f"{_NOTES_MODEL_PREFIX}{blackboard_model or 'vision'}"
            self._reporter.info(
                f"    [OK] Blackboard LaTeX notes: {len(blackboard_latex)} raw chars "
                f"-> {len(notes)} note chars; no LLM summarization"
            )
            self._db.update_summary(sub_id, notes, model_used)
            return notes
        except Exception as exc:
            self._reporter.info(
                f"    [FAIL] Blackboard LaTeX note generation error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._db.update_error(sub_id, "blackboard-notes", str(exc))
            raise
