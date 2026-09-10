"""LectureRunner extension for proof-preserving blackboard notes.

For whitelisted mathematics courses, the expensive vision stage creates and
caches a chronological raw board transcription. The final student-facing notes
use LLM semantic de-duplication over that evidence. The global editor may add
short explanatory notes, but every model-authored addition is explicitly marked
as a purple ``AI 补充`` block. A strict local finalizer then repairs only concrete
rendering/transcription defects against the raw vision evidence before storage.
"""

from __future__ import annotations

from typing import Optional

from src.ai.blackboard_editor_annotated import BlackboardEditor
from src.ai.blackboard_vision import course_requires_blackboard
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner


_NOTES_MODEL_PREFIX = "blackboard-llm-editor-v6/"


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

        return str(existing.get("summary_model") or "").startswith(
            _NOTES_MODEL_PREFIX
        )

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
                    "blackboard transcription empty; no notes generated",
                )
                return None

            # Deliberately exclude transcript/transcript_segments here. The raw
            # board timeline remains the evidence source. Any explanatory model
            # additions are visually marked as AI supplements by the editor.
            editor = BlackboardEditor()
            notes, editor_model = editor.edit(blackboard_latex)
            if not notes.strip():
                self._reporter.info(
                    "    [FAIL] Blackboard editor produced empty output."
                )
                self._db.update_error(
                    sub_id,
                    "blackboard-editor",
                    "LLM transcription editor produced empty output",
                )
                return None

            model_used = (
                f"{_NOTES_MODEL_PREFIX}{editor_model}"
                f"|vision={blackboard_model or 'vision'}"
            )
            self._reporter.info(
                f"    [OK] Blackboard edited transcript: "
                f"{len(blackboard_latex)} raw chars -> {len(notes)} final chars; "
                "LLM semantic de-duplication + global faithfulness audit + "
                "strict local render/transcription preflight; optional model "
                "explanations are marked as purple AI supplements; "
                "audio transcript excluded"
            )
            self._db.update_summary(sub_id, notes, model_used)
            return notes
        except Exception as exc:
            self._reporter.info(
                f"    [FAIL] Blackboard editor error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._db.update_error(sub_id, "blackboard-editor", str(exc))
            raise
