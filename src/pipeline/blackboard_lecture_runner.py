"""LectureRunner extension for proof-preserving blackboard notes.

For whitelisted mathematics courses, the expensive vision stage creates and
caches a chronological raw board transcription. The student-facing output first
contains the proof-preserving blackboard transcription/editing pipeline, with
model-authored explanations visibly marked as purple ``AI 补充`` blocks. A
strict local finalizer repairs concrete rendering/transcription defects against
the raw vision evidence. After that complete blackboard section, the repository's
original audio+PPT AI summary is generated with the normal bucketer/Summarizer
pipeline and appended as a clearly separated final section.
"""

from __future__ import annotations

import time
from typing import Optional

from src.ai import bucketer
from src.ai.blackboard_editor_annotated import BlackboardEditor
from src.ai.blackboard_vision import course_requires_blackboard
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner


_NOTES_MODEL_PREFIX = "blackboard-llm-editor-v7/"

_AI_SUMMARY_SEPARATOR = """\
---

### AI 课程总结

> 以下部分使用原有课程总结流程，根据录音转写与 PPT OCR 生成；它与上方的老师板书转写及紫色 AI 补充相互独立。
""".strip()


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

    def _generate_original_ai_summary(
        self,
        sub_id: str,
        course_title: str,
        transcript: str,
        transcript_segments: list[dict] | None,
    ) -> tuple[str, str]:
        """Run exactly the repository's normal audio+PPT summary path.

        This intentionally does *not* feed the blackboard notes into the normal
        summarizer. The appended section therefore remains comparable with what
        a non-blackboard course receives: ASR + PPT OCR assembled by bucketer,
        then ``Summarizer.summarize`` with the existing system prompt.
        """
        kept_pages = self._db.get_done_ppt_pages(sub_id)
        prompt_text, mode = bucketer.assemble(
            transcript,
            transcript_segments,
            kept_pages,
        )
        self._reporter.info(
            f"    [Time] Generating appended original AI summary at "
            f"{time.strftime('%H:%M:%S')}"
            f" — mode={mode}, prompt={len(prompt_text)} chars"
        )
        summary, model_used = self._summarizer.summarize(
            course_title,
            prompt_text,
        )
        if not summary or not summary.strip():
            raise RuntimeError("original audio+PPT summarizer returned empty output")
        self._reporter.info(
            f"    [OK] Appended original AI summary by {model_used}: "
            f"{len(summary)} chars"
        )
        return summary.strip(), model_used

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

            # The blackboard editor's evidence source remains the raw board only.
            # Audio/PPT are used later by a completely separate summary call.
            editor = BlackboardEditor()
            board_notes, editor_model = editor.edit(blackboard_latex)
            if not board_notes.strip():
                self._reporter.info(
                    "    [FAIL] Blackboard editor produced empty output."
                )
                self._db.update_error(
                    sub_id,
                    "blackboard-editor",
                    "LLM transcription editor produced empty output",
                )
                return None

            self._reporter.info(
                f"    [OK] Blackboard edited transcript: "
                f"{len(blackboard_latex)} raw chars -> {len(board_notes)} final chars; "
                "LLM semantic de-duplication + global faithfulness audit + "
                "strict local render/transcription preflight; optional model "
                "explanations are marked as purple AI supplements"
            )

            # Append the *original* normal course summary after the completed
            # board section. Keeping the two model calls independent avoids
            # contaminating the proof-preserving board evidence with ASR noise.
            ai_summary, ai_summary_model = self._generate_original_ai_summary(
                sub_id,
                course_title,
                transcript,
                transcript_segments,
            )
            combined = (
                board_notes.rstrip()
                + "\n\n"
                + _AI_SUMMARY_SEPARATOR
                + "\n\n"
                + ai_summary
            )

            model_used = (
                f"{_NOTES_MODEL_PREFIX}{editor_model}"
                f"|vision={blackboard_model or 'vision'}"
                f"|summary={ai_summary_model}"
            )
            self._reporter.info(
                f"    [OK] Combined Functional Analysis notes: "
                f"board={len(board_notes)} chars + "
                f"AI-summary={len(ai_summary)} chars -> "
                f"{len(combined)} chars total; one email section"
            )
            self._db.update_summary(sub_id, combined, model_used)
            return combined
        except Exception as exc:
            self._reporter.info(
                f"    [FAIL] Blackboard/AI-summary pipeline error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._db.update_error(sub_id, "blackboard-editor", str(exc))
            raise
