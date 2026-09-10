"""LectureRunner extension for proof-preserving blackboard notes.

Functional Analysis keeps its proof-preserving blackboard section.  Its appended
normal AI summary now uses the same timestamped, AI-proofread ASR evidence and
validated video provenance as every other course.  The proofreader additionally
receives timestamp-aligned raw blackboard frames as terminology evidence.
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


_NOTES_MODEL_PREFIX = "blackboard-llm-editor-v8/"

_AI_SUMMARY_SEPARATOR = """\
---

### AI 课程总结（带视频定位）

> 以下部分根据 AI 校订后的录音转写与 PPT OCR 生成；每个知识点的视频时段由真实 ASR 时间轴验证。它与上方老师板书转写及紫色 AI 补充相互独立。
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
        if not course_requires_blackboard(self._active_course_title):
            return super()._has_summary(existing)
        if not existing or not existing.get("summary"):
            return False

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

    def _generate_traceable_ai_summary(
        self,
        sub_id: str,
        course_title: str,
        corrected_segments: list[dict],
        kept_pages: list[dict],
    ) -> tuple[str, str]:
        corrected_flat = " ".join(
            str(seg.get("text") or "").strip()
            for seg in corrected_segments
            if str(seg.get("text") or "").strip()
        )
        prompt_text, mode, video_windows = bucketer.assemble_traceable(
            corrected_flat,
            corrected_segments,
            kept_pages,
        )
        if not video_windows:
            raise RuntimeError("Functional Analysis summary has no video windows")

        self._reporter.info(
            f"    [Time] Generating appended traceable AI summary at "
            f"{time.strftime('%H:%M:%S')}"
            f" — mode={mode}, prompt={len(prompt_text)} chars, "
            f"windows={len(video_windows)}"
        )
        summary, model_used = self._summarizer.summarize(
            course_title,
            prompt_text,
            video_windows=video_windows,
        )
        if not summary or not summary.strip():
            raise RuntimeError("traceable audio+PPT summarizer returned empty output")
        self._reporter.info(
            f"    [OK] Appended traceable AI summary by {model_used}: "
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
                "strict local render/transcription preflight"
            )

            # The attachment and appended summary use the speech stream, but the
            # transcript proofreader may consult timestamp-aligned raw board
            # frames to correct mathematical terms. The board itself is never
            # copied into the speech transcript unless the lecturer said it.
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            _, corrected_segments, proofread_model = self._get_proofread_transcript(
                sub_id,
                transcript_segments,
                kept_pages,
                raw_blackboard=blackboard_latex,
            )
            ai_summary, ai_summary_model = self._generate_traceable_ai_summary(
                sub_id,
                course_title,
                corrected_segments,
                kept_pages,
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
                f"|proofread={proofread_model}"
                f"|summary={ai_summary_model}"
            )
            self._reporter.info(
                f"    [OK] Combined Functional Analysis notes: "
                f"board={len(board_notes)} chars + "
                f"traceable-summary={len(ai_summary)} chars -> "
                f"{len(combined)} chars total; one email section + transcript attachment"
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
