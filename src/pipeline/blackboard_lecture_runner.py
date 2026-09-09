"""LectureRunner extension that makes blackboard LaTeX a hard requirement.

All non-whitelisted courses preserve the original pipeline unchanged.  For a
whitelisted course (default: 泛函分析), a lecture is considered complete only
when a raw blackboard Markdown+LaTeX transcription exists.  That raw evidence
is fed into the normal summarizer and also appended verbatim to the final
summary so the email/export always contains it.
"""

from __future__ import annotations

from typing import Optional

from src.ai import bucketer
from src.ai.blackboard_vision import (
    BLACKBOARD_SUMMARY_MARKER,
    course_requires_blackboard,
)
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner


class BlackboardLectureRunner(BaseLectureRunner):
    """Original lecture state machine plus required blackboard transcription."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_course_id = ""
        self._active_course_title = ""

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        self._active_course_id = str(course_id)
        self._active_course_title = course_title or ""
        return super().run(
            course_id, course_title, lecture, next_info=next_info,
        )

    def _has_summary(self, existing: dict | None) -> bool:
        """A required course is not done until blackboard evidence exists."""
        if not existing or not existing.get("summary"):
            return False
        if not course_requires_blackboard(self._active_course_title):
            return True
        return bool(existing.get("blackboard_latex"))

    def _ensure_blackboard(self, sub_id: str, course_title: str) -> tuple[str, str]:
        cached = get_blackboard(self._db, sub_id)
        if cached:
            markdown, model = cached
            self._reporter.info(
                f"    [Blackboard] cached transcription exists "
                f"({len(markdown)} chars), reusing."
            )
            return markdown, model

        try:
            pipeline = BlackboardPipeline(self._client, self._reporter)
            markdown, model = pipeline.run(
                self._active_course_id, course_title, sub_id,
            )
            save_blackboard(self._db, sub_id, markdown, model)
            return markdown, model
        except Exception as exc:
            self._reporter.info(
                f"    [FAIL] Blackboard transcription error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._db.update_error(sub_id, "blackboard", str(exc))
            raise

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        """Inject blackboard evidence into the LLM prompt when required."""
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(
                transcript, transcript_segments, kept_pages,
            )

            blackboard_latex = ""
            blackboard_model = ""
            if course_requires_blackboard(course_title):
                blackboard_latex, blackboard_model = self._ensure_blackboard(
                    sub_id, course_title,
                )
                prompt_text = (
                    f"{prompt_text}\n\n"
                    "【黑板板书（按时间转写；数学公式已转为 LaTeX）】\n"
                    f"{blackboard_latex}\n\n"
                    "整合笔记时，数学公式和黑板上实际写出的推导优先参考以上板书；"
                    "不得擅自补全 [unclear] 或录像中不可见的推导。"
                ).strip()

            self._reporter.info(
                f"    [Time] Generating summary — mode={mode}, "
                f"prompt={len(prompt_text)} chars"
            )
            summary, model_used = self._summarizer.summarize(
                course_title, prompt_text,
            )

            # Preserve raw board evidence verbatim.  This is deliberately not
            # left to the summarizer: the user's core requirement is that the
            # functional-analysis note always contains an inspectable LaTeX
            # transcription of what was actually written on the board.
            if blackboard_latex and BLACKBOARD_SUMMARY_MARKER not in summary:
                summary = (
                    summary.rstrip()
                    + "\n\n---\n\n"
                    + blackboard_latex.strip()
                )

            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
                + (f"; board={blackboard_model}" if blackboard_model else "")
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as exc:
            # Blackboard failures already carry the more precise stage.  Do
            # not overwrite it with the generic summarization stage.
            row = self._db.get_lecture(sub_id)
            if not row or row.get("error_stage") != "blackboard":
                self._reporter.info(
                    f"    [FAIL] Summarization error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._db.update_error(sub_id, "summarize", str(exc))
            raise
