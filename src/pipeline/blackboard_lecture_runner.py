"""LectureRunner extension with best-effort blackboard LaTeX transcription.

All non-whitelisted courses preserve the original pipeline unchanged. For a
whitelisted course (default: 泛函分析), blackboard evidence is preferred and is
fed into the normal summarizer when available. A partial or failed vision pass
must never prevent transcript-based notes and email delivery.
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
    """Original lecture state machine plus best-effort blackboard transcription."""

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
        """A completed summary is sufficient even if vision was unavailable.

        Blackboard transcription remains cached and reused whenever present,
        but it is enhancement evidence rather than a hard completion gate.
        """
        return bool(existing and existing.get("summary"))

    def _ensure_blackboard(self, sub_id: str, course_title: str) -> tuple[str, str]:
        cached = get_blackboard(self._db, sub_id)
        if cached:
            markdown, model = cached
            self._reporter.info(
                f"    [Blackboard] cached transcription exists "
                f"({len(markdown)} chars), reusing."
            )
            return markdown, model

        pipeline = BlackboardPipeline(self._client, self._reporter)
        markdown, model = pipeline.run(
            self._active_course_id, course_title, sub_id,
        )
        save_blackboard(self._db, sub_id, markdown, model)
        return markdown, model

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        """Inject board evidence when available; otherwise continue safely."""
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(
                transcript, transcript_segments, kept_pages,
            )

            blackboard_latex = ""
            blackboard_model = ""
            blackboard_warning = ""
            if course_requires_blackboard(course_title):
                try:
                    blackboard_latex, blackboard_model = self._ensure_blackboard(
                        sub_id, course_title,
                    )
                except Exception as exc:
                    # Vision is supplementary evidence. Preserve the precise
                    # error for diagnostics but do not kill the lecture.
                    blackboard_warning = f"{type(exc).__name__}: {exc}"
                    self._reporter.info(
                        "    [WARN] Blackboard transcription unavailable; "
                        f"continuing with transcript/PPT evidence: {blackboard_warning}"
                    )
                    self._db.update_error(sub_id, "blackboard", blackboard_warning)

                if blackboard_latex:
                    prompt_text = (
                        f"{prompt_text}\n\n"
                        "【黑板板书（按时间转写；数学公式已转为 LaTeX）】\n"
                        f"{blackboard_latex}\n\n"
                        "整合笔记时，数学公式和黑板上实际写出的推导优先参考以上板书；"
                        "不得擅自补全 [unclear]、[blackboard frame unresolved: ...] "
                        "或录像中不可见的推导。"
                    ).strip()
                else:
                    prompt_text = (
                        f"{prompt_text}\n\n"
                        "【板书识别状态】本节课的黑板视觉转写未能可靠取得。"
                        "请仅依据语音转写和可用 PPT/OCR 内容整理笔记；"
                        "不要根据课程知识臆造黑板公式或推导。"
                    ).strip()

            self._reporter.info(
                f"    [Time] Generating summary — mode={mode}, "
                f"prompt={len(prompt_text)} chars"
            )
            summary, model_used = self._summarizer.summarize(
                course_title, prompt_text,
            )

            # Preserve raw board evidence verbatim so the final note remains
            # auditable instead of relying on the summarizer to reproduce it.
            if blackboard_latex and BLACKBOARD_SUMMARY_MARKER not in summary:
                summary = (
                    summary.rstrip()
                    + "\n\n---\n\n"
                    + blackboard_latex.strip()
                )
            elif blackboard_warning and BLACKBOARD_SUMMARY_MARKER not in summary:
                summary = (
                    summary.rstrip()
                    + "\n\n---\n\n"
                    + "### 黑板板书识别状态\n\n"
                    + "本节课黑板视觉转写未完整取得；以上笔记基于可用的语音转写与 PPT/OCR 证据生成。"
                )

            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
                + (f"; board={blackboard_model}" if blackboard_model else "")
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as exc:
            self._reporter.info(
                f"    [FAIL] Summarization error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._db.update_error(sub_id, "summarize", str(exc))
            raise
