"""LectureRunner extension with proof-preserving blackboard LaTeX transcription.

Whitelisted mathematics courses are considered complete only when both a
summary and a current-version blackboard transcription exist. This makes an
algorithm upgrade regenerate stale board evidence and then rebuild the summary.
Vision remains best-effort: runtime failures still fall back to transcript/PPT.
"""

from __future__ import annotations

from typing import Optional

from src.ai import bucketer
from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER, course_requires_blackboard
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner


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
        if not sub_id:
            return False
        # get_blackboard is version-aware. A stale board cache deliberately
        # makes this lecture incomplete so the upgraded proof selector reruns.
        return get_blackboard(self._db, sub_id) is not None

    def _ensure_blackboard(self, sub_id: str, course_title: str) -> tuple[str, str]:
        cached = get_blackboard(self._db, sub_id)
        if cached:
            markdown, model = cached
            self._reporter.info(f"    [Blackboard] current cache exists ({len(markdown)} chars), reusing.")
            return markdown, model

        self._reporter.info("    [Blackboard] cache missing/stale; regenerating proof-preserving board timeline.")
        pipeline = BlackboardPipeline(self._client, self._reporter)
        markdown, model = pipeline.run(self._active_course_id, course_title, sub_id)
        # Do not bless an empty degraded result as a valid cache. A later run
        # should be allowed to retry visual extraction.
        if markdown.strip():
            save_blackboard(self._db, sub_id, markdown, model)
        return markdown, model

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(transcript, transcript_segments, kept_pages)

            blackboard_latex = ""
            blackboard_model = ""
            blackboard_warning = ""
            if course_requires_blackboard(course_title):
                try:
                    blackboard_latex, blackboard_model = self._ensure_blackboard(sub_id, course_title)
                except Exception as exc:
                    blackboard_warning = f"{type(exc).__name__}: {exc}"
                    self._reporter.info(
                        "    [WARN] Blackboard transcription unavailable; continuing with transcript/PPT evidence: "
                        f"{blackboard_warning}"
                    )
                    self._db.update_error(sub_id, "blackboard", blackboard_warning)

                if blackboard_latex:
                    prompt_text = (
                        f"{prompt_text}\n\n"
                        "【黑板板书证据（按时间逐帧转写；必须保留证明过程）】\n"
                        f"{blackboard_latex}\n\n"
                        "整理数学笔记时必须优先保留黑板上实际出现的定义、定理、引理、等式链、不等式链和逐步证明。"
                        "不得把多步证明压缩成一句‘由某不等式可得’，除非黑板本身没有写出中间步骤。"
                        "不得补全 [unclear] 或 [blackboard frame unresolved: ...] 中不可见的内容。"
                    ).strip()
                else:
                    prompt_text = (
                        f"{prompt_text}\n\n"
                        "【板书识别状态】本节课的黑板视觉转写未能可靠取得。"
                        "请仅依据语音转写和可用 PPT/OCR 内容整理笔记，不得臆造黑板证明。"
                    ).strip()

            self._reporter.info(f"    [Time] Generating summary — mode={mode}, prompt={len(prompt_text)} chars")
            summary, model_used = self._summarizer.summarize(course_title, prompt_text)

            # Always append the raw chronological board evidence. The polished
            # summary may reorganize material, but it must never be the only
            # surviving representation of a proof written on the board.
            if blackboard_latex:
                if BLACKBOARD_SUMMARY_MARKER in summary:
                    # Avoid two copies if the summarizer happened to reproduce
                    # the marker; raw evidence below is authoritative.
                    summary = summary.split(BLACKBOARD_SUMMARY_MARKER, 1)[0].rstrip()
                summary = summary.rstrip() + "\n\n---\n\n" + blackboard_latex.strip()
            elif blackboard_warning and BLACKBOARD_SUMMARY_MARKER not in summary:
                summary = (
                    summary.rstrip() + "\n\n---\n\n### 黑板板书识别状态\n\n"
                    "本节课黑板视觉转写未完整取得；以上笔记基于可用的语音转写与 PPT/OCR 证据生成。"
                )

            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
                + (f"; board={blackboard_model}" if blackboard_model else "")
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as exc:
            self._reporter.info(f"    [FAIL] Summarization error: {type(exc).__name__}: {exc}")
            self._db.update_error(sub_id, "summarize", str(exc))
            raise
