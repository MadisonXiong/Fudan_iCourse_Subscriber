"""LectureRunner extension for proof-preserving blackboard notes.

Functional Analysis keeps its proof-preserving blackboard section. Its appended
normal AI summary uses the same timestamped, AI-proofread ASR evidence and
validated video provenance as every other course.

The raw 200k+ vision transcription is the preferred evidence source and is now
mirrored into durable meta storage. For the one lecture whose raw cache was
already damaged by an earlier schema-order migration, the previous high-quality
final board notes are also a valid recovery source: they were produced from that
raw cache before it was lost. In that narrow case we reuse the existing board
section verbatim instead of needlessly re-running 240 vision frames.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from src.ai import bucketer
from src.ai.ppt_dedup import clean_ppt_text
from src.ai.blackboard_editor_annotated import BlackboardEditor
from src.ai.blackboard_vision import course_requires_blackboard
from src.data.blackboard_store import get_blackboard, save_blackboard
from src.pipeline.blackboard_pipeline import BlackboardPipeline
from src.pipeline.lecture_runner import LectureRunner as BaseLectureRunner
from src.runtime import config


_NOTES_MODEL_PREFIX = "blackboard-llm-editor-v9/"
_VISUAL_ONLY_MODEL_PREFIX = _NOTES_MODEL_PREFIX + "visual-only/no-speech"
_BOARD_NOTES_MARKER = "### 黑板板书整理稿"
_AI_SUMMARY_HEADING_RE = re.compile(
    r"\n\s*---\s*\n\s*###\s+AI\s*课程总结(?:（带视频定位）)?\s*\n",
    re.IGNORECASE,
)

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
        if (str(lecture["sub_id"]) in config.SILENT_VISUAL_SUB_IDS
                and course_requires_blackboard(course_title)):
            return self._run_visual_only(course_id, course_title, lecture, next_info)
        return super().run(course_id, course_title, lecture, next_info=next_info)

    def _run_visual_only(self, course_id: str, course_title: str,
                         lecture: dict,
                         next_info: Optional[tuple[str, str]]) -> str:
        """Produce sourced visual notes for a subscriber-confirmed silent video."""
        sub_id = str(lecture["sub_id"])
        sub_title = str(lecture.get("sub_title") or sub_id)
        started = time.time()
        self._reporter.lecture_start(course_title, sub_title, lecture.get("date", ""))
        self._schedule_next(next_info)

        existing = self._db.get_lecture(sub_id)
        if (existing and existing.get("summary") and
                str(existing.get("summary_model") or "").startswith(
                    _VISUAL_ONLY_MODEL_PREFIX)):
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return existing["summary"]

        try:
            self._reporter.info(
                "    [Visual only] Subscriber confirmed this recording has no "
                "human speech; skipping ASR and the audio-based summary."
            )
            self._ppt.submit(self._client, course_id, sub_id).drain()
            pages = self._db.get_done_ppt_pages(sub_id)
            raw_board, vision_model = self._ensure_raw_blackboard(
                sub_id, course_title
            )
            board_notes = ""
            editor_model = "none"
            if raw_board.strip():
                board_notes, editor_model = self._edit_blackboard_or_fallback(
                    sub_id, raw_board
                )

            ppt_parts = []
            for page in pages:
                text = clean_ppt_text(page.get("text") or "").strip()
                if text:
                    minute, second = divmod(int(page.get("created_sec") or 0), 60)
                    ppt_parts.append(
                        f"#### 第 {page['page_num']} 页（视频 {minute:02d}:{second:02d}）\n\n{text}"
                    )
            if not board_notes.strip() and not ppt_parts:
                raise RuntimeError(
                    "Confirmed silent recording has no usable visual "
                    "transcription or PPT OCR; cannot produce notes"
                )

            sections = [
                "### 录播说明\n\n"
                "> 这节录播没有人声。以下笔记仅依据可见的板书和课件画面；"
                "没有语音逐字稿，也不提供基于语音的课程总结。"
            ]
            if board_notes.strip():
                sections.append(board_notes.strip())
            if ppt_parts:
                sections.append("### 课件画面文字（OCR）\n\n" + "\n\n".join(ppt_parts))
            summary = "\n\n".join(sections)
            model = (
                f"{_VISUAL_ONLY_MODEL_PREFIX}|vision={vision_model or 'none'}"
                f"|editor={editor_model}|ppt={len(ppt_parts)}"
            )
            self._db.update_summary(sub_id, summary, model)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            self._reporter.lecture_done(course_title, sub_title, time.time() - started)
            return summary
        except Exception as exc:
            self._db.update_error(sub_id, "visual-only", str(exc))
            raise

    def _prior_board_notes(self, sub_id: str) -> str:
        """Recover a previously-produced final board section, if present.

        This is deliberately not treated as raw vision evidence. It is only used
        as the already-finished student-facing board section when the raw cache
        is unavailable. If a newer combined summary already contains the normal
        AI-summary suffix, only the board part before that separator is reused.
        """
        existing = self._db.get_lecture(sub_id)
        if not existing:
            return ""
        summary = str(existing.get("summary") or "").strip()
        if not summary.startswith(_BOARD_NOTES_MARKER):
            return ""
        match = _AI_SUMMARY_HEADING_RE.search(summary)
        if match:
            summary = summary[:match.start()].strip()
        return summary if len(summary) >= 5000 else ""

    def _edit_blackboard_or_fallback(
        self,
        sub_id: str,
        blackboard_latex: str,
    ) -> tuple[str, str]:
        """Keep a lecture deliverable when the optional editor is unsafe."""
        try:
            editor = BlackboardEditor(db=self._db, sub_id=sub_id)
            board_notes, editor_model = editor.edit(blackboard_latex)
            if not board_notes.strip():
                raise RuntimeError("blackboard editor produced empty output")
            self._reporter.info(
                f"    [OK] Blackboard edited transcript: "
                f"{len(blackboard_latex)} raw chars -> {len(board_notes)} final chars"
            )
            return board_notes, editor_model
        except Exception as exc:
            self._reporter.info(
                f"    [WARN] Blackboard editor unavailable or unsafe "
                f"({type(exc).__name__}: {exc}); preserving the raw timestamped "
                "blackboard transcript so summary and email delivery can continue."
            )
            fallback = blackboard_latex.strip()
            if not fallback.startswith(_BOARD_NOTES_MARKER):
                fallback = (
                    _BOARD_NOTES_MARKER
                    + "\n\n"
                    + "> ⚠️ AI 板书整理不可用；以下保留带时间定位的原始板书转写。\n\n"
                    + fallback
                )
            return fallback, f"raw-blackboard-fallback/{type(exc).__name__}"

    def _has_summary(self, existing: dict | None) -> bool:
        if not course_requires_blackboard(self._active_course_title):
            return super()._has_summary(existing)
        if not existing or not existing.get("summary"):
            return False

        model = str(existing.get("summary_model") or "")
        if model.startswith(_NOTES_MODEL_PREFIX):
            return True
        return False

    def _ensure_raw_blackboard(self, sub_id: str, course_title: str) -> tuple[str, str]:
        cached = get_blackboard(self._db, sub_id)
        if cached:
            markdown, model = cached
            self._reporter.info(
                f"    [Blackboard] current cache exists ({len(markdown)} chars), reusing."
            )
            return markdown, model

        self._reporter.info(
            "    [Blackboard] raw cache absent; no reusable raw board transcription found."
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
            raw_cached = get_blackboard(self._db, sub_id)
            prior_board_notes = self._prior_board_notes(sub_id)

            if raw_cached:
                blackboard_latex, blackboard_model = raw_cached
                self._reporter.info(
                    f"    [Blackboard] current cache exists ({len(blackboard_latex)} chars), reusing."
                )
                board_notes, editor_model = self._edit_blackboard_or_fallback(
                    sub_id, blackboard_latex
                )
                proofread_board_evidence = blackboard_latex
            elif prior_board_notes:
                board_notes = prior_board_notes
                blackboard_latex = ""
                blackboard_model = "recovered-prior-final-board-notes"
                editor_model = "reused-prior-final-board-notes"
                proofread_board_evidence = ""
                self._reporter.info(
                    f"    [Blackboard] raw cache unavailable, but prior final board notes "
                    f"exist ({len(board_notes)} chars); reusing them verbatim. "
                    "Vision regeneration is intentionally skipped."
                )
            else:
                blackboard_latex, blackboard_model = self._ensure_raw_blackboard(
                    sub_id, course_title
                )
                if not blackboard_latex.strip():
                    raise RuntimeError("blackboard transcription empty")
                board_notes, editor_model = self._edit_blackboard_or_fallback(
                    sub_id, blackboard_latex
                )
                proofread_board_evidence = blackboard_latex

            kept_pages = self._db.get_done_ppt_pages(sub_id)
            _, corrected_segments, proofread_model = self._get_proofread_transcript(
                sub_id,
                transcript_segments,
                kept_pages,
                raw_blackboard=proofread_board_evidence,
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
                f"|vision={blackboard_model or 'none'}"
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
