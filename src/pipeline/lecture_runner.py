"""Per-lecture state machine: prefetch → timed ASR → OCR → proofread → summary.

Every normal course summary is now traceable to validated video windows.  The
ASR timestamp segments are persisted, AI-proofread against nearby PPT evidence,
and reused on later runs.  The corrected timed transcript is also persisted so
the emailer can attach it as a Markdown transcript.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from src.ai import bucketer
from src.ai.transcript_proofreader import TranscriptProofreader
from src.data.transcript_store import (
    load_proofread,
    load_transcript_segments,
    proofread_is_current,
    save_proofread,
    save_transcript_segments,
)
from src.pipeline.ppt_pipeline import PPTPipeline
from src.ai.transcriber import IncompleteAudioError, NoAudioStreamError

if TYPE_CHECKING:
    from src.data.database import Database
    from src.api.icourse import ICourseClient
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler
    from src.ai.summarizer import Summarizer
    from src.ai.transcriber import Transcriber


TRACEABLE_SUMMARY_PREFIX = "traceable-summary-v1/"


class LectureRunner:
    """Run lectures sequentially using cached expensive intermediate data."""

    def __init__(self, client: "ICourseClient", db: "Database",
                 scheduler: "Scheduler", transcriber: "Transcriber",
                 summarizer: "Summarizer", reporter: "Reporter"):
        self._client = client
        self._db = db
        self._scheduler = scheduler
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._reporter = reporter
        self._ppt = PPTPipeline(db, scheduler, reporter)
        self._proofreader: TranscriptProofreader | None = None

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        sub_id = str(lecture["sub_id"])
        sub_title = lecture.get("sub_title", sub_id)
        date = lecture.get("date", "")
        t_start = time.time()
        self._reporter.lecture_start(course_title, sub_title, date)

        existing = self._db.get_lecture(sub_id)
        if self._has_summary(existing):
            self._reporter.lecture_skip_v2_done(
                sub_title, len(existing["summary"])
            )
            self._schedule_next(next_info)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return existing["summary"]

        # PPT fetch/dedup now; OCR itself is deferred until after ASR.
        ppt_handle = self._ppt.submit(
            self._client, course_id, sub_id, defer_ocr=True,
        )
        self._schedule_next(next_info)

        transcript, transcript_segments = self._get_transcript(
            existing, course_id, sub_id,
        )
        if transcript is None:
            return None

        ppt_stats = ppt_handle.drain()
        _ = ppt_stats

        if next_info:
            next_course, next_sub = next_info
            try:
                self._ppt.prefetch_and_ocr(
                    self._client, next_course, next_sub,
                )
            except Exception as e:
                self._reporter.info(
                    f"    [Prefetch OCR] {next_sub} failed: "
                    f"{type(e).__name__}: {e}"
                )

        if not transcript.strip():
            self._reporter.info("    Empty transcript, skipping summary.")
            self._release_audio(sub_id)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return None

        summary = self._summarize(
            sub_id, course_title, transcript, transcript_segments,
        )
        if summary is None:
            self._release_audio(sub_id)
            return None

        self._db.mark_processed(sub_id)
        self._db.clear_error(sub_id)
        self._release_audio(sub_id)

        elapsed = time.time() - t_start
        self._reporter.lecture_done(course_title, sub_title, elapsed)
        return summary

    def _has_summary(self, existing: dict | None) -> bool:
        """Only the current traceable-summary format counts as complete."""
        return bool(
            existing
            and existing.get("summary")
            and str(existing.get("summary_model") or "").startswith(
                TRACEABLE_SUMMARY_PREFIX
            )
        )

    def _schedule_next(self, next_info: Optional[tuple[str, str]]):
        if next_info is None:
            return
        next_course, next_sub = next_info
        self._scheduler.prefetch_lecture(self._client, next_course, next_sub)

    def _get_transcript(self, existing: dict | None, course_id: str,
                        sub_id: str) -> tuple[Optional[str], Optional[list]]:
        """Return flattened ASR plus persisted timestamp segments.

        Legacy cached transcripts that lack segments are intentionally
        re-transcribed once.  Without source timestamps there is no honest way
        to attach a video range to a summary point.
        """
        if existing and existing.get("transcript"):
            cached_segments = load_transcript_segments(self._db, sub_id)
            if cached_segments:
                self._reporter.info(
                    f"    Transcript + timing exists "
                    f"({len(existing['transcript'])} chars, "
                    f"{len(cached_segments)} segments), skipping transcription."
                )
                return existing["transcript"], cached_segments
            self._reporter.info(
                "    Legacy transcript has no timestamp segments; "
                "re-transcribing once to enable video provenance."
            )

        downloader = self._scheduler.audio_downloader
        downloader.schedule(self._client, course_id, sub_id)
        try:
            handle = downloader.get(sub_id, timeout=120)
        except TimeoutError as e:
            self._reporter.info(f"    [SKIP] {e}")
            self._db.update_error(sub_id, "transcribe", str(e))
            return None, None
        if handle is None:
            self._reporter.lecture_skip_no_video(
                existing.get("sub_title", sub_id) if existing else sub_id
            )
            return None, None

        try:
            transcript, segments = self._transcriber.transcribe_tail(
                handle.path, handle.process, handle.stderr_chunks,
            )
        except NoAudioStreamError as e:
            self._reporter.info(f"    [SKIP] Video-only (no audio stream): {e}")
            self._db.update_error(sub_id, "transcribe", str(e))
            self._db.mark_processed(sub_id)
            self._release_audio(sub_id)
            return None, None
        except IncompleteAudioError as e:
            self._reporter.info(f"    [WARN] Incomplete audio: {e}")
            transcript = self._transcriber._last_transcript
            segments = self._transcriber._last_segments
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Transcription error: {type(e).__name__}: {e}"
            )
            self._db.update_error(sub_id, "transcribe", str(e))
            self._release_audio(sub_id)
            raise

        self._db.update_transcript(sub_id, transcript)
        save_transcript_segments(self._db, sub_id, segments)
        self._reporter.info(
            f"    [OK] Persisted {len(segments or [])} ASR timing segments "
            "for future traceable summaries."
        )
        return transcript, segments

    def _get_proofread_transcript(
        self,
        sub_id: str,
        transcript_segments: list[dict] | None,
        ppt_pages: list[dict],
        *,
        raw_blackboard: str = "",
    ) -> tuple[str, list[dict], str]:
        """Return durable AI-proofread transcript and timed chunks."""
        uses_blackboard = bool(raw_blackboard.strip())
        cached = load_proofread(self._db, sub_id)
        if cached:
            markdown, segments, model = cached
            if segments and proofread_is_current(model, uses_blackboard=uses_blackboard):
                self._reporter.info(
                    f"    AI-proofread transcript exists ({len(markdown)} chars, "
                    f"{len(segments)} timed chunks), reusing."
                )
                return markdown, segments, model

        if not transcript_segments:
            raise RuntimeError(
                "timestamped ASR segments missing; cannot proofread or cite video safely"
            )

        if self._proofreader is None:
            self._proofreader = TranscriptProofreader()
        self._reporter.info(
            "    [Time] AI-proofreading ASR against timed PPT"
            + (" + blackboard evidence..." if uses_blackboard else " evidence...")
        )
        result = self._proofreader.proofread(
            transcript_segments,
            ppt_pages,
            raw_blackboard=raw_blackboard,
        )
        save_proofread(
            self._db,
            sub_id,
            result.markdown,
            result.segments,
            result.model_label,
        )
        self._reporter.info(
            f"    [OK] AI-proofread transcript: {len(result.segments)} timed chunks, "
            f"{len(result.markdown)} chars"
        )
        return result.markdown, result.segments, result.model_label

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            _, corrected_segments, proofread_model = self._get_proofread_transcript(
                sub_id,
                transcript_segments,
                kept_pages,
            )
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
                raise RuntimeError(
                    "traceable prompt has no validated video windows"
                )
            self._reporter.info(
                f"    [Time] Generating traceable summary at "
                f"{time.strftime('%H:%M:%S')}"
                f" — mode={mode}, prompt={len(prompt_text)} chars, "
                f"windows={len(video_windows)}"
            )
            summary, model_used = self._summarizer.summarize(
                course_title,
                prompt_text,
                video_windows=video_windows,
            )
            stored_model = (
                f"{TRACEABLE_SUMMARY_PREFIX}{model_used}"
                f"|proofread={proofread_model}"
            )
            self._reporter.info(
                f"    [OK] Traceable summary by {model_used}: "
                f"{len(summary)} chars"
            )
            self._db.update_summary(sub_id, summary, stored_model)
            return summary
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Summarization error: {type(e).__name__}: {e}"
            )
            self._db.update_error(sub_id, "summarize", str(e))
            raise

    def _release_audio(self, sub_id: str):
        try:
            self._scheduler.audio_downloader.release(sub_id)
        except Exception as e:
            self._reporter.info(
                f"    [WARN] audio release failed: {type(e).__name__}: {e}"
            )
