"""Blackboard video-frame extraction and LaTeX transcription pipeline.

This pipeline is deliberately opt-in.  ``LectureRunner`` invokes it only for
courses whitelisted by ``course_requires_blackboard`` (default: 泛函分析).
It samples the real lecture video, drops near-identical consecutive frames,
uses a multimodal model to classify/transcribe the survivors, and returns a
chronological Markdown document whose mathematics is LaTeX.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imagehash
from PIL import Image

from src.ai.blackboard_vision import (
    BLACKBOARD_SUMMARY_MARKER,
    BlackboardVision,
    VisionFrame,
    VisionResult,
)
from src.runtime import config


class BlackboardPipeline:
    """Extract, deduplicate, classify and transcribe blackboard frames."""

    def __init__(self, client, reporter=None):
        self._client = client
        self._reporter = reporter
        self._vision = BlackboardVision()
        self.sample_sec = max(5, int(config.BLACKBOARD_SAMPLE_SEC))
        self.hash_distance = max(0, int(config.BLACKBOARD_HASH_DISTANCE))
        self.batch_size = max(1, int(config.BLACKBOARD_VISION_BATCH_SIZE))
        self.max_frames = max(0, int(config.BLACKBOARD_MAX_FRAMES))

    def run(self, course_id: str, course_title: str, sub_id: str) -> tuple[str, str]:
        """Return ``(markdown, model_description)`` for one lecture.

        Raises when the video cannot be sampled, all vision models fail, or no
        blackboard frame is found.  For a required course that failure is
        intentional: the lecture must remain retryable rather than being
        silently marked processed without its board notes.
        """
        self._info(
            f"    [Blackboard] required for {course_title}; "
            f"sampling every {self.sample_sec}s"
        )
        temp_dir = tempfile.mkdtemp(prefix=f"icourse-board-{sub_id}-")
        try:
            frames = self._extract_frames(course_id, sub_id, temp_dir)
            if not frames:
                raise RuntimeError("ffmpeg produced no video frames")
            frames = self._dedup_frames(frames)
            if self.max_frames and len(frames) > self.max_frames:
                # Evenly retain coverage across the whole lecture rather than
                # truncating the end.  Default is unlimited (0).
                step = (len(frames) - 1) / max(1, self.max_frames - 1)
                indices = sorted({round(i * step) for i in range(self.max_frames)})
                frames = [frames[i] for i in indices]
            self._info(
                f"    [Blackboard] {len(frames)} candidate frame(s) after "
                "near-duplicate filtering"
            )

            results: list[VisionResult] = []
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start:start + self.batch_size]
                batch_results = self._vision.transcribe_batch(batch, course_title)
                results.extend(batch_results)
                board_so_far = sum(1 for r in results if r.is_blackboard and r.markdown.strip())
                self._info(
                    f"    [Blackboard] vision {min(start + len(batch), len(frames))}/"
                    f"{len(frames)}; board frames={board_so_far}"
                )

            board = [r for r in results if r.is_blackboard and r.markdown.strip()]
            if not board:
                raise RuntimeError(
                    "No handwritten blackboard content was detected in sampled video frames"
                )

            markdown = self._assemble(board)
            if not markdown.strip():
                raise RuntimeError("Blackboard frames were detected but transcription is empty")
            models = ", ".join(dict.fromkeys(r.model for r in board))
            self._info(
                f"    [Blackboard] OK: {len(board)} board frame(s), "
                f"{len(markdown)} chars, model={models}"
            )
            return markdown, models
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_frames(self, course_id: str, sub_id: str, temp_dir: str) -> list[VisionFrame]:
        video_url = self._client.get_video_url(course_id, sub_id)
        if not video_url:
            raise RuntimeError("No video URL available for blackboard extraction")
        vpn_url, headers = self._client.get_stream_params(video_url)
        output_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-headers", headers,
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", vpn_url,
            "-an",
            "-vf", f"fps=1/{self.sample_sec}",
            "-q:v", "4",
            output_pattern,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=int(config.BLACKBOARD_FFMPEG_TIMEOUT),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"blackboard ffmpeg timed out after {config.BLACKBOARD_FFMPEG_TIMEOUT}s"
            ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"blackboard ffmpeg failed with code {proc.returncode}: {tail}"
            )

        paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        return [
            VisionFrame(
                frame_id=i,
                timestamp_sec=(i - 1) * self.sample_sec,
                path=str(path),
            )
            for i, path in enumerate(paths, start=1)
        ]

    def _dedup_frames(self, frames: list[VisionFrame]) -> list[VisionFrame]:
        """Drop only near-identical *consecutive* frames.

        We intentionally avoid global image dedup: a lecturer may erase the
        board and later return to a visually similar layout containing
        different mathematics.  Consecutive pHash filtering removes static
        camera shots while retaining changes in handwriting.
        """
        kept: list[VisionFrame] = []
        previous_hash = None
        for frame in frames:
            try:
                with Image.open(frame.path) as image:
                    current_hash = imagehash.phash(image.convert("RGB"))
            except Exception:
                # Keep undecodable-to-hash frames; the vision layer will make
                # the final classification and can still report a useful error.
                kept.append(frame)
                previous_hash = None
                continue
            if previous_hash is not None and (current_hash - previous_hash) <= self.hash_distance:
                continue
            kept.append(frame)
            previous_hash = current_hash
        return kept

    @staticmethod
    def _timestamp(seconds: int) -> str:
        h, rem = divmod(max(0, int(seconds)), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _assemble(self, results: list[VisionResult]) -> str:
        """Create a chronological raw transcription with light exact dedup."""
        out = [BLACKBOARD_SUMMARY_MARKER, ""]
        previous_normalized = None
        retained = 0
        for result in sorted(results, key=lambda r: r.timestamp_sec):
            text = result.markdown.strip()
            normalized = "".join(text.split())
            if normalized and normalized == previous_normalized:
                continue
            previous_normalized = normalized
            retained += 1
            out.append(f"#### {self._timestamp(result.timestamp_sec)}")
            out.append("")
            out.append(text)
            out.append("")
        if not retained:
            return ""
        return "\n".join(out).strip()

    def _info(self, message: str) -> None:
        if self._reporter:
            self._reporter.info(message)
        else:
            print(message, flush=True)
