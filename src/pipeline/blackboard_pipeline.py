"""Blackboard video-frame extraction and LaTeX transcription pipeline.

This pipeline is deliberately opt-in. ``LectureRunner`` invokes it only for
courses whitelisted by ``course_requires_blackboard`` (default: 泛函分析).

The expensive vision model does *not* receive every sampled frame. We sample
the whole lecture densely, then run a local temporal selector designed for a
fixed classroom camera:

* periodic coverage anchors guarantee the entire lecture remains represented;
* persistent-change detection keeps board states that changed and then stayed
  stable, which is much less sensitive to a lecturer walking across the view;
* a soft frame cap preserves all coverage anchors first and then the strongest
  extra change events.

Only the selected frames are sent to the multimodal model for Markdown+LaTeX
transcription.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.ai.blackboard_vision import (
    BLACKBOARD_SUMMARY_MARKER,
    BlackboardVision,
    VisionFrame,
    VisionResult,
)
from src.runtime import config


class BlackboardPipeline:
    """Extract, locally select, classify and transcribe blackboard frames."""

    def __init__(self, client, reporter=None):
        self._client = client
        self._reporter = reporter
        self._vision = BlackboardVision()
        self.sample_sec = max(5, int(config.BLACKBOARD_SAMPLE_SEC))
        self.batch_size = max(1, int(config.BLACKBOARD_VISION_BATCH_SIZE))
        self.max_frames = max(0, int(config.BLACKBOARD_MAX_FRAMES))
        self.coverage_sec = max(self.sample_sec, int(config.BLACKBOARD_COVERAGE_SEC))
        self.analysis_width = max(96, int(config.BLACKBOARD_ANALYSIS_WIDTH))
        self.change_threshold = max(1, int(config.BLACKBOARD_CHANGE_THRESHOLD))
        self.stable_threshold = max(0, int(config.BLACKBOARD_STABLE_THRESHOLD))
        self.change_ratio = max(0.0, float(config.BLACKBOARD_CHANGE_RATIO))
        self.event_gap_sec = max(self.sample_sec, int(config.BLACKBOARD_EVENT_GAP_SEC))

    def run(self, course_id: str, course_title: str, sub_id: str) -> tuple[str, str]:
        """Return ``(markdown, model_description)`` for one lecture.

        Raises when the video cannot be sampled, all vision models fail, or no
        blackboard frame is found. For a required course that failure is
        intentional: the lecture remains retryable rather than being silently
        marked processed without board notes.
        """
        self._info(
            f"    [Blackboard] required for {course_title}; dense sampling "
            f"every {self.sample_sec}s, coverage anchor every {self.coverage_sec}s"
        )
        temp_dir = tempfile.mkdtemp(prefix=f"icourse-board-{sub_id}-")
        try:
            frames = self._extract_frames(course_id, sub_id, temp_dir)
            if not frames:
                raise RuntimeError("ffmpeg produced no video frames")
            raw_count = len(frames)

            try:
                frames, selector_stats = self._select_keyframes(frames)
                self._info(
                    "    [Blackboard] local selector: "
                    f"{raw_count} sampled -> {len(frames)} vision frame(s) "
                    f"(anchors={selector_stats['anchors']}, "
                    f"change-events={selector_stats['events']}, "
                    f"cap={self.max_frames or 'none'})"
                )
            except Exception as exc:
                # Selection is an optimization, never a reason to lose the
                # required transcript. Fall back to evenly spaced coverage.
                self._info(
                    f"    [Blackboard] selector warning: {type(exc).__name__}: "
                    f"{exc}; falling back to even coverage"
                )
                frames = self._fallback_even_coverage(frames)
                self._info(
                    f"    [Blackboard] fallback selected {len(frames)}/"
                    f"{raw_count} frame(s)"
                )

            if not frames:
                raise RuntimeError("local blackboard selector produced no frames")

            results: list[VisionResult] = []
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start:start + self.batch_size]
                batch_results = self._vision.transcribe_batch(batch, course_title)
                results.extend(batch_results)
                board_so_far = sum(
                    1 for r in results if r.is_blackboard and r.markdown.strip()
                )
                self._info(
                    f"    [Blackboard] vision "
                    f"{min(start + len(batch), len(frames))}/{len(frames)}; "
                    f"board frames={board_so_far}"
                )

            board = [r for r in results if r.is_blackboard and r.markdown.strip()]
            if not board:
                raise RuntimeError(
                    "No handwritten blackboard content was detected in selected video frames"
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

    def _analysis_gray(self, frame: VisionFrame) -> np.ndarray:
        """Load one low-resolution grayscale image for temporal comparison."""
        with Image.open(frame.path) as image:
            image = image.convert("L")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid frame size for {frame.path}")
            new_h = max(54, round(height * self.analysis_width / width))
            image = image.resize(
                (self.analysis_width, new_h),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _robust_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Absolute frame difference after compensating global brightness.

        Classroom cameras often auto-adjust exposure as a lecturer moves. A
        global luminance shift should not look like a board rewrite, so we
        subtract a robust median shift estimated on a sparse pixel grid.
        """
        delta = b.astype(np.int16) - a.astype(np.int16)
        sparse = delta[::4, ::4]
        shift = float(np.median(sparse)) if sparse.size else 0.0
        return np.abs(delta.astype(np.float32) - shift)

    def _select_keyframes(self, frames: list[VisionFrame]) -> tuple[list[VisionFrame], dict[str, int]]:
        """Choose full-lecture coverage plus persistent board-change events.

        A genuine newly written/erased board region differs from the previous
        sample but is usually still present in the next sample. A moving
        lecturer tends to differ in both directions. We therefore score the
        mask ``changed_from_previous AND stable_to_next``. Periodic anchors are
        selected separately, so the transcript does not rely on this heuristic
        to cover the lecture timeline.
        """
        n = len(frames)
        if n <= 2:
            return list(frames), {"anchors": n, "events": 0}

        gray = [self._analysis_gray(frame) for frame in frames]
        if len({arr.shape for arr in gray}) != 1:
            raise ValueError("sampled frames have inconsistent dimensions")

        # Motion-after score is also used to choose a relatively stable anchor
        # inside each coverage window. Ties prefer the latest frame, which
        # tends to contain the most accumulated writing in that window.
        motion_after = [1.0] * n
        for i in range(n - 1):
            diff = self._robust_abs_diff(gray[i], gray[i + 1])
            motion_after[i] = float(np.mean(diff >= self.change_threshold))
        motion_after[-1] = motion_after[-2]

        anchors: set[int] = {0, n - 1}
        windows: dict[int, list[int]] = {}
        for i, frame in enumerate(frames):
            key = frame.timestamp_sec // self.coverage_sec
            windows.setdefault(key, []).append(i)
        for indices in windows.values():
            best = min(indices, key=lambda i: (motion_after[i], -frames[i].timestamp_sec))
            anchors.add(best)

        event_scores: dict[int, float] = {}
        for i in range(1, n - 1):
            changed = self._robust_abs_diff(gray[i - 1], gray[i])
            stable_next = self._robust_abs_diff(gray[i], gray[i + 1])
            persistent = (
                (changed >= self.change_threshold)
                & (stable_next <= self.stable_threshold)
            )
            score = float(np.mean(persistent))
            if score >= self.change_ratio:
                # Use the following frame: the new state has survived one more
                # sample and is more likely to be readable after the lecturer
                # has moved away from freshly written text.
                target = min(i + 1, n - 1)
                event_scores[target] = max(score, event_scores.get(target, 0.0))

        # Collapse dense event clusters. Process strongest changes first so a
        # board erase/rewrite wins over tiny transient changes nearby.
        event_indices: list[int] = []
        for idx, _score in sorted(event_scores.items(), key=lambda item: item[1], reverse=True):
            ts = frames[idx].timestamp_sec
            if any(abs(ts - frames[j].timestamp_sec) < self.event_gap_sec for j in event_indices):
                continue
            event_indices.append(idx)

        selected = set(anchors)
        # If a cap is active, coverage anchors have priority. Extra event slots
        # are filled by change strength. For unusually long lectures where the
        # anchors alone exceed the cap, downsample anchors evenly rather than
        # truncating the end of the lecture.
        if self.max_frames and len(selected) > self.max_frames:
            ordered = sorted(selected)
            step = (len(ordered) - 1) / max(1, self.max_frames - 1)
            selected = {ordered[round(i * step)] for i in range(self.max_frames)}
            event_indices = []
        else:
            ranked_events = sorted(
                event_indices,
                key=lambda idx: event_scores.get(idx, 0.0),
                reverse=True,
            )
            for idx in ranked_events:
                if idx in selected:
                    continue
                if self.max_frames and len(selected) >= self.max_frames:
                    break
                selected.add(idx)

        ordered_selected = [frames[i] for i in sorted(selected)]
        event_kept = sum(1 for i in selected if i in event_scores and i not in anchors)
        return ordered_selected, {"anchors": len(anchors), "events": event_kept}

    def _fallback_even_coverage(self, frames: list[VisionFrame]) -> list[VisionFrame]:
        """Safe fallback when local image analysis unexpectedly fails."""
        if not frames:
            return []
        target = self.max_frames if self.max_frames else max(
            2,
            int(frames[-1].timestamp_sec / self.coverage_sec) + 2,
        )
        if len(frames) <= target:
            return list(frames)
        step = (len(frames) - 1) / max(1, target - 1)
        indices = sorted({round(i * step) for i in range(target)})
        return [frames[i] for i in indices]

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
