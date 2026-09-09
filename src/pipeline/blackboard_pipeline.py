"""Blackboard video-frame extraction and LaTeX transcription pipeline.

The pipeline keeps full-lecture coverage while treating vision transcription
as best-effort evidence: individual malformed frames never abort a lecture.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER, BlackboardVision, VisionFrame, VisionResult
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
        self._info(f"    [Blackboard] enabled for {course_title}; dense sampling every {self.sample_sec}s, coverage anchor every {self.coverage_sec}s")
        temp_dir = tempfile.mkdtemp(prefix=f"icourse-board-{sub_id}-")
        try:
            frames = self._extract_frames(course_id, sub_id, temp_dir)
            if not frames:
                raise RuntimeError("ffmpeg produced no video frames")
            raw_count = len(frames)
            try:
                frames, stats = self._select_keyframes(frames)
                self._info(f"    [Blackboard] local selector: {raw_count} sampled -> {len(frames)} vision frame(s) (anchors={stats['anchors']}, change-events={stats['events']}, cap={self.max_frames or 'none'})")
            except Exception as exc:
                self._info(f"    [Blackboard] selector warning: {type(exc).__name__}: {exc}; falling back to even coverage")
                frames = self._fallback_even_coverage(frames)
            if not frames:
                raise RuntimeError("local blackboard selector produced no frames")

            results: list[VisionResult] = []
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start:start + self.batch_size]
                try:
                    batch_results = self._vision.transcribe_batch(batch, course_title)
                except Exception as exc:
                    self._info(f"    [Blackboard] batch warning: {type(exc).__name__}: {exc}; marking batch unresolved")
                    batch_results = [VisionResult(f.frame_id, f.timestamp_sec, True, "[blackboard frame unresolved]", "unresolved") for f in batch]
                results.extend(batch_results)
                board_so_far = sum(1 for r in results if r.is_blackboard and r.markdown.strip())
                self._info(f"    [Blackboard] vision {min(start + len(batch), len(frames))}/{len(frames)}; board frames={board_so_far}")

            board = [r for r in results if r.is_blackboard and r.markdown.strip()]
            if not board:
                self._info("    [Blackboard] warning: no usable handwritten board content detected; summary will continue from transcript/PPT")
                return "", ""
            markdown = self._assemble(board)
            if not markdown.strip():
                self._info("    [Blackboard] warning: board transcription empty after assembly; summary will continue")
                return "", ""
            models = ", ".join(dict.fromkeys(r.model for r in board))
            unresolved = sum(1 for r in board if r.markdown.startswith("[blackboard frame unresolved"))
            self._info(f"    [Blackboard] OK: {len(board)} board frame(s), {unresolved} unresolved, {len(markdown)} chars, model={models}")
            return markdown, models
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_frames(self, course_id: str, sub_id: str, temp_dir: str) -> list[VisionFrame]:
        video_url = self._client.get_video_url(course_id, sub_id)
        if not video_url:
            raise RuntimeError("No video URL available for blackboard extraction")
        vpn_url, headers = self._client.get_stream_params(video_url)
        output_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-headers", headers, "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5", "-i", vpn_url, "-an", "-vf", f"fps=1/{self.sample_sec}", "-q:v", "4", output_pattern]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=int(config.BLACKBOARD_FFMPEG_TIMEOUT), check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"blackboard ffmpeg timed out after {config.BLACKBOARD_FFMPEG_TIMEOUT}s") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"blackboard ffmpeg failed with code {proc.returncode}: {tail}")
        paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        return [VisionFrame(i, (i - 1) * self.sample_sec, str(path)) for i, path in enumerate(paths, start=1)]

    def _analysis_gray(self, frame: VisionFrame) -> np.ndarray:
        with Image.open(frame.path) as image:
            image = image.convert("L")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid frame size for {frame.path}")
            new_h = max(54, round(height * self.analysis_width / width))
            image = image.resize((self.analysis_width, new_h), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _robust_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = b.astype(np.int16) - a.astype(np.int16)
        sparse = delta[::4, ::4]
        shift = float(np.median(sparse)) if sparse.size else 0.0
        return np.abs(delta.astype(np.float32) - shift)

    def _select_keyframes(self, frames: list[VisionFrame]) -> tuple[list[VisionFrame], dict[str, int]]:
        n = len(frames)
        if n <= 2:
            return list(frames), {"anchors": n, "events": 0}
        gray = [self._analysis_gray(frame) for frame in frames]
        if len({arr.shape for arr in gray}) != 1:
            raise ValueError("sampled frames have inconsistent dimensions")
        motion_after = [1.0] * n
        for i in range(n - 1):
            diff = self._robust_abs_diff(gray[i], gray[i + 1])
            motion_after[i] = float(np.mean(diff >= self.change_threshold))
        motion_after[-1] = motion_after[-2]
        anchors: set[int] = {0, n - 1}
        windows: dict[int, list[int]] = {}
        for i, frame in enumerate(frames):
            windows.setdefault(frame.timestamp_sec // self.coverage_sec, []).append(i)
        for indices in windows.values():
            anchors.add(min(indices, key=lambda i: (motion_after[i], -frames[i].timestamp_sec)))
        event_scores: dict[int, float] = {}
        for i in range(1, n - 1):
            changed = self._robust_abs_diff(gray[i - 1], gray[i])
            stable_next = self._robust_abs_diff(gray[i], gray[i + 1])
            persistent = (changed >= self.change_threshold) & (stable_next <= self.stable_threshold)
            score = float(np.mean(persistent))
            if score >= self.change_ratio:
                target = min(i + 1, n - 1)
                event_scores[target] = max(score, event_scores.get(target, 0.0))
        event_indices: list[int] = []
        for idx, _score in sorted(event_scores.items(), key=lambda item: item[1], reverse=True):
            ts = frames[idx].timestamp_sec
            if any(abs(ts - frames[j].timestamp_sec) < self.event_gap_sec for j in event_indices):
                continue
            event_indices.append(idx)
        selected = set(anchors)
        if self.max_frames and len(selected) > self.max_frames:
            ordered = sorted(selected)
            step = (len(ordered) - 1) / max(1, self.max_frames - 1)
            selected = {ordered[round(i * step)] for i in range(self.max_frames)}
            event_indices = []
        else:
            for idx in sorted(event_indices, key=lambda x: event_scores.get(x, 0.0), reverse=True):
                if idx in selected:
                    continue
                if self.max_frames and len(selected) >= self.max_frames:
                    break
                selected.add(idx)
        event_kept = sum(1 for i in selected if i in event_scores and i not in anchors)
        return [frames[i] for i in sorted(selected)], {"anchors": len(anchors), "events": event_kept}

    def _fallback_even_coverage(self, frames: list[VisionFrame]) -> list[VisionFrame]:
        if not frames:
            return []
        target = self.max_frames if self.max_frames else max(2, int(frames[-1].timestamp_sec / self.coverage_sec) + 2)
        if len(frames) <= target:
            return list(frames)
        step = (len(frames) - 1) / max(1, target - 1)
        return [frames[i] for i in sorted({round(j * step) for j in range(target)})]

    @staticmethod
    def _timestamp(seconds: int) -> str:
        h, rem = divmod(max(0, int(seconds)), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _assemble(self, results: list[VisionResult]) -> str:
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
            out.extend([f"#### {self._timestamp(result.timestamp_sec)}", "", text, ""])
        return "\n".join(out).strip() if retained else ""

    def _info(self, message: str) -> None:
        if self._reporter:
            self._reporter.info(message)
        else:
            print(message, flush=True)
