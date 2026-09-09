"""Vision transcription for handwritten blackboard content.

This module is intentionally separate from RapidOCR. PPT OCR is optimized
for printed slide text; handwritten mathematics needs a multimodal model that
can reason about two-dimensional notation and emit LaTeX directly.

Only courses matched by ``course_requires_blackboard`` should invoke this
module. The default whitelist contains only ``泛函分析`` and can be overridden
with the comma-separated ``BLACKBOARD_COURSES`` environment variable.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image


BLACKBOARD_SUMMARY_MARKER = "### 黑板板书 LaTeX 转写"


def _csv_env(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def course_requires_blackboard(course_title: str) -> bool:
    """Return True only for explicitly whitelisted course-title fragments."""
    title = (course_title or "").casefold()
    patterns = _csv_env("BLACKBOARD_COURSES", "泛函分析")
    return any(p.casefold() in title for p in patterns)


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    timestamp_sec: int
    path: str


@dataclass(frozen=True)
class VisionResult:
    frame_id: int
    timestamp_sec: int
    is_blackboard: bool
    markdown: str
    model: str


_FRAME_RE = re.compile(
    r"<<<FRAME\s+id=(\d+)\s+ts=(\d+)\s+status=(board|no_board)>>>\s*"
    r"(.*?)\s*<<<END_FRAME>>>",
    re.DOTALL | re.IGNORECASE,
)


_SYSTEM_PROMPT = r"""You are transcribing university mathematics blackboards from lecture-video frames.

Your output is evidence transcription, not a reconstruction from mathematical knowledge.

Rules:
1. Decide independently for every supplied frame whether it contains a real blackboard/whiteboard with handwritten academic content. Projected slides, computer screens, subtitles, UI chrome, and printed posters are NOT blackboard content.
2. For a blackboard frame, transcribe ALL legible handwritten text and mathematics that is actually visible. Preserve reading order and logical layout as closely as Markdown permits.
3. Convert mathematical notation directly to LaTeX. Use $...$ for inline math and $$...$$ for displayed equations. Preserve subscripts, superscripts, norms, inner products, set operations, arrows, weak/weak-star convergence symbols, quantifiers, Greek letters, calligraphic letters, operators, fractions, matrices, cases, and implication/equivalence symbols exactly when visible.
4. Do not infer a missing theorem, proof step, symbol, or formula from context. If a local region is genuinely unreadable, write [unclear] at that position rather than guessing.
5. Do not add explanations, definitions, commentary, or corrections that are not written on the board.
6. If a lecturer blocks part of the board, transcribe only what remains visible.
7. Output one block for every input frame, in the same order, using EXACTLY this wrapper syntax and no surrounding code fence:

<<<FRAME id=<integer> ts=<integer> status=board>>>
<Markdown + LaTeX transcription>
<<<END_FRAME>>>

or

<<<FRAME id=<integer> ts=<integer> status=no_board>>>
<<<END_FRAME>>>

The course may be functional analysis, but that context is only for recognizing notation; never use it to invent text that is not visible."""


class BlackboardVision:
    """ModelScope OpenAI-compatible vision client.

    ModelScope's serverless catalogue changes independently of model pages.
    The first real lecture run confirmed Qwen3-VL-8B-Instruct is callable on
    the configured endpoint, while Qwen3-VL-32B-Instruct currently is not.
    Defaults therefore use the verified 8B endpoint; users can override the
    comma-separated model list with BLACKBOARD_VISION_MODELS.
    """

    def __init__(self):
        token = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not token:
            raise ValueError(
                "Blackboard transcription requires DASHSCOPE_API_KEY to contain "
                "a valid ModelScope access token."
            )
        self.base_url = os.environ.get(
            "BLACKBOARD_VISION_BASE_URL",
            "https://api-inference.modelscope.cn/v1/",
        ).strip()
        self.models = _csv_env(
            "BLACKBOARD_VISION_MODELS",
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        if not self.models:
            raise ValueError("BLACKBOARD_VISION_MODELS is empty")
        self.max_edge = int(os.environ.get("BLACKBOARD_VISION_MAX_EDGE", "1600"))
        self.max_tokens = int(os.environ.get("BLACKBOARD_VISION_MAX_TOKENS", "4096"))
        self.timeout = int(os.environ.get("BLACKBOARD_VISION_TIMEOUT", "180"))
        self.client = OpenAI(api_key=token, base_url=self.base_url)

    def _image_data_url(self, path: str) -> str:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > self.max_edge:
                im.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _parse_response_partial(text: str, frames: list[VisionFrame], model: str) -> dict[int, VisionResult]:
        """Parse every valid frame block present in a model response.

        Multimodal endpoints sometimes truncate or omit one item in a batch.
        Keeping the valid blocks lets the caller retry only missing frames
        instead of discarding the whole batch.
        """
        expected = {f.frame_id: f for f in frames}
        parsed: dict[int, VisionResult] = {}
        for match in _FRAME_RE.finditer(text or ""):
            frame_id = int(match.group(1))
            if frame_id not in expected:
                continue
            frame = expected[frame_id]
            status = match.group(3).lower()
            body = (match.group(4) or "").strip()
            parsed[frame_id] = VisionResult(
                frame_id=frame_id,
                timestamp_sec=frame.timestamp_sec,
                is_blackboard=(status == "board"),
                markdown=body,
                model=model,
            )
        return parsed

    @staticmethod
    def _parse_response(text: str, frames: list[VisionFrame], model: str) -> list[VisionResult]:
        """Strict parser retained for tests and one-frame recovery calls."""
        parsed = BlackboardVision._parse_response_partial(text, frames, model)
        expected = {f.frame_id for f in frames}
        if set(parsed) != expected:
            missing = sorted(expected - set(parsed))
            extra = sorted(set(parsed) - expected)
            raise ValueError(
                f"vision response frame mismatch: missing={missing}, extra={extra}"
            )
        return [parsed[f.frame_id] for f in frames]

    def _content(self, frames: list[VisionFrame], course_title: str) -> list[dict]:
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Course: {course_title}. The frames below are chronological. "
                    "Return one wrapper block for EVERY frame. Transcribe strictly "
                    "according to the system rules."
                ),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"FRAME id={frame.frame_id} ts={frame.timestamp_sec} "
                        f"({Path(frame.path).name})"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(frame.path)},
                }
            )
        return content

    def _request(self, model: str, frames: list[VisionFrame], course_title: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._content(frames, course_title)},
            ],
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    def transcribe_batch(self, frames: list[VisionFrame], course_title: str) -> list[VisionResult]:
        """Classify/transcribe a batch, recovering omitted frames individually.

        For each configured model, first make one efficient batch request. If
        the model returns only a subset of wrapper blocks, preserve those valid
        results and retry only the missing frames one-by-one. This addresses
        the partial-output behavior observed in the first real 泛函分析 run.
        """
        if not frames:
            return []

        errors: list[str] = []
        for model in self.models:
            try:
                text = self._request(model, frames, course_title)
                parsed = self._parse_response_partial(text, frames, model)
                missing = [f for f in frames if f.frame_id not in parsed]

                if missing:
                    print(
                        f"[BlackboardVision] {model} returned {len(parsed)}/"
                        f"{len(frames)} frames; retrying {len(missing)} missing "
                        "frame(s) individually",
                        flush=True,
                    )
                    for frame in missing:
                        one_text = self._request(model, [frame], course_title)
                        one = self._parse_response(one_text, [frame], model)
                        parsed[frame.frame_id] = one[0]

                return [parsed[f.frame_id] for f in frames]
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                print(
                    f"[BlackboardVision] {model} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        raise RuntimeError("All blackboard vision models failed:\n" + "\n".join(errors))
