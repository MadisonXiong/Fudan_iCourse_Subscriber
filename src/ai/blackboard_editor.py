"""LLM-assisted lossless editing of raw blackboard transcriptions.

The vision stage already produces a high-quality chronological transcription of
selected full-board frames. Because a physical blackboard moves vertically and
is written incrementally, that raw timeline contains many overlapping snapshots
of the same material.

This module is NOT a lecture summarizer. It performs conservative, chunked text
editing: remove repeated board snapshots, keep every unique mathematical step,
and normalize Markdown/LaTeX for email rendering. Suspiciously short model
outputs are rejected and the offending source chunk is split and retried.
"""

from __future__ import annotations

import os
import re
import statistics
import time
from dataclasses import dataclass

from openai import OpenAI

from src.runtime import config


EDITOR_MARKER = "### 黑板板书整理稿"

# Smaller than the first implementation on purpose. The 229k-character lecture
# will normally produce about 14-16 first-pass chunks instead of 10. This costs
# a few more calls but materially reduces the risk that the editor mistakes a
# large span of unique proof material for repeated frames.
_DEFAULT_RAW_CHUNK_CHARS = 16000
_DEFAULT_SECOND_PASS_CHARS = 20000
_DEFAULT_BOUNDARY_CHARS = 4000
_DEFAULT_TIMEOUT = 600
_MIN_RECURSIVE_CHUNK_CHARS = 6500
_MAX_SPLIT_DEPTH = 2

# Qwen3-235B-A22B-Instruct-2507 was removed because ModelScope's hosted
# inference endpoint currently returns "has no provider supported" for it.
# Qwen3-30B-A3B-Instruct-2507 is confirmed working in this workflow.
_MODELSCOPE_EDITOR_MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Instruct",
]

_FRAME_HEADING_RE = re.compile(r"^####\s+[^\n]+$", re.MULTILINE)
_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)
_BARE_LATEX_RE = re.compile(
    r"\\(?:mathbb|mathcal|mathrm|mathbf|frac|sum|int|lim|to|in|notin|"
    r"subset|subseteq|supset|supseteq|leq|geq|leqslant|geqslant|forall|"
    r"exists|varepsilon|epsilon|lambda|mu|nu|delta|Rightarrow|Leftrightarrow|"
    r"leftarrow|rightarrow|infty|cdot|times|Vert|lVert|rVert|begin|end)\b"
)


EDITOR_SYSTEM_PROMPT = r"""
你是“数学板书转写编辑器”，不是课程总结助手。

输入来自泛函分析课程的逐帧黑板转写。视觉识别已经较准确，但由于教室黑板会整体上下移动、老师会逐步补写，同一段板书会在连续或稍后帧中反复出现；同一内容的 LaTeX 写法也可能有轻微差异。

你的任务是把这些重复的黑板快照整理成一份忠实、连续、可直接阅读的完整板书稿。

【必须遵守：内容】
1. 这不是摘要。必须保留所有唯一的教学内容：定义、命题、定理、例子、证明、推导步骤、公式、条件、CHECK、Why、反例、作业或课堂提示。尤其不得为了缩短输出而省略证明步骤。
2. 只删除“能够从上下文确认是同一块板书重复出现”的内容。若不能确认是重复，宁可保留，也不要删除。
3. 同一句/同一公式先出现不完整版本、后出现完整版本时，只保留最完整版本。
4. 同一内容在不同帧中只有轻微 OCR/LaTeX 差异时，根据前后多行上下文判断是否为同一板书；若确认相同，选择最完整、最一致的版本。不得把数学含义真正不同的两步强行合并。
5. 不得利用你自己的数学知识补证明、补定义、补条件或改写成教材答案。输入没有出现的数学内容不能新增。
6. 若两种转写确实冲突且无法判断，保留两个版本或保留较完整版本并标注“[转写存疑]”，不要猜。
7. 删除明显非教学噪声，例如随机的“######2016”、摄像头/录播界面标签、无意义姓名地点 OCR、纯粹的“右侧黑板”“下方黑板”等位置描述。若位置描述后面跟着真实板书，只删位置描述，不删板书。
8. 严格保持课堂出现顺序，不按你的知识体系重新排序。

【必须遵守：Markdown / LaTeX 邮件兼容性】
1. 输出必须是 Markdown；直接输出正文，不要解释你的工作。
2. 所有数学变量、关系和 LaTeX 命令必须位于 `$...$` 或 `$$...$$` 内。绝对不能让 `\\mathbb`、`\\sum`、`\\int`、`\\to`、`\\forall` 等命令裸露在普通文本中。
3. 中文说明放在数学环境之外；公式内禁止使用 `\\text{中文}`。
4. 禁止使用 `\\begin{aligned}`、`\\begin{array}`、`\\begin{cases}` 等多行环境。长推导拆成若干独立的 `$$...$$` 公式块。
5. 每个 `$$...$$` 尽量只包含一行等式/不等式，单个 display 公式尽量不超过 400 个字符，避免邮件公式图片 URL 过长。
6. 优先使用兼容性高的常见命令，如 `\\leq`、`\\geq`、`\\Rightarrow`、`\\Leftrightarrow`、`\\mathbb{R}`、`\\mathcal{B}`、`\\frac`、`\\sum`、`\\int`、`\\lim`、`\\infty`。
7. 行内公式只放短表达式；较长表达式使用独立 `$$...$$`。
8. 不要用 Markdown 代码块包裹 LaTeX。

【排版】
- 只允许 `###`、`####`、`#####` 标题。
- 定义/定理/例/证明尽量沿用板书中的原编号和措辞。
- 不保留逐帧时间戳，不保留“板书片段 N”这类机器生成结构。
- 允许把明显的板书标题整理成 Markdown 标题，但不得凭空创造课程内容。
- 宁可输出偏长的完整转写，也不要漏掉唯一的数学内容。
""".strip()


CHUNK_SYSTEM_PROMPT = EDITOR_SYSTEM_PROMPT + r"""

当前输入只是整节课的一小段。你还会看到“上一段整理稿的末尾”。那部分只用于识别跨分段重复：
- 不要重新输出上一段末尾已经完整出现的内容；
- 如果当前原始板书是在继续补写上一段末尾的一行/一段，则输出补全后的完整版本以及当前新出现的后续步骤；
- 当前段可能包含大量唯一证明内容。除非能确认是重复快照，否则不得删除；
- 不得因为输出看起来较长而主动压缩。
"""


SECOND_PASS_SYSTEM_PROMPT = EDITOR_SYSTEM_PROMPT + r"""

当前输入已经是第一轮去重稿。第二轮只能处理分段边界残留的重复和 LaTeX 格式问题。
因为第一轮已经去掉了大量逐帧重复，第二轮原则上应保留绝大多数文本：
- 只删除能够明确确认的跨块重复；
- 不再做大幅压缩；
- 保留所有唯一的定义、例子、定理、证明和推导步骤；
- 不得把完整转写概括成摘要。
"""


REPAIR_SYSTEM_PROMPT = r"""
你是 LaTeX 格式修复器。输入是一份已经完成内容整理的数学板书稿。
只能修复 Markdown/LaTeX 格式，不得删除、补充、总结、重排或改变任何数学内容。

要求：
- 所有数学内容必须在 `$...$` 或 `$$...$$` 内；
- 不得有裸露的 LaTeX 命令；
- 中文必须在数学环境外；
- 不使用 aligned/array/cases 等多行环境；
- 过长 display 公式拆成多个独立 `$$...$$`；
- 不使用代码块。
直接输出修复后的全文，文本长度原则上应与输入接近。
""".strip()


@dataclass(frozen=True)
class _Provider:
    name: str
    client: OpenAI
    models: tuple[str, ...]


@dataclass(frozen=True)
class _Result:
    text: str
    model_id: str
    finish_reason: str


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _pack_pieces(pieces: list[str], target_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        piece_len = len(piece) + 2
        if current and current_len + piece_len > target_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(piece)
        current_len += piece_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_at_frame_headings(text: str, target_chars: int) -> list[str]:
    """Split raw board text without cutting one frame snapshot in half."""
    starts = [match.start() for match in _FRAME_HEADING_RE.finditer(text)]
    if not starts:
        return _split_paragraphs(text, target_chars)

    pieces: list[str] = []
    prefix = text[: starts[0]].strip()
    if prefix:
        pieces.append(prefix)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
    return _pack_pieces(pieces, target_chars)


def _split_paragraphs(text: str, target_chars: int) -> list[str]:
    pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _pack_pieces(pieces, target_chars) if pieces else ([text] if text else [])


def _split_in_two(text: str, raw_phase: bool) -> tuple[str, str] | None:
    """Split near the middle at a safe frame/paragraph boundary."""
    if raw_phase:
        starts = [match.start() for match in _FRAME_HEADING_RE.finditer(text)]
        candidates = [pos for pos in starts if 0 < pos < len(text)]
    else:
        candidates = [m.start() for m in re.finditer(r"\n\s*\n", text)]

    if not candidates:
        return None
    midpoint = len(text) / 2
    split_at = min(candidates, key=lambda pos: abs(pos - midpoint))
    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if min(len(left), len(right)) < 1000:
        return None
    return left, right


def _outside_math(text: str) -> str:
    return _INLINE_RE.sub(" ", _DISPLAY_RE.sub(" ", text))


def _format_issues(text: str) -> list[str]:
    issues: list[str] = []
    if not text.strip():
        return ["empty output"]
    for env in ("aligned", "array", "cases"):
        if rf"\begin{{{env}}}" in text:
            issues.append(f"contains {env} environment")
    oversized = [block for block in _DISPLAY_RE.findall(text) if len(block) > 700]
    if oversized:
        issues.append(f"{len(oversized)} oversized display formula(s)")
    bare = _BARE_LATEX_RE.findall(_outside_math(text))
    if bare:
        issues.append(f"{len(bare)} bare LaTeX command(s)")
    if text.count("$") % 2:
        issues.append("unbalanced dollar delimiter")
    return issues


class BlackboardEditor:
    """Two-pass chunked editor with anti-overcompression quality gates."""

    def __init__(self):
        self.providers = self._build_providers()
        if not self.providers:
            raise ValueError(
                "No model provider available for blackboard editing. "
                "DASHSCOPE_API_KEY is recommended."
            )
        self.raw_chunk_chars = int(
            os.environ.get("BLACKBOARD_EDITOR_CHUNK_CHARS", _DEFAULT_RAW_CHUNK_CHARS)
        )
        self.second_pass_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_SECOND_PASS_CHARS", _DEFAULT_SECOND_PASS_CHARS
            )
        )
        self.boundary_chars = int(
            os.environ.get("BLACKBOARD_EDITOR_BOUNDARY_CHARS", _DEFAULT_BOUNDARY_CHARS)
        )
        self.timeout = int(
            os.environ.get("BLACKBOARD_EDITOR_TIMEOUT", _DEFAULT_TIMEOUT)
        )

    @staticmethod
    def _build_providers() -> list[_Provider]:
        providers: list[_Provider] = []
        resolved = config.resolve_model_providers()
        modelscope = next((p for p in resolved if p["name"] == "modelscope"), None)
        if modelscope:
            override = os.environ.get("BLACKBOARD_EDITOR_MODELS", "").strip()
            models = (
                [m.strip() for m in override.split(",") if m.strip()]
                if override
                else _MODELSCOPE_EDITOR_MODELS
            )
            providers.append(
                _Provider(
                    "modelscope",
                    OpenAI(
                        api_key=modelscope["api_key"],
                        base_url=modelscope["base_url"],
                    ),
                    tuple(models),
                )
            )

        for provider in resolved:
            if provider["name"] == "modelscope":
                continue
            providers.append(
                _Provider(
                    provider["name"],
                    OpenAI(
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                    ),
                    tuple(provider["models"]),
                )
            )
        return providers

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 12000,
    ) -> _Result:
        errors: list[str] = []
        for provider in self.providers:
            for model in provider.models:
                model_id = f"{provider.name}/{model}"
                t0 = time.time()
                try:
                    response = provider.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        timeout=self.timeout,
                    )
                    choice = response.choices[0]
                    text = _strip_code_fence(choice.message.content or "")
                    finish_reason = str(getattr(choice, "finish_reason", "") or "")
                    if not text:
                        raise RuntimeError("empty response")
                    elapsed = time.time() - t0
                    usage = getattr(response, "usage", None)
                    usage_text = ""
                    if usage is not None:
                        usage_text = (
                            f", tokens prompt={getattr(usage, 'prompt_tokens', '?')}"
                            f" completion={getattr(usage, 'completion_tokens', '?')}"
                        )
                    print(
                        f"[BlackboardEditor] {model_id}: {len(user_prompt)} chars -> "
                        f"{len(text)} chars in {elapsed:.0f}s, "
                        f"finish={finish_reason}{usage_text}",
                        flush=True,
                    )
                    return _Result(text, model_id, finish_reason)
                except Exception as exc:
                    print(
                        f"[BlackboardEditor] {model_id} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "All blackboard editor models failed:\n" + "\n".join(errors)
        )

    @staticmethod
    def _minimum_ratio(phase: str, previous_ratios: list[float]) -> float:
        """Return a conservative retention floor for this phase.

        Pass 1 is allowed to shrink heavily because adjacent frames duplicate
        full blackboards. Even there, sudden collapse relative to neighboring
        chunks is suspicious. Pass 2 should only remove boundary duplicates and
        therefore must retain most of its input.
        """
        if phase == "pass1":
            floor = 0.08
            if previous_ratios:
                typical = statistics.median(previous_ratios[-5:])
                floor = max(floor, min(0.16, typical * 0.45))
            return floor
        if phase == "pass2":
            floor = 0.52
            if previous_ratios:
                typical = statistics.median(previous_ratios[-4:])
                floor = max(floor, min(0.72, typical * 0.70))
            return floor
        return 0.80

    def _edit_one_resilient(
        self,
        source: str,
        *,
        boundary: str,
        system_prompt: str,
        phase: str,
        phase_label: str,
        previous_ratios: list[float],
        depth: int = 0,
    ) -> tuple[str, list[str], list[float]]:
        prompt = (
            "【上一段整理稿末尾——仅用于跨段去重，不要重复输出】\n"
            f"{boundary or '（无）'}\n\n"
            "【当前待整理内容】\n"
            f"{source}"
        )
        result = self._call(system_prompt, prompt)
        ratio = len(result.text) / max(1, len(source))
        floor = self._minimum_ratio(phase, previous_ratios)
        suspicious = (
            result.finish_reason.lower() == "length"
            or len(result.text) < 500
            or ratio < floor
        )

        if not suspicious:
            print(
                f"[BlackboardEditor] {phase_label} accepted: "
                f"{len(source)} -> {len(result.text)} chars "
                f"(retention={ratio:.1%}, floor={floor:.1%})",
                flush=True,
            )
            return result.text.strip(), [result.model_id], [ratio]

        reason = (
            "max_tokens truncation"
            if result.finish_reason.lower() == "length"
            else f"suspicious compression {ratio:.1%} < {floor:.1%}"
        )
        print(
            f"[BlackboardEditor] {phase_label} rejected: {reason}",
            flush=True,
        )

        raw_phase = phase == "pass1"
        can_split = (
            depth < _MAX_SPLIT_DEPTH
            and len(source) >= _MIN_RECURSIVE_CHUNK_CHARS * 2
        )
        split = _split_in_two(source, raw_phase=raw_phase) if can_split else None
        if split is None:
            raise RuntimeError(
                f"{phase_label} produced unsafe output and cannot be split further: "
                f"{len(source)} -> {len(result.text)} chars ({ratio:.1%})"
            )

        left, right = split
        print(
            f"[BlackboardEditor] {phase_label}: retrying as two smaller pieces "
            f"({len(left)} + {len(right)} chars), depth={depth + 1}",
            flush=True,
        )
        left_text, left_models, left_ratios = self._edit_one_resilient(
            left,
            boundary=boundary,
            system_prompt=system_prompt,
            phase=phase,
            phase_label=phase_label + ".a",
            previous_ratios=previous_ratios,
            depth=depth + 1,
        )
        right_boundary = left_text[-self.boundary_chars:]
        right_text, right_models, right_ratios = self._edit_one_resilient(
            right,
            boundary=right_boundary,
            system_prompt=system_prompt,
            phase=phase,
            phase_label=phase_label + ".b",
            previous_ratios=previous_ratios + left_ratios,
            depth=depth + 1,
        )
        return (
            (left_text + "\n\n" + right_text).strip(),
            left_models + right_models,
            left_ratios + right_ratios,
        )

    def _run_pass(
        self,
        chunks: list[str],
        *,
        system_prompt: str,
        phase: str,
        label: str,
    ) -> tuple[str, list[str]]:
        outputs: list[str] = []
        models: list[str] = []
        ratios: list[float] = []
        previous_tail = ""
        print(f"[BlackboardEditor] {label}: {len(chunks)} chunk(s)", flush=True)

        for index, chunk in enumerate(chunks, start=1):
            text, used_models, chunk_ratios = self._edit_one_resilient(
                chunk,
                boundary=previous_tail[-self.boundary_chars:],
                system_prompt=system_prompt,
                phase=phase,
                phase_label=f"{label} chunk {index}/{len(chunks)}",
                previous_ratios=ratios,
            )
            outputs.append(text)
            models.extend(used_models)
            ratios.extend(chunk_ratios)
            previous_tail = text

        return "\n\n".join(outputs).strip(), models

    def _repair_if_needed(self, text: str) -> tuple[str, list[str]]:
        issues = _format_issues(text)
        if not issues:
            return text, []

        print(
            "[BlackboardEditor] final LaTeX repair required: "
            + "; ".join(issues),
            flush=True,
        )
        chunks = _split_paragraphs(text, self.second_pass_chars)
        repaired: list[str] = []
        models: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            result = self._call(
                REPAIR_SYSTEM_PROMPT,
                "【待修复文本】\n" + chunk,
            )
            ratio = len(result.text) / max(1, len(chunk))
            if result.finish_reason.lower() == "length" or ratio < 0.80:
                raise RuntimeError(
                    f"LaTeX repair chunk {index} changed content too aggressively: "
                    f"{len(chunk)} -> {len(result.text)} chars ({ratio:.1%})"
                )
            repaired.append(result.text.strip())
            models.append(result.model_id)

        repaired_text = "\n\n".join(repaired).strip()
        remaining = _format_issues(repaired_text)
        if remaining:
            print(
                "[BlackboardEditor] warning: format validator still sees: "
                + "; ".join(remaining),
                flush=True,
            )
        return repaired_text, models

    def edit(self, raw_blackboard: str) -> tuple[str, str]:
        """Return ``(edited_markdown, model_label)`` without one-shot 220k calls."""
        raw = (raw_blackboard or "").strip()
        if not raw:
            return "", ""

        raw_chunks = _split_at_frame_headings(raw, self.raw_chunk_chars)
        print(
            f"[BlackboardEditor] raw timeline: {len(raw)} chars -> "
            f"{len(raw_chunks)} first-pass chunk(s); no audio transcript added",
            flush=True,
        )
        pass1, models1 = self._run_pass(
            raw_chunks,
            system_prompt=CHUNK_SYSTEM_PROMPT,
            phase="pass1",
            label="pass 1 / raw snapshot de-duplication",
        )

        second_chunks = _split_paragraphs(pass1, self.second_pass_chars)
        pass2, models2 = self._run_pass(
            second_chunks,
            system_prompt=SECOND_PASS_SYSTEM_PROMPT,
            phase="pass2",
            label="pass 2 / boundary merge + LaTeX normalization",
        )

        final_text, repair_models = self._repair_if_needed(pass2)
        if not final_text:
            raise RuntimeError("blackboard editor produced empty final output")
        if not final_text.startswith(EDITOR_MARKER):
            final_text = f"{EDITOR_MARKER}\n\n{final_text}"

        all_models = models1 + models2 + repair_models
        model_label = "+".join(dict.fromkeys(all_models)) or "unknown"
        print(
            f"[BlackboardEditor] complete: {len(raw)} raw chars -> "
            f"{len(final_text)} edited chars; models={model_label}",
            flush=True,
        )
        return final_text, model_label
