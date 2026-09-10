"""LLM-assisted lossless editing of raw blackboard transcriptions.

The vision stage already produces a high-quality chronological transcription of
selected full-board frames.  Because a physical blackboard moves vertically and
is written incrementally, that raw timeline contains many overlapping snapshots
of the same material.  This module edits those snapshots into one readable
Markdown/LaTeX transcript.

This is deliberately NOT a lecture summarizer.  It does not receive the audio
transcript and it is instructed to preserve every unique definition, theorem,
example, proof step and formula.  Its only semantic operations are de-duplication
of repeated board states, choosing the most complete repeated transcription, and
normalizing LaTeX for the email renderer.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from src.runtime import config


EDITOR_MARKER = "### 黑板板书整理稿"

# Keep individual calls comfortably below hosted context limits.  The current
# 229k-ish raw lecture is normally split into about 9-11 pieces.
_DEFAULT_RAW_CHUNK_CHARS = 24000
_DEFAULT_SECOND_PASS_CHARS = 28000
_DEFAULT_BOUNDARY_CHARS = 4500
_DEFAULT_TIMEOUT = 600

# ModelScope already works for the blackboard vision stage, so prefer Qwen text
# models on the same OpenAI-compatible endpoint.  The known-working VL model is
# retained as the final ModelScope fallback because it also accepts text-only
# chat requests.
_MODELSCOPE_EDITOR_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
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
2. 删除由黑板上下移动、摄像机重复拍摄、逐步补写造成的重复。同一句/同一公式先出现不完整版本、后出现完整版本时，只保留最完整版本。
3. 同一内容在不同帧中只有轻微 OCR/LaTeX 差异时，根据前后多行上下文判断是否为同一板书；若确认是同一内容，选择最完整、最一致的版本。不得把数学含义真正不同的两步强行合并。
4. 不得利用你自己的数学知识补证明、补定义、补条件或改写成教材答案。输入没有出现的数学内容不能新增。
5. 若两种转写确实冲突且无法从重复上下文判断，保留较完整版本并在其后写“[转写存疑]”，不要猜。
6. 删除明显非教学噪声，例如随机的“######2016”、摄像头/录播界面标签、无意义姓名地点 OCR、纯粹的“右侧黑板”“下方黑板”等位置描述。若位置描述后面跟着真实板书，只删位置描述，不删板书。
7. 严格保持课堂出现顺序，不按你的知识体系重新排序。

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

当前输入只是整节课的一段。你还会看到“上一段整理稿的末尾”。那部分只用于识别跨分段重复：
- 不要重新输出上一段末尾已经完整出现的内容；
- 如果当前原始板书是在继续补写上一段末尾的一行/一段，则只输出补全后的完整版本，并保留当前新出现的后续步骤；
- 不得因为只看到局部上下文而概括或省略当前段的唯一内容。
"""


SECOND_PASS_SYSTEM_PROMPT = EDITOR_SYSTEM_PROMPT + r"""

当前输入已经是第一轮整理稿，不再包含逐帧原始快照。请做第二轮校订：
- 只删除第一轮分段边界残留的重复；
- 统一并修复 LaTeX 邮件兼容格式；
- 保留所有唯一的定义、例子、定理、证明和推导步骤；
- 不得把完整转写进一步概括成摘要。
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
直接输出修复后的全文。
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


def _split_at_frame_headings(text: str, target_chars: int) -> list[str]:
    """Split raw board text without cutting an individual frame snapshot."""
    starts = [match.start() for match in _FRAME_HEADING_RE.finditer(text)]
    if not starts:
        return _split_paragraphs(text, target_chars)

    prefix = text[: starts[0]].strip()
    pieces: list[str] = []
    if prefix:
        pieces.append(prefix)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
    return _pack_pieces(pieces, target_chars)


def _split_paragraphs(text: str, target_chars: int) -> list[str]:
    """Split cleaned Markdown at paragraph boundaries for the second pass."""
    pieces = [piece.strip() for piece in re.split(r"\n\s*\n", text) if piece.strip()]
    if not pieces:
        return [text.strip()] if text.strip() else []
    return _pack_pieces(pieces, target_chars)


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


def _outside_math(text: str) -> str:
    masked = _DISPLAY_RE.sub(" ", text)
    masked = _INLINE_RE.sub(" ", masked)
    return masked


def _format_issues(text: str) -> list[str]:
    """Cheap deterministic checks for the failure modes seen in email output."""
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
    """Two-pass chunked editor for a complete raw blackboard timeline."""

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
                "BLACKBOARD_EDITOR_SECOND_PASS_CHARS",
                _DEFAULT_SECOND_PASS_CHARS,
            )
        )
        self.boundary_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_BOUNDARY_CHARS",
                _DEFAULT_BOUNDARY_CHARS,
            )
        )
        self.timeout = int(
            os.environ.get("BLACKBOARD_EDITOR_TIMEOUT", _DEFAULT_TIMEOUT)
        )

    @staticmethod
    def _build_providers() -> list[_Provider]:
        providers: list[_Provider] = []
        resolved = config.resolve_model_providers()

        # Prefer ModelScope/Qwen for this task.  It is already the proven route
        # for blackboard vision and avoids the previously failing giant prompt.
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

        # Other configured providers remain fallbacks.  Do not duplicate the
        # ModelScope models above.
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
                    finish_reason = str(
                        getattr(choice, "finish_reason", "") or ""
                    )
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
                        f"[BlackboardEditor] {model_id}: "
                        f"{len(user_prompt)} chars -> {len(text)} chars "
                        f"in {elapsed:.0f}s, finish={finish_reason}{usage_text}",
                        flush=True,
                    )
                    return _Result(text, model_id, finish_reason)
                except Exception as exc:
                    print(
                        f"[BlackboardEditor] {model_id} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    errors.append(
                        f"{model_id}: {type(exc).__name__}: {exc}"
                    )

        raise RuntimeError(
            "All blackboard editor models failed:\n" + "\n".join(errors)
        )

    def _edit_chunks(
        self,
        chunks: list[str],
        *,
        system_prompt: str,
        phase_name: str,
    ) -> tuple[str, list[str]]:
        cleaned: list[str] = []
        models_used: list[str] = []
        previous_tail = ""

        print(
            f"[BlackboardEditor] {phase_name}: {len(chunks)} chunk(s)",
            flush=True,
        )
        for index, chunk in enumerate(chunks, start=1):
            boundary = (
                previous_tail[-self.boundary_chars:]
                if previous_tail
                else "（无）"
            )
            prompt = (
                f"这是本轮第 {index}/{len(chunks)} 段。\n\n"
                "【上一段整理稿末尾——仅用于跨段去重，不要重复输出】\n"
                f"{boundary}\n\n"
                "【当前待整理内容】\n"
                f"{chunk}"
            )
            result = self._call(system_prompt, prompt)
            if result.finish_reason.lower() == "length":
                raise RuntimeError(
                    f"{phase_name} chunk {index} was truncated by max_tokens"
                )

            # A chunk can legitimately shrink a lot because raw snapshots are
            # highly repetitive, but a near-empty response is unsafe.
            minimum = max(500, int(len(chunk) * 0.03))
            if len(result.text) < minimum:
                raise RuntimeError(
                    f"{phase_name} chunk {index} output suspiciously short: "
                    f"{len(result.text)} chars for {len(chunk)} input chars"
                )

            cleaned.append(result.text.strip())
            previous_tail = result.text
            models_used.append(result.model_id)
            print(
                f"[BlackboardEditor] {phase_name} chunk {index}/{len(chunks)} "
                f"accepted ({len(chunk)} -> {len(result.text)} chars)",
                flush=True,
            )

        return "\n\n".join(cleaned).strip(), models_used

    def _repair_if_needed(self, text: str) -> tuple[str, list[str]]:
        issues = _format_issues(text)
        if not issues:
            return text, []

        print(
            "[BlackboardEditor] final LaTeX repair required: "
            + "; ".join(issues),
            flush=True,
        )

        # Repair in chunks too; never send the entire long lecture in one
        # request.  No cross-boundary semantic editing is needed at this stage.
        chunks = _split_paragraphs(text, self.second_pass_chars)
        repaired: list[str] = []
        models: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = "【待修复文本】\n" + chunk
            result = self._call(REPAIR_SYSTEM_PROMPT, prompt)
            if result.finish_reason.lower() == "length":
                raise RuntimeError(
                    f"LaTeX repair chunk {index} was truncated by max_tokens"
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
        """Return ``(edited_markdown, model_label)``.

        The 220k+ raw timeline is never sent as one prompt.  Pass 1 cleans
        chronological frame chunks; pass 2 removes cross-chunk repetition and
        normalizes LaTeX once more.  A narrow repair pass runs only if cheap
        deterministic checks still detect the failure modes seen in email.
        """
        raw = (raw_blackboard or "").strip()
        if not raw:
            return "", ""

        raw_chunks = _split_at_frame_headings(raw, self.raw_chunk_chars)
        print(
            f"[BlackboardEditor] raw timeline: {len(raw)} chars -> "
            f"{len(raw_chunks)} first-pass chunk(s); no audio transcript added",
            flush=True,
        )
        pass1, models1 = self._edit_chunks(
            raw_chunks,
            system_prompt=CHUNK_SYSTEM_PROMPT,
            phase_name="pass 1 / raw snapshot de-duplication",
        )

        second_chunks = _split_paragraphs(pass1, self.second_pass_chars)
        pass2, models2 = self._edit_chunks(
            second_chunks,
            system_prompt=SECOND_PASS_SYSTEM_PROMPT,
            phase_name="pass 2 / boundary merge + LaTeX normalization",
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
