"""LLM-only semantic editing of raw blackboard transcriptions.

The expensive vision stage remains the source of truth and is cached unchanged.
All decisions about repeated mathematical content are delegated to text LLMs:

1. local editing of chronological raw-frame chunks;
2. one global semantic consolidation of the much shorter first-pass draft;
3. an LLM coverage/faithfulness audit and, if needed, a corrective pass;
4. a format-only repair pass when deterministic syntax checks find bad LaTeX.

No deterministic fuzzy/exact content de-duplication is performed.  Python only
splits prompts, enforces safety gates, and validates Markdown/LaTeX syntax.
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

_DEFAULT_RAW_CHUNK_CHARS = 16000
_DEFAULT_LOCAL_BOUNDARY_CHARS = 4500
_DEFAULT_GLOBAL_LIMIT_CHARS = 70000
_DEFAULT_HIERARCHICAL_CHUNK_CHARS = 30000
_DEFAULT_REPAIR_CHUNK_CHARS = 10000
_DEFAULT_TIMEOUT = 600
_MIN_RECURSIVE_CHUNK_CHARS = 6500
_MAX_SPLIT_DEPTH = 2

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


LOCAL_SYSTEM_PROMPT = r"""
你是“数学板书忠实整理器”，不是课程总结助手。

输入是一节泛函分析课按时间抽取的连续黑板快照。黑板会上下移动、老师会逐步补写，所以同一段定义、定理、证明或公式可能反复出现很多次。视觉识别已经较准确。

你的任务是把当前这一小段快照整理成“无重复但信息无损”的板书稿。

硬性规则：
1. 不是摘要。保留所有唯一的定义、定理、例子、证明步骤、推导、公式、条件、反例、CHECK、Why、课堂提示。
2. 只有能确认是同一块板书的重复快照时才删除重复；不确定时宁可保留。
3. 同一内容从半句逐步补写成完整版本时，只保留最完整版本。
4. 可以统一明显等价的 LaTeX 排版差异，但不得改变数学含义。
5. 不得补充原文没有的数学知识、解释、结论、例子或评价。
6. 不得自行创造“补充说明”“附加讨论”“总结”“回顾”等标题；只有输入本身明确出现这些内容时才保留。
7. 不得写诸如“这说明……”“可用于……”“从某种意义上……”之类输入中没有的解释性句子。
8. 保持课堂原有顺序。
9. 不保留逐帧时间戳和机器标签。

Markdown/LaTeX：
- 数学表达全部放入 `$...$` 或 `$$...$$`；
- 中文放在数学环境外；
- 禁止 aligned、array、cases 等多行环境；
- 一个 display 公式尽量只含一行关系，避免超长公式块；
- 不用 Markdown 代码块。

直接输出整理后的板书正文。
""".strip()


GLOBAL_SYSTEM_PROMPT = r"""
你是“整节数学板书总编辑”。输入已经经过第一轮局部去重，但不同分块之间仍可能出现相隔很远的语义重复，例如同一个 Minkowski 定理/证明、同一个 C([a,b]) 完备性证明、同一个 Hölder 回顾被再次完整写出。

请把整份第一轮稿整理成一份最终的、连续的、无重复的完整板书稿。

这是语义级全局去重，不是字符串匹配。必须：
1. 识别同一数学内容的不同措辞、不同 LaTeX 写法、完整/不完整版本；若确认为同一内容，只保留一份最完整版本。
2. 如果后一次重复包含前一次没有的新证明步骤或条件，把这些新增信息并入第一次完整出现的位置；不要丢掉唯一内容。
3. 同一定理后面若课堂真的出现了新的证明、另一种证明、不同条件下的变体，则必须保留；只有实质相同的重复才删除。
4. 不做摘要，不缩写证明，不省略唯一公式或推导。
5. 不新增任何输入中不存在的数学内容、解释、标题、评价或结论。
6. 删除第一轮模型可能擅自添加的解释性废话，例如没有证据支持的“这说明……”“可用于……”“从某种意义上……”。
7. 不自行创造“补充说明”“附加讨论”“总结”“回顾”等标题，除非这些标题或相应内容确实存在于输入中且不是重复。
8. 按课堂第一次出现的先后顺序组织。不要按教材知识体系重新排序。

同时统一 Markdown/LaTeX：
- 所有数学内容位于 `$...$` 或 `$$...$$`；
- 中文在数学环境外；
- 禁止 aligned、array、cases；
- 长推导拆成若干独立 display 公式；
- 不使用代码块。

直接输出最终完整板书稿，不要解释你的处理过程。
""".strip()


AUDIT_SYSTEM_PROMPT = r"""
你是“数学转写完整性审计器”。你会看到：
A. 第一轮局部整理稿（它仍可能有重复，但应包含全部原始唯一内容）；
B. 全局整理后的候选最终稿。

只做审计，不要重写全文。检查两件事：
1. MISSING：A 中存在、B 中缺失的唯一数学内容，包括定义、条件、定理、例子、证明步骤、公式、反例、CHECK/Why。
   重复内容、等价措辞、纯格式差异不要报缺失。
2. UNSUPPORTED：B 中出现但 A 不支持的新增数学陈述、解释性结论、标题或评价。

如果两类问题都没有，只输出：
OK

否则严格输出：
ISSUES
MISSING:
- ...
UNSUPPORTED:
- ...

每一项尽量引用或紧贴原文，精确指出内容，不要自己补数学。
""".strip()


CORRECTION_SYSTEM_PROMPT = r"""
你是“数学板书最终校订器”。输入包含候选最终稿以及独立审计器发现的问题。

请只按审计报告修正：
- 把 MISSING 中的唯一内容恢复到课堂顺序中最合适的位置；
- 删除 UNSUPPORTED 中没有来源支持的新增内容；
- 同时继续消除明确的语义重复；
- 不得做摘要，不得删掉其他唯一证明步骤；
- 不得新增审计报告和候选稿之外的数学内容。

保持 Markdown/LaTeX 可渲染：数学放入 `$...$` 或 `$$...$$`，不用 aligned/array/cases。
只输出修正后的完整正文。
""".strip()


REPAIR_SYSTEM_PROMPT = r"""
你是 LaTeX 格式修复器。输入是一份已经完成内容整理的数学板书稿。
只能修复 Markdown/LaTeX 格式，不得删除、补充、总结、重排或改变任何数学内容。

要求：
- 所有数学内容必须在 `$...$` 或 `$$...$$` 内；
- 不得有裸露的 LaTeX 命令；
- 中文必须在数学环境外；
- 不使用 aligned/array/cases；
- 过长 display 公式拆成多个独立 `$$...$$`；
- 不使用代码块；
- 文本长度原则上应与输入接近。
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


def _split_raw_in_two(text: str) -> tuple[str, str] | None:
    starts = [match.start() for match in _FRAME_HEADING_RE.finditer(text)]
    candidates = [pos for pos in starts if 0 < pos < len(text)]
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
    """Chunk locally, then let an LLM perform semantic global consolidation."""

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
        self.local_boundary_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_BOUNDARY_CHARS",
                _DEFAULT_LOCAL_BOUNDARY_CHARS,
            )
        )
        self.global_limit_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_GLOBAL_LIMIT_CHARS",
                _DEFAULT_GLOBAL_LIMIT_CHARS,
            )
        )
        self.hierarchical_chunk_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_HIERARCHICAL_CHARS",
                _DEFAULT_HIERARCHICAL_CHUNK_CHARS,
            )
        )
        self.repair_chunk_chars = int(
            os.environ.get(
                "BLACKBOARD_EDITOR_REPAIR_CHARS",
                _DEFAULT_REPAIR_CHUNK_CHARS,
            )
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
        max_tokens: int = 16000,
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
                        temperature=0.05,
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
    def _minimum_local_ratio(previous_ratios: list[float]) -> float:
        floor = 0.08
        if previous_ratios:
            typical = statistics.median(previous_ratios[-5:])
            floor = max(floor, min(0.16, typical * 0.45))
        return floor

    def _edit_local_resilient(
        self,
        source: str,
        *,
        boundary: str,
        previous_ratios: list[float],
        label: str,
        depth: int = 0,
    ) -> tuple[str, list[str], list[float]]:
        prompt = (
            "【上一段整理稿末尾——仅用于跨段识别重复，不要照抄】\n"
            f"{boundary or '（无）'}\n\n"
            "【当前原始逐帧板书】\n"
            f"{source}"
        )
        result = self._call(LOCAL_SYSTEM_PROMPT, prompt, max_tokens=12000)
        ratio = len(result.text) / max(1, len(source))
        floor = self._minimum_local_ratio(previous_ratios)
        suspicious = (
            result.finish_reason.lower() == "length"
            or len(result.text) < 500
            or ratio < floor
        )

        if not suspicious:
            print(
                f"[BlackboardEditor] {label} accepted: "
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
        print(f"[BlackboardEditor] {label} rejected: {reason}", flush=True)

        can_split = (
            depth < _MAX_SPLIT_DEPTH
            and len(source) >= _MIN_RECURSIVE_CHUNK_CHARS * 2
        )
        split = _split_raw_in_two(source) if can_split else None
        if split is None:
            raise RuntimeError(
                f"{label} produced unsafe output and cannot be split further: "
                f"{len(source)} -> {len(result.text)} chars ({ratio:.1%})"
            )

        left, right = split
        print(
            f"[BlackboardEditor] {label}: retrying as two smaller pieces "
            f"({len(left)} + {len(right)} chars), depth={depth + 1}",
            flush=True,
        )
        left_text, left_models, left_ratios = self._edit_local_resilient(
            left,
            boundary=boundary,
            previous_ratios=previous_ratios,
            label=label + ".a",
            depth=depth + 1,
        )
        right_text, right_models, right_ratios = self._edit_local_resilient(
            right,
            boundary=left_text[-self.local_boundary_chars:],
            previous_ratios=previous_ratios + left_ratios,
            label=label + ".b",
            depth=depth + 1,
        )
        return (
            (left_text + "\n\n" + right_text).strip(),
            left_models + right_models,
            left_ratios + right_ratios,
        )

    def _run_local_pass(self, raw_chunks: list[str]) -> tuple[str, list[str]]:
        outputs: list[str] = []
        models: list[str] = []
        ratios: list[float] = []
        previous_tail = ""

        print(
            f"[BlackboardEditor] pass 1 / LLM local de-duplication: "
            f"{len(raw_chunks)} chunk(s)",
            flush=True,
        )
        for index, chunk in enumerate(raw_chunks, start=1):
            text, used_models, chunk_ratios = self._edit_local_resilient(
                chunk,
                boundary=previous_tail[-self.local_boundary_chars:],
                previous_ratios=ratios,
                label=f"pass 1 chunk {index}/{len(raw_chunks)}",
            )
            outputs.append(text)
            models.extend(used_models)
            ratios.extend(chunk_ratios)
            previous_tail = text

        combined = "\n\n".join(outputs).strip()
        return combined, models

    @staticmethod
    def _global_output_safe(source: str, result: _Result) -> tuple[bool, str]:
        if result.finish_reason.lower() == "length":
            return False, "max_tokens truncation"
        if len(result.text) < 2500:
            return False, "output too short"
        ratio = len(result.text) / max(1, len(source))
        if ratio < 0.22:
            return False, f"suspicious global compression ({ratio:.1%})"
        return True, ""

    def _global_consolidate(self, first_pass: str) -> tuple[str, list[str]]:
        """Give the entire manageable draft to an LLM for semantic de-duplication."""
        models: list[str] = []

        if len(first_pass) <= self.global_limit_chars:
            print(
                f"[BlackboardEditor] pass 2 / GLOBAL LLM consolidation: "
                f"one complete {len(first_pass)}-char draft",
                flush=True,
            )
            result = self._call(
                GLOBAL_SYSTEM_PROMPT,
                "【第一轮完整整理稿】\n" + first_pass,
                max_tokens=20000,
            )
            safe, reason = self._global_output_safe(first_pass, result)
            if safe:
                print(
                    f"[BlackboardEditor] global consolidation accepted: "
                    f"{len(first_pass)} -> {len(result.text)} chars",
                    flush=True,
                )
                return result.text.strip(), [result.model_id]
            print(
                f"[BlackboardEditor] global consolidation rejected: {reason}; "
                "falling back to hierarchical LLM consolidation.",
                flush=True,
            )

        groups = _split_paragraphs(first_pass, self.hierarchical_chunk_chars)
        group_outputs: list[str] = []
        print(
            f"[BlackboardEditor] hierarchical global fallback: "
            f"{len(groups)} semantic group(s)",
            flush=True,
        )
        for index, group in enumerate(groups, start=1):
            result = self._call(
                GLOBAL_SYSTEM_PROMPT,
                f"【第 {index}/{len(groups)} 个第一轮整理稿分组】\n{group}",
                max_tokens=16000,
            )
            safe, reason = self._global_output_safe(group, result)
            if not safe:
                raise RuntimeError(
                    f"hierarchical global group {index} unsafe: {reason}"
                )
            group_outputs.append(result.text.strip())
            models.append(result.model_id)

        merged = "\n\n".join(group_outputs).strip()
        if len(merged) <= self.global_limit_chars:
            final = self._call(
                GLOBAL_SYSTEM_PROMPT,
                "【分组全局整理后的完整稿，请做最终跨组语义去重】\n" + merged,
                max_tokens=20000,
            )
            safe, reason = self._global_output_safe(merged, final)
            if safe:
                models.append(final.model_id)
                print(
                    f"[BlackboardEditor] final hierarchical global pass accepted: "
                    f"{len(merged)} -> {len(final.text)} chars",
                    flush=True,
                )
                return final.text.strip(), models
            print(
                f"[BlackboardEditor] final hierarchical pass rejected: {reason}; "
                "keeping safely consolidated groups.",
                flush=True,
            )
        return merged, models

    def _audit(self, first_pass: str, final_text: str) -> tuple[str, list[str]]:
        combined_len = len(first_pass) + len(final_text)
        if combined_len <= 85000:
            prompt = (
                "【A：第一轮局部整理稿】\n"
                f"{first_pass}\n\n"
                "【B：候选最终稿】\n"
                f"{final_text}"
            )
            result = self._call(AUDIT_SYSTEM_PROMPT, prompt, max_tokens=6000)
            return result.text.strip(), [result.model_id]

        reports: list[str] = []
        models: list[str] = []
        audit_chunks = _split_paragraphs(first_pass, 14000)
        print(
            f"[BlackboardEditor] audit fallback: {len(audit_chunks)} source chunk(s)",
            flush=True,
        )
        for index, chunk in enumerate(audit_chunks, start=1):
            prompt = (
                f"【A：第一轮局部整理稿的第 {index}/{len(audit_chunks)} 段】\n"
                f"{chunk}\n\n"
                "【B：候选最终稿】\n"
                f"{final_text}"
            )
            result = self._call(AUDIT_SYSTEM_PROMPT, prompt, max_tokens=3500)
            models.append(result.model_id)
            if result.text.strip() != "OK":
                reports.append(result.text.strip())

        return ("\n\n".join(reports).strip() or "OK"), models

    def _correct_if_needed(
        self,
        first_pass: str,
        final_text: str,
    ) -> tuple[str, list[str]]:
        report, audit_models = self._audit(first_pass, final_text)
        if report.strip() == "OK":
            print(
                "[BlackboardEditor] LLM faithfulness audit: OK "
                "(no missing unique content or unsupported additions found)",
                flush=True,
            )
            return final_text, audit_models

        print(
            "[BlackboardEditor] LLM faithfulness audit found issues; "
            "running one corrective pass.",
            flush=True,
        )
        prompt = (
            "【候选最终稿】\n"
            f"{final_text}\n\n"
            "【审计报告】\n"
            f"{report}"
        )
        result = self._call(CORRECTION_SYSTEM_PROMPT, prompt, max_tokens=20000)
        if result.finish_reason.lower() == "length":
            raise RuntimeError("corrective global pass was truncated")
        ratio = len(result.text) / max(1, len(final_text))
        if not (0.70 <= ratio <= 1.40):
            raise RuntimeError(
                "corrective global pass changed document size too aggressively: "
                f"{len(final_text)} -> {len(result.text)} ({ratio:.1%})"
            )
        return result.text.strip(), audit_models + [result.model_id]

    def _repair_if_needed(self, text: str) -> tuple[str, list[str]]:
        issues = _format_issues(text)
        if not issues:
            return text, []

        print(
            "[BlackboardEditor] final LaTeX repair required: "
            + "; ".join(issues),
            flush=True,
        )
        chunks = _split_paragraphs(text, self.repair_chunk_chars)
        repaired: list[str] = []
        models: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            chunk_issues = _format_issues(chunk)
            if not chunk_issues:
                repaired.append(chunk)
                continue

            result = self._call(
                REPAIR_SYSTEM_PROMPT,
                "【待修复文本】\n" + chunk,
                max_tokens=10000,
            )
            ratio = len(result.text) / max(1, len(chunk))
            if (
                result.finish_reason.lower() != "length"
                and 0.80 <= ratio <= 1.25
            ):
                repaired.append(result.text.strip())
                models.append(result.model_id)
                print(
                    f"[BlackboardEditor] LaTeX repair chunk {index}/{len(chunks)} "
                    f"accepted (retention={ratio:.1%})",
                    flush=True,
                )
            else:
                repaired.append(chunk)
                print(
                    f"[BlackboardEditor] LaTeX repair chunk {index}/{len(chunks)} "
                    f"rejected; keeping original mathematical content",
                    flush=True,
                )

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

        The 229k raw vision transcript is never sent in one request.  It is
        locally reduced first; then the entire shortened draft is deliberately
        given to one text LLM so semantic duplicates can be recognized anywhere
        in the lecture, not merely at adjacent chunk seams.
        """
        raw = (raw_blackboard or "").strip()
        if not raw:
            return "", ""

        raw_chunks = _split_at_frame_headings(raw, self.raw_chunk_chars)
        print(
            f"[BlackboardEditor] raw timeline: {len(raw)} chars -> "
            f"{len(raw_chunks)} local LLM chunk(s); no audio transcript added",
            flush=True,
        )

        first_pass, models1 = self._run_local_pass(raw_chunks)
        print(
            f"[BlackboardEditor] pass 1 complete: "
            f"{len(raw)} raw -> {len(first_pass)} chars",
            flush=True,
        )

        global_text, models2 = self._global_consolidate(first_pass)
        audited_text, models3 = self._correct_if_needed(first_pass, global_text)
        final_text, models4 = self._repair_if_needed(audited_text)

        if not final_text:
            raise RuntimeError("blackboard editor produced empty final output")
        if not final_text.startswith(EDITOR_MARKER):
            final_text = f"{EDITOR_MARKER}\n\n{final_text}"

        all_models = models1 + models2 + models3 + models4
        model_label = "+".join(dict.fromkeys(all_models)) or "unknown"
        print(
            f"[BlackboardEditor] complete: {len(raw)} raw chars -> "
            f"{len(final_text)} final chars; LLM-only semantic de-duplication "
            f"+ global audit; models={model_label}",
            flush=True,
        )
        return final_text, model_label
