"""LLM-assisted layout repair for over-wide display mathematics.

This pass is deliberately separate from syntax repair.  It runs only after
Tectonic has produced a valid PDF, reads severe ``Overfull \\hbox`` warnings,
and may reflow one display-math block at a time.  The model is not allowed to
change mathematical content: after removing a small whitelist of layout-only
constructs, the repaired formula must have exactly the same token stream as the
original formula.  The caller recompiles with Tectonic and only keeps a repair
when the overflow score improves.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass

from openai import OpenAI

from src.runtime import config


_TIMEOUT = int(os.environ.get("FICS_LATEX_LAYOUT_TIMEOUT", "120"))
_MIN_OVERFULL_PT = float(os.environ.get("FICS_LATEX_LAYOUT_MIN_OVERFULL_PT", "24"))
_MAX_BLOCK_LINES = max(8, int(os.environ.get("FICS_LATEX_LAYOUT_MAX_BLOCK_LINES", "80")))

_OVERFULL_PREFIX_RE = re.compile(
    r"(?:warning:\s*)?(?:notes\.tex|[^\s/:]+\.tex):(\d+):\s*"
    r"Overfull \\hbox \(([0-9]+(?:\.[0-9]+)?)pt too wide\)",
    re.I,
)
_OVERFULL_TEX_LOG_RE = re.compile(
    r"Overfull \\hbox \(([0-9]+(?:\.[0-9]+)?)pt too wide\)"
    r"[^\n]*?at lines?\s+(\d+)",
    re.I,
)
_CODE_FENCE_RE = re.compile(r"```(?:latex|tex)?\s*\n?(.*?)\n?```", re.I | re.S)
_FORBIDDEN = re.compile(
    r"\\(?:documentclass|usepackage|input|include|write18|openout|read|catcode|csname|newcommand|renewcommand|def)\b",
    re.I,
)
# Do not let the layout pass touch formulae whose row/column separators already
# carry mathematical structure.  Those are better left alone than normalized by
# a token-insensitive line-break comparison.
_STRUCTURED_MATH = re.compile(
    r"\\begin\{(?:matrix|pmatrix|bmatrix|vmatrix|Vmatrix|array|cases|aligned|alignedat|split|gathered|multlined)\*?\}",
    re.I,
)

_SYSTEM = r"""
你是 LaTeX 数学公式排版修复器。输入是一段已经能够正确编译、但横向超出页面的 display math。

任务只有一个：在不改变任何数学内容、符号、变量、运算顺序或文字的前提下，把这段公式改成适合页面宽度的多行排版。

允许的改动：
- 添加 \\begin{aligned} ... \\end{aligned} 或 \\begin{multlined} ... \\end{multlined}；
- 添加对齐标记 &；
- 在合适的 =、\\Rightarrow、\\Longrightarrow、+、- 等边界前后添加 LaTeX 换行 \\\\；
- 必要时添加 \\displaystyle。

禁止：
- 改写、删减、补充任何数学内容或中文/英文文字；
- 改变量名、上下标、积分限、求和范围、等号/不等号、括号或函数；
- 使用 resizebox、scalebox、缩小字号等方式把整条公式压小；
- 新增宏包、宏定义、文件访问或 shell 命令。

请只返回修复后的“完整公式块”，包括原来的 \\[ 与 \\]（或原来的 display 环境）。不要解释，不要 JSON。可以使用 latex code fence，也可以直接输出 LaTeX。
""".strip()


class LatexLayoutRepairError(RuntimeError):
    """Raised when no safe display-math layout repair is available."""


@dataclass(frozen=True)
class OverfullIssue:
    line: int
    width_pt: float


@dataclass(frozen=True)
class LayoutAudit:
    model: str
    compiler_line: int
    overflow_pt: float
    start_line: int
    end_line: int
    before: str
    replacement: str
    raw_response: str
    elapsed_sec: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayoutRepairResult:
    tex: str
    audit: LayoutAudit


def overfull_issues(output: str, *, min_pt: float = 0.0) -> list[OverfullIssue]:
    raw = str(output or "")
    pairs = {
        (int(line), float(width))
        for line, width in _OVERFULL_PREFIX_RE.findall(raw)
        if float(width) >= float(min_pt)
    }
    pairs.update(
        (int(line), float(width))
        for width, line in _OVERFULL_TEX_LOG_RE.findall(raw)
        if float(width) >= float(min_pt)
    )
    issues = [OverfullIssue(line=line, width_pt=width) for line, width in pairs]
    return sorted(issues, key=lambda item: item.width_pt, reverse=True)


def overflow_score(output: str, *, min_pt: float = _MIN_OVERFULL_PT) -> float:
    """Scalar quality score; lower is better, zero means no severe overflow."""
    return round(sum(item.width_pt for item in overfull_issues(output, min_pt=min_pt)), 3)


def _display_intervals(tex: str) -> list[tuple[int, int]]:
    lines = str(tex or "").splitlines()
    intervals: list[tuple[int, int]] = []

    # Pandoc's normal display-math representation.
    stack: list[int] = []
    for idx, line in enumerate(lines, 1):
        if r"\[" in line:
            stack.append(idx)
        if r"\]" in line and stack:
            start = stack.pop()
            intervals.append((start, idx))

    # Also support standard outer display environments.  Nested aligned/split
    # environments are intentionally excluded from the outer-target set.
    env_names = ("equation", "align", "gather", "multline", "displaymath")
    for env in env_names:
        begin_re = re.compile(rf"\\begin\{{{env}\*?\}}")
        end_re = re.compile(rf"\\end\{{{env}\*?\}}")
        starts: list[int] = []
        for idx, line in enumerate(lines, 1):
            if begin_re.search(line):
                starts.append(idx)
            if end_re.search(line) and starts:
                intervals.append((starts.pop(), idx))

    return intervals


def _math_block(tex: str, line_no: int) -> tuple[int, int, str] | None:
    lines = str(tex or "").splitlines()
    candidates = [
        pair for pair in _display_intervals(tex)
        if pair[0] <= int(line_no) <= pair[1]
        and pair[1] - pair[0] + 1 <= _MAX_BLOCK_LINES
    ]
    if not candidates:
        return None
    # Prefer the smallest enclosing outer display block.
    start, end = min(candidates, key=lambda pair: pair[1] - pair[0])
    block = "\n".join(lines[start - 1:end])
    if _STRUCTURED_MATH.search(block):
        return None
    return start, end, block


def _clean_response(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = _CODE_FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip("\n")
    if raw.lower().startswith(("here is", "修复", "解释", "下面")):
        return None
    return raw.strip("\n") or None


def _semantic_fingerprint(text: str) -> str:
    """Remove only layout constructs that the model is allowed to add."""
    value = str(text or "")
    value = re.sub(r"\\begin\{(?:aligned|multlined)\}", "", value)
    value = re.sub(r"\\end\{(?:aligned|multlined)\}", "", value)
    value = value.replace(r"\displaystyle", "")
    value = value.replace("&", "")
    value = re.sub(r"\\\\(?:\[[^\]]*\])?", "", value)
    value = re.sub(r"\s+", "", value)
    return value


def _validate_replacement(before: str, replacement: str) -> None:
    if _FORBIDDEN.search(replacement):
        raise LatexLayoutRepairError("layout replacement contains a forbidden LaTeX command")
    if _STRUCTURED_MATH.search(before):
        raise LatexLayoutRepairError("formula already contains structured math; refusing automatic reflow")
    if _semantic_fingerprint(before) != _semantic_fingerprint(replacement):
        raise LatexLayoutRepairError("layout replacement changed the mathematical/content token stream")
    if replacement == before:
        raise LatexLayoutRepairError("layout replacement made no change")
    if not (
        r"\begin{aligned}" in replacement
        or r"\begin{multlined}" in replacement
        or r"\\" in replacement
    ):
        raise LatexLayoutRepairError("layout replacement did not introduce a line-breaking construct")


_RELATIONS = (r"\Longleftrightarrow", r"\Longrightarrow", r"\Leftrightarrow", r"\Rightarrow", "=")
_MAX_REFLOW_SOURCE_CHARS = 112


def _top_level_relation_positions(body: str) -> list[int]:
    """Locate relation tokens outside braces and delimiter pairs."""
    positions: list[int] = []
    brace_depth = 0
    delimiter_depth = 0
    i = 0
    while i < len(body):
        char = body[i]
        if char == "{" and (i == 0 or body[i - 1] != "\\"):
            brace_depth += 1
            i += 1
            continue
        if char == "}" and (i == 0 or body[i - 1] != "\\"):
            brace_depth = max(0, brace_depth - 1)
            i += 1
            continue
        if char in "([" and (i == 0 or body[i - 1] != "\\"):
            delimiter_depth += 1
            i += 1
            continue
        if char in ")]" and (i == 0 or body[i - 1] != "\\"):
            delimiter_depth = max(0, delimiter_depth - 1)
            i += 1
            continue
        if brace_depth == 0 and delimiter_depth == 0:
            token = next((value for value in _RELATIONS if body.startswith(value, i)), None)
            if token is not None:
                positions.append(i)
                i += len(token)
                continue
        i += 1
    return positions


def _top_level_left_positions(body: str) -> list[int]:
    r"""Locate top-level ``\left`` groups that are safe continuation points."""
    positions: list[int] = []
    brace_depth = 0
    delimiter_depth = 0
    i = 0
    while i < len(body):
        if brace_depth == 0 and delimiter_depth == 0 and body.startswith(r"\left", i):
            positions.append(i)
        char = body[i]
        if char == "{" and (i == 0 or body[i - 1] != "\\"):
            brace_depth += 1
        elif char == "}" and (i == 0 or body[i - 1] != "\\"):
            brace_depth = max(0, brace_depth - 1)
        elif char in "([" and (i == 0 or body[i - 1] != "\\"):
            delimiter_depth += 1
        elif char in ")]" and (i == 0 or body[i - 1] != "\\"):
            delimiter_depth = max(0, delimiter_depth - 1)
        i += 1
    return positions


def _split_long_piece(piece: str) -> list[str]:
    """Split a long product before a top-level parenthesized factor."""
    if len(piece) <= _MAX_REFLOW_SOURCE_CHARS:
        return [piece]
    candidates = [
        pos for pos in _top_level_left_positions(piece)
        if 16 <= pos <= len(piece) - 16
    ]
    if not candidates:
        return [piece]
    midpoint = len(piece) / 2
    split_at = min(candidates, key=lambda pos: abs(pos - midpoint))
    return [piece[:split_at].rstrip(), piece[split_at:].lstrip()]


def _deterministic_relation_reflow(block: str) -> str | None:
    """Break a long relation chain without changing its mathematical tokens."""
    match = re.fullmatch(r"\s*\\\[\s*(.*?)\s*\\\]\s*", str(block or ""), re.S)
    if not match or _STRUCTURED_MATH.search(block):
        return None
    body = match.group(1).strip()
    if len(body) < 140:
        return None
    positions = [pos for pos in _top_level_relation_positions(body) if pos > 0]
    if not positions:
        return None
    relation_pieces = [body[:positions[0]].rstrip()]
    relation_pieces.extend(
        body[start:stop].strip()
        for start, stop in zip(positions, positions[1:] + [len(body)])
    )
    pieces = [part for piece in relation_pieces for part in _split_long_piece(piece)]
    if any(not piece for piece in pieces):
        return None
    replacement = (
        "\\[\n\\begin{multlined}\n"
        + " \\\\\n".join(pieces)
        + "\n\\end{multlined}\n\\]"
    )
    _validate_replacement(block, replacement)
    return replacement


def _apply_block(tex: str, start: int, end: int, replacement: str) -> str:
    lines = str(tex or "").splitlines()
    new_lines = lines[: start - 1] + replacement.splitlines() + lines[end:]
    trailing = "\n" if str(tex or "").endswith("\n") else ""
    return "\n".join(new_lines) + trailing


def _provider_models() -> list[tuple[str, OpenAI, tuple[str, ...]]]:
    override = [
        item.strip()
        for item in os.environ.get("FICS_LATEX_LAYOUT_MODELS", "").split(",")
        if item.strip()
    ]
    out = []
    for provider in config.resolve_model_providers():
        if override:
            models = list(override)
        elif provider["name"] == "modelscope":
            models = [
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "deepseek-ai/DeepSeek-V4-Pro",
            ]
        else:
            models = list(provider["models"])
        if not models:
            continue
        out.append(
            (
                provider["name"],
                OpenAI(api_key=provider["api_key"], base_url=provider["base_url"]),
                tuple(models),
            )
        )
    return out


def repair_overfull_math(tex: str, compiler_output: str) -> LayoutRepairResult:
    """Reflow the most severe repairable over-wide display formula."""
    selected: tuple[OverfullIssue, int, int, str] | None = None
    for issue in overfull_issues(compiler_output, min_pt=_MIN_OVERFULL_PT):
        block = _math_block(tex, issue.line)
        if block is not None:
            start, end, before = block
            selected = (issue, start, end, before)
            break
    if selected is None:
        raise LatexLayoutRepairError("no severe repairable display-math overflow was found")

    issue, start, end, before = selected
    deterministic = _deterministic_relation_reflow(before)
    if deterministic is not None:
        return LayoutRepairResult(
            tex=_apply_block(tex, start, end, deterministic),
            audit=LayoutAudit(
                model="deterministic-relation-reflow",
                compiler_line=issue.line,
                overflow_pt=issue.width_pt,
                start_line=start,
                end_line=end,
                before=before,
                replacement=deterministic,
                raw_response="",
                elapsed_sec=0.0,
            ),
        )
    prompt = (
        f"Tectonic 报告该公式约超出页面 {issue.width_pt:.1f}pt。\n"
        f"原文件第 {start}--{end} 行：\n"
        "---BEGIN LATEX---\n"
        f"{before}\n"
        "---END LATEX---\n\n"
        "请只通过 aligned/multlined、& 和 \\\\ 对它重新断行；数学与文字 token 必须完全不变。"
    )

    providers = _provider_models()
    if not providers:
        raise LatexLayoutRepairError("no configured model provider is available for layout repair")

    errors: list[str] = []
    for provider_name, client, models in providers:
        for model in models:
            model_id = f"{provider_name}/{model}"
            started = time.time()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=1800,
                    timeout=_TIMEOUT,
                )
                raw = (response.choices[0].message.content or "").strip()
                replacement = _clean_response(raw)
                if replacement is None:
                    raise LatexLayoutRepairError("model returned no usable layout replacement")
                _validate_replacement(before, replacement)
                candidate = _apply_block(tex, start, end, replacement)
                elapsed = time.time() - started
                print(
                    f"[LaTeX Layout] {model_id}: overflow {issue.width_pt:.1f}pt at line "
                    f"{issue.line}, reflowed lines {start}-{end} in {elapsed:.1f}s",
                    flush=True,
                )
                return LayoutRepairResult(
                    tex=candidate,
                    audit=LayoutAudit(
                        model=model_id,
                        compiler_line=issue.line,
                        overflow_pt=issue.width_pt,
                        start_line=start,
                        end_line=end,
                        before=before,
                        replacement=replacement,
                        raw_response=raw,
                        elapsed_sec=round(elapsed, 3),
                    ),
                )
            except Exception as exc:
                message = f"{model_id}: {type(exc).__name__}: {exc}"
                errors.append(message)
                print(f"[LaTeX Layout] {message}", flush=True)

    raise LatexLayoutRepairError("all LaTeX layout models failed: " + " | ".join(errors))
