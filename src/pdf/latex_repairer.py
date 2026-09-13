"""Compiler-local LLM repair for ambiguous LaTeX failures.

The repair model sees only a small slice around Tectonic's concrete error line
and returns a repaired version of that slice.  The caller recompiles the whole
document with Tectonic; model output is never trusted as valid by itself.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass

from openai import OpenAI

from src.runtime import config


_TIMEOUT = int(os.environ.get("FICS_LATEX_REPAIR_TIMEOUT", "120"))
_RADIUS = max(6, int(os.environ.get("FICS_LATEX_REPAIR_CONTEXT_LINES", "12")))
_MIN_RATIO = float(os.environ.get("FICS_LATEX_REPAIR_MIN_RATIO", "0.45"))
_MAX_RATIO = float(os.environ.get("FICS_LATEX_REPAIR_MAX_RATIO", "1.90"))

_ERROR_LINE_RE = re.compile(r"(?:notes\.tex|[^\s/:]+\.tex):(\d+):")
_CODE_FENCE_RE = re.compile(r"```(?:latex|tex)?\s*\n?(.*?)\n?```", re.I | re.S)
_FORBIDDEN = re.compile(
    r"\\(?:documentclass|usepackage|input|include|write18|openout|read|catcode|csname|newcommand|renewcommand|def)\b",
    re.IGNORECASE,
)
_STRUCTURAL_LINE_RE = re.compile(
    r"^\\(?:hypertarget|(?:sub)*section|paragraph)\b"
)

_SYSTEM = r"""
你是 LaTeX 局部编译错误修复器。

只修复给出的 LaTeX 片段，使当前 Tectonic 错误消失。
- 保留片段中已有的全部课程文字、数学含义、变量和顺序；不要总结、解释或补充数学知识。
- 可以合并被错误拆开的数学环境，修复 \left/\right、花括号、美元符、\begin/\end，以及把已有中文放入 \text{...}。
- 不得新增原片段不存在的命题、条件、定义、证明、例子或变量。
- 不得新增宏包、宏定义、文件访问或 shell 命令。
- 如果必须猜数学内容才能修复，只输出 NO_SAFE_REPAIR。
- 否则只输出“修复后的完整片段”，不要解释，不要行号，不要 JSON。可以使用一个 latex code fence，也可以直接输出 LaTeX。
""".strip()


class LatexRepairError(RuntimeError):
    """Raised when no safe compiler-local repair can be obtained."""


@dataclass(frozen=True)
class RepairAudit:
    model: str
    compiler_error_line: int
    allowed_start_line: int
    allowed_end_line: int
    start_line: int
    end_line: int
    before: str
    replacement: str
    raw_response: str
    elapsed_sec: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RepairResult:
    tex: str
    audit: RepairAudit


def compiler_error_line(output: str) -> int | None:
    matches = _ERROR_LINE_RE.findall(str(output or ""))
    return int(matches[-1]) if matches else None


def _window(tex: str, line_no: int) -> tuple[int, int, str]:
    lines = str(tex or "").splitlines()
    if not lines:
        raise LatexRepairError("LaTeX source is empty")
    line_no = min(max(1, int(line_no)), len(lines))
    start = max(1, line_no - _RADIUS)
    end = min(len(lines), line_no + _RADIUS)
    return start, end, "\n".join(lines[start - 1:end])


def _error_excerpt(output: str, line_no: int) -> str:
    """Keep only the compiler-local diagnostic instead of sending a huge log."""
    lines = str(output or "").splitlines()
    needle = f":{line_no}:"
    hit = -1
    for i, line in enumerate(lines):
        if needle in line and ("error:" in line.lower() or ".tex:" in line.lower()):
            hit = i
    if hit < 0:
        for i in range(len(lines) - 1, -1, -1):
            if "error:" in lines[i].lower():
                hit = i
                break
    if hit < 0:
        return "Tectonic compilation failed."
    start = max(0, hit - 2)
    end = min(len(lines), hit + 3)
    return "\n".join(lines[start:end])[-1800:]


def _clean_response(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw or raw == "NO_SAFE_REPAIR":
        return None
    fenced = _CODE_FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip("\n")
    # Reject explanatory prose around the code.  The model is asked to return
    # only the replacement slice, so extra wrappers are a protocol failure.
    if raw.startswith("<<<") or raw.lower().startswith(("here is", "修复", "解释")):
        return None
    return raw.strip("\n") or None


def _validate_replacement(before: str, replacement: str) -> None:
    if _FORBIDDEN.search(replacement):
        raise LatexRepairError("model replacement contains a forbidden LaTeX command")
    if "```" in replacement:
        raise LatexRepairError("model replacement contains an unterminated code fence")
    # Pandoc-generated headings are known-good document structure.  A local
    # syntax repair must never duplicate, delete, rename, or re-brace them;
    # doing so can turn one bad formula into a permanent ``Too many }`` loop.
    before_structure = [
        line.strip()
        for line in before.splitlines()
        if _STRUCTURAL_LINE_RE.match(line.strip())
    ]
    replacement_structure = [
        line.strip()
        for line in replacement.splitlines()
        if _STRUCTURAL_LINE_RE.match(line.strip())
    ]
    if replacement_structure != before_structure:
        raise LatexRepairError("model replacement changed Pandoc heading structure")
    ratio = len(replacement) / max(1, len(before))
    if ratio < _MIN_RATIO or ratio > _MAX_RATIO:
        raise LatexRepairError(
            f"model replacement changed local size too aggressively ({ratio:.2f}x)"
        )


def _apply_window(tex: str, start: int, end: int, replacement: str) -> str:
    lines = str(tex or "").splitlines()
    new_lines = lines[: start - 1] + replacement.splitlines() + lines[end:]
    trailing = "\n" if str(tex or "").endswith("\n") else ""
    return "\n".join(new_lines) + trailing


def _provider_models() -> list[tuple[str, OpenAI, tuple[str, ...]]]:
    """Use known-working short-repair models; never try unsupported Flash IDs."""
    override = [
        item.strip()
        for item in os.environ.get("FICS_LATEX_REPAIR_MODELS", "").split(",")
        if item.strip()
    ]
    out = []
    for provider in config.resolve_model_providers():
        if override:
            models = list(override)
        elif provider["name"] == "modelscope":
            # Qwen3-30B is already proven usable by the math-transcript pipeline.
            # DeepSeek-V4-Flash is deliberately excluded because ModelScope has
            # returned "has no provider supported" for that model ID.
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


def repair_latex(tex: str, compiler_output: str) -> RepairResult:
    """Return one minimal local repair; Tectonic must validate it afterwards."""
    error_line = compiler_error_line(compiler_output)
    if error_line is None:
        raise LatexRepairError("compiler output does not identify a concrete .tex line")

    start, end, before = _window(tex, error_line)
    diagnostic = _error_excerpt(compiler_output, error_line)
    prompt = (
        f"Tectonic 错误：\n{diagnostic}\n\n"
        f"下面是唯一允许修改的 LaTeX 片段（原文件第 {start}--{end} 行）：\n"
        "---BEGIN LATEX---\n"
        f"{before}\n"
        "---END LATEX---\n\n"
        "返回修复后的完整片段。必须保留原片段的课程内容；只修 LaTeX 结构。"
    )

    providers = _provider_models()
    if not providers:
        raise LatexRepairError("no configured model provider is available for LaTeX repair")

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
                choice = response.choices[0]
                raw = (choice.message.content or "").strip()
                replacement = _clean_response(raw)
                if replacement is None:
                    raise LatexRepairError("model returned no safe replacement")
                _validate_replacement(before, replacement)
                candidate = _apply_window(tex, start, end, replacement)
                elapsed = time.time() - started
                print(
                    f"[LaTeX Repair] {model_id}: error line {error_line}, "
                    f"replaced local lines {start}-{end} in {elapsed:.1f}s",
                    flush=True,
                )
                return RepairResult(
                    tex=candidate,
                    audit=RepairAudit(
                        model=model_id,
                        compiler_error_line=error_line,
                        allowed_start_line=start,
                        allowed_end_line=end,
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
                print(f"[LaTeX Repair] {message}", flush=True)

    raise LatexRepairError("all LaTeX repair models failed: " + " | ".join(errors))
