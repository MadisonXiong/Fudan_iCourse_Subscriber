"""Final local quality gate for blackboard notes.

The semantic editor can produce an otherwise excellent lecture with one or two
local defects: an unsupported ``cases`` environment, a naked LaTeX command, or
an OCR/model placeholder such as ``##C##`` / ``xxxx``. Re-running the whole
lecture for those defects is both expensive and risky.

This module therefore separates two kinds of repair:

* deterministic format-only repairs, where no mathematical judgement is needed;
  in particular Chinese inside ``\\text{...}`` is moved out of the math span;
* evidence-constrained LLM repair for content/structure defects such as OCR
  placeholders, broken environments or genuinely malformed LaTeX.

Only small problematic chunks are ever sent back to the model. Correct chunks
are never rewritten. Hard failure is reserved for defects that can still make
the document misleading or structurally unsafe after the available repairs;
a harmless, mechanically repairable ``\\text{中文}`` can no longer abort an
otherwise valid lecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_DEFAULT_CHUNK_CHARS = 4200
_DEFAULT_CONTEXT_CHARS = 1400
_DEFAULT_EVIDENCE_CHARS = 7000
_MAX_REPAIR_ATTEMPTS = 2

ISSUE_FORBIDDEN_ENV = (
    "forbidden multi-line LaTeX environment (cases/aligned/array/gathered/split)"
)
ISSUE_CJK_TEXT = "Chinese text inside \\text{...}"
ISSUE_PLACEHOLDER = (
    "suspicious transcription placeholder (xxxx/[unclear]/replacement chars)"
)
ISSUE_DOUBLE_HASH = "suspicious ## token outside a Markdown heading"
ISSUE_BARE_LATEX = "bare LaTeX command outside math delimiters"
ISSUE_UNBALANCED_DOLLAR = "unbalanced dollar delimiter"
ISSUE_OVERSIZED_DISPLAY = "oversized display formula (>700 chars)"
ISSUE_UNCLOSED_AI = "unclosed AI supplement HTML block"

# Environments that are fragile in the current email renderer and unnecessary
# for these notes. The repair model rewrites them as bullets / short displays.
_FORBIDDEN_ENV_RE = re.compile(
    r"\\(?:begin|end)\s*\{(?:cases|aligned|array|gathered|split)\}",
    re.IGNORECASE,
)

# CodeCogs does not reliably render CJK glyphs in \text{...}. This pattern is
# deliberately limited to non-nested braces so a mechanical split cannot alter
# the mathematical parse of an arbitrarily complicated command.
_CJK_TEXT_COMMAND_RE = re.compile(
    r"\\text\s*\{([^{}]*[\u4e00-\u9fff][^{}]*)\}",
    re.DOTALL,
)

# Common obvious transcription/model artefacts. ``[转写存疑]`` is deliberately
# NOT included: that is our safe, human-readable fallback when evidence is truly
# ambiguous.
_PLACEHOLDER_RE = re.compile(
    r"(?:\[unclear\]|\b[xX]{4,}\b|\ufffd|(?:\?\s*){4,})",
    re.IGNORECASE,
)

# LaTeX commands that must not appear outside a math delimiter.
_BARE_LATEX_RE = re.compile(
    r"\\(?:mathbb|mathcal|mathrm|mathbf|operatorname|frac|sqrt|sum|"
    r"prod|int|lim|sup|inf|max|min|to|in|notin|subset|subseteq|supset|"
    r"supseteq|leq|geq|leqslant|geqslant|neq|forall|exists|varepsilon|"
    r"epsilon|lambda|mu|nu|delta|alpha|beta|gamma|Rightarrow|Leftrightarrow|"
    r"leftarrow|rightarrow|mapsto|infty|cdot|times|Vert|lVert|rVert|"
    r"langle|rangle|overline|bar|hat|begin|end)\b"
)

_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)
_FRAME_RE = re.compile(r"(?=^####\s+[^\n]+$)", re.MULTILINE)
_REF_RE = re.compile(r"\b\d+(?:\.\d+){1,4}\b")
_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_LATIN_RE = re.compile(r"\b[A-Za-z]{2,}\b")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


FINAL_REPAIR_SYSTEM_PROMPT = r"""
你是“数学板书最终局部修复器”。整节课已经完成全局去重和整理；现在只有一个很小的片段被程序检测到格式或转写异常。

你会收到：
1. 程序检测到的异常类型；
2. 待修复片段前后的只读上下文；
3. 从原始逐帧视觉转写中自动检索出的相关证据帧；
4. 待修复片段本身。

你的权限严格局限于“待修复片段”。直接输出修复后的该片段，不要输出上下文、证据或解释。

必须遵守：
- 保留片段中已经正确的数学内容、顺序、例号、证明步骤和 AI 补充；不要重新总结或改写整段课程。
- 对 `##C##`、`xxxx`、`[unclear]`、乱码等转写污染：优先根据原始证据帧和前后上下文恢复最有证据支持的原表达。不能可靠判断时写 `[转写存疑]`，不要猜。
- 对 `cases` / `aligned` / `array` / `gathered` / `split`：保持数学含义不变，改写成普通 Markdown 条目、短行内公式或多个独立的 `$$...$$`，不得继续使用这些环境。
- 所有 LaTeX 命令都必须放在 `$...$` 或 `$$...$$` 中，不能裸露。
- 不要在 `\\text{...}` 中放中文；把中文移到数学环境外。
- 不得产生 `##...##`、`xxxx`、`[unclear]`、乱码占位符。
- 不使用 Markdown 代码块。
- 若片段含 `<div data-ai-note="true" ...>...</div>`，必须保留其 AI 补充身份和紫色 inline style；不得把 AI 内容混入老师板书正文。
- 除已有 AI 补充外，本次局部修复不要新增新的解释性内容。

只输出修复后的片段正文。
""".strip()


@dataclass(frozen=True)
class RepairResult:
    text: str
    models: list[str]
    repaired_chunks: int


def _outside_math(text: str) -> str:
    masked = _DISPLAY_RE.sub(" ", text)
    masked = _INLINE_RE.sub(" ", masked)
    return masked


def _has_nonheading_double_hash(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.lstrip()
        if re.match(r"^#{3,5}\s+\S", stripped):
            rest = re.sub(r"^#{3,5}\s+", "", stripped, count=1)
            if "##" in rest:
                return True
            continue
        if "##" in line:
            return True
    return False


def _contains_cjk_text_in_math(text: str) -> bool:
    """True only when a CJK ``\\text{}`` actually occurs inside math."""
    for match in _DISPLAY_RE.finditer(text):
        if _CJK_TEXT_COMMAND_RE.search(match.group(1)):
            return True
    # Mask display blocks before scanning inline math so their dollar signs do
    # not participate in the inline expression.
    masked = _DISPLAY_RE.sub(" ", text)
    return any(
        _CJK_TEXT_COMMAND_RE.search(match.group(1))
        for match in _INLINE_RE.finditer(masked)
    )


def _split_cjk_text_math_content(content: str, delimiter: str) -> tuple[str, int]:
    """Move plain CJK text out of one math span without changing math tokens.

    Example (inline): ``$x>0, \\text{其中 }x\\in X$`` becomes
    ``$x>0,$ 其中 $x\\in X$``.

    Display math is split into short display blocks separated by plain text.
    Only the literal wrapper ``\\text{...}`` is removed; its inner text and all
    surrounding mathematical source are preserved verbatim apart from boundary
    whitespace.
    """
    matches = list(_CJK_TEXT_COMMAND_RE.finditer(content))
    if not matches:
        return f"{delimiter}{content}{delimiter}", 0

    pieces: list[tuple[str, str]] = []
    cursor = 0
    count = 0
    for match in matches:
        math_part = content[cursor:match.start()].strip()
        if math_part:
            pieces.append(("math", math_part))
        plain_part = match.group(1).strip()
        if plain_part:
            pieces.append(("text", plain_part))
        cursor = match.end()
        count += 1
    tail = content[cursor:].strip()
    if tail:
        pieces.append(("math", tail))

    if delimiter == "$$":
        rendered: list[str] = []
        for kind, value in pieces:
            if kind == "math":
                rendered.append(f"$${value}$$")
            else:
                rendered.append(value)
        return "\n\n".join(rendered), count

    rendered_inline: list[str] = []
    for kind, value in pieces:
        if kind == "math":
            rendered_inline.append(f"${value}$")
        else:
            # Spaces make the source readable while remaining semantically
            # neutral in Markdown/HTML.
            rendered_inline.append(f" {value} ")
    return "".join(rendered_inline).strip(), count


def _sanitize_cjk_text_in_math(text: str) -> tuple[str, int]:
    """Deterministically eliminate CJK ``\\text{}`` from math spans.

    This is a renderer compatibility transformation, not a mathematical edit,
    so it is safe to use as a non-LLM fallback.
    """
    count = 0

    def replace_display(match: re.Match) -> str:
        nonlocal count
        replaced, n = _split_cjk_text_math_content(match.group(1), "$$")
        count += n
        return replaced

    # Display math first. Replacements stay wrapped in $$...$$, so the inline
    # regex below will continue to ignore them.
    result = _DISPLAY_RE.sub(replace_display, text)

    def replace_inline(match: re.Match) -> str:
        nonlocal count
        replaced, n = _split_cjk_text_math_content(match.group(1), "$")
        count += n
        return replaced

    result = _INLINE_RE.sub(replace_inline, result)
    return result, count


def scan_anomalies(text: str) -> list[str]:
    """Return concrete final-output defects.

    Detection targets renderer failures and obvious placeholders, not stylistic
    preferences. CJK text is only flagged when it occurs inside a math span.
    """
    issues: list[str] = []
    if not text.strip():
        return ["empty output"]

    if _FORBIDDEN_ENV_RE.search(text):
        issues.append(ISSUE_FORBIDDEN_ENV)

    if _contains_cjk_text_in_math(text):
        issues.append(ISSUE_CJK_TEXT)

    if _PLACEHOLDER_RE.search(text):
        issues.append(ISSUE_PLACEHOLDER)

    if _has_nonheading_double_hash(text):
        issues.append(ISSUE_DOUBLE_HASH)

    outside = _outside_math(text)
    if _BARE_LATEX_RE.search(outside):
        issues.append(ISSUE_BARE_LATEX)

    if text.count("$") % 2:
        issues.append(ISSUE_UNBALANCED_DOLLAR)

    if any(len(block) > 700 for block in _DISPLAY_RE.findall(text)):
        issues.append(ISSUE_OVERSIZED_DISPLAY)

    ai_open = text.count('data-ai-note="true"')
    if ai_open and text.count("</div>") < ai_open:
        issues.append(ISSUE_UNCLOSED_AI)

    return issues


def _atomic_blocks(text: str) -> list[str]:
    """Split on blank lines while keeping an AI supplement div atomic."""
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    blocks: list[str] = []
    pending: list[str] = []
    in_ai = False

    for para in paras:
        starts_ai = 'data-ai-note="true"' in para
        if starts_ai and not in_ai:
            in_ai = True
            pending = [para]
            if "</div>" in para:
                blocks.append("\n\n".join(pending))
                pending = []
                in_ai = False
            continue

        if in_ai:
            pending.append(para)
            if "</div>" in para:
                blocks.append("\n\n".join(pending))
                pending = []
                in_ai = False
            continue

        blocks.append(para)

    if pending:
        blocks.append("\n\n".join(pending))
    return blocks


def _pack_blocks(blocks: list[str], target_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        addition = len(block) + (2 if current else 0)
        if current and current_len + addition > target_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _token_features(text: str) -> set[str]:
    features: set[str] = set()
    features.update(f"ref:{x}" for x in _REF_RE.findall(text))
    features.update(f"cmd:{x}" for x in _COMMAND_RE.findall(text))
    features.update(f"lat:{x.lower()}" for x in _LATIN_RE.findall(text))

    for run in _CJK_RUN_RE.findall(text):
        for n in (2, 3):
            if len(run) >= n:
                features.update(
                    f"zh:{run[i:i+n]}" for i in range(len(run) - n + 1)
                )
    return features


def _feature_weight(feature: str) -> float:
    if feature.startswith("ref:"):
        return 8.0
    if feature.startswith("cmd:"):
        return 1.5
    if feature.startswith("zh:"):
        return 1.0
    return 0.6


def _raw_frames(raw: str) -> list[str]:
    parts = [p.strip() for p in _FRAME_RE.split(raw) if p.strip()]
    return parts if parts else ([raw.strip()] if raw.strip() else [])


def _retrieve_raw_evidence(
    raw: str,
    query: str,
    max_chars: int = _DEFAULT_EVIDENCE_CHARS,
) -> str:
    """Retrieve a few raw vision frames relevant to a bad final chunk."""
    q_features = _token_features(query)
    if not q_features:
        return "（未检索到可靠原始证据片段）"

    scored: list[tuple[float, str]] = []
    for frame in _raw_frames(raw):
        f_features = _token_features(frame)
        common = q_features & f_features
        if not common:
            continue
        score = sum(_feature_weight(feature) for feature in common)
        if any(feature.startswith("ref:") for feature in common):
            score *= 1.35
        scored.append((score, frame))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    used = 0
    for _, frame in scored[:6]:
        if selected and used + len(frame) + 2 > max_chars:
            continue
        selected.append(frame)
        used += len(frame) + 2
        if used >= max_chars:
            break

    return "\n\n".join(selected) if selected else "（未检索到可靠原始证据片段）"


def _context(chunks: list[str], index: int, before: bool, chars: int) -> str:
    if before:
        if index <= 0:
            return "（无）"
        return chunks[index - 1][-chars:]
    if index + 1 >= len(chunks):
        return "（无）"
    return chunks[index + 1][:chars]


def _call_local_repair(editor, prompt: str):
    """Call the parent editor without importing its private result type."""
    return editor._call(  # noqa: SLF001 - intentional narrow adapter
        FINAL_REPAIR_SYSTEM_PROMPT,
        prompt,
        max_tokens=9000,
    )


def _apply_safe_format_fallback(text: str) -> tuple[str, int]:
    """Apply all currently supported deterministic, semantics-free fixes."""
    return _sanitize_cjk_text_in_math(text)


def repair_final_anomalies(
    editor,
    text: str,
    raw_blackboard: str,
    *,
    target_chars: int = _DEFAULT_CHUNK_CHARS,
) -> RepairResult:
    """Repair only chunks that fail final render/transcription validation.

    Format-only defects that can be corrected mechanically are handled without
    an LLM and are nonfatal. Content/structural defects still use evidence-based
    local LLM repair and still fail closed if they remain unresolved.
    """
    document_issues = scan_anomalies(text)
    if not document_issues:
        print(
            "[BlackboardFinalizer] preflight: clean; no local repair needed",
            flush=True,
        )
        return RepairResult(text=text, models=[], repaired_chunks=0)

    # If CJK \text{} is the only problem, there is no reason to involve an LLM.
    # Moving the literal Chinese out of the math delimiters preserves every
    # token while avoiding the CodeCogs font/renderer failure seen in production.
    if set(document_issues) == {ISSUE_CJK_TEXT}:
        sanitized, count = _apply_safe_format_fallback(text)
        remaining = scan_anomalies(sanitized)
        if not remaining:
            print(
                f"[BlackboardFinalizer] deterministic format cleanup moved "
                f"{count} CJK \\text{{...}} span(s) outside math; preflight passed",
                flush=True,
            )
            return RepairResult(
                text=sanitized,
                models=[],
                repaired_chunks=1 if count else 0,
            )

    print(
        "[BlackboardFinalizer] preflight found: " + "; ".join(document_issues),
        flush=True,
    )

    chunks = _pack_blocks(_atomic_blocks(text), target_chars)
    models: list[str] = []
    repaired_count = 0

    for index in range(len(chunks)):
        issues = scan_anomalies(chunks[index])
        if not issues:
            continue

        # Same fast path at chunk scope. This fixes the exact failure mode from
        # the previous Single Run without risking a rewrite of a 4k-char chunk.
        if set(issues) == {ISSUE_CJK_TEXT}:
            sanitized, count = _apply_safe_format_fallback(chunks[index])
            if count and not scan_anomalies(sanitized):
                chunks[index] = sanitized
                repaired_count += 1
                print(
                    f"[BlackboardFinalizer] chunk {index + 1}/{len(chunks)}: "
                    f"deterministically fixed {count} CJK \\text{{...}} span(s)",
                    flush=True,
                )
                continue

        original = chunks[index]
        evidence = _retrieve_raw_evidence(raw_blackboard, original)
        before = _context(chunks, index, True, _DEFAULT_CONTEXT_CHARS)
        after = _context(chunks, index, False, _DEFAULT_CONTEXT_CHARS)
        candidate = original
        resolved = False

        for attempt in range(1, _MAX_REPAIR_ATTEMPTS + 1):
            current_issues = scan_anomalies(candidate)
            prompt = (
                "【程序检测到的异常】\n- "
                + "\n- ".join(current_issues)
                + "\n\n【前文，只读，不要输出】\n"
                + before
                + "\n\n【后文，只读，不要输出】\n"
                + after
                + "\n\n【相关原始逐帧板书证据，只用于核对内容】\n"
                + evidence
                + "\n\n【待修复片段，只输出这一部分的修复结果】\n"
                + candidate
            )
            result = _call_local_repair(editor, prompt)
            models.append(result.model_id)

            # Even a good LLM repair may leave a harmless \text{中文}; apply the
            # deterministic compatibility rewrite before judging the attempt.
            cleaned_result, cjk_fixed = _apply_safe_format_fallback(
                result.text.strip()
            )
            ratio = len(cleaned_result) / max(1, len(original))
            new_issues = scan_anomalies(cleaned_result)
            safe_size = 0.55 <= ratio <= 1.65
            not_truncated = result.finish_reason.lower() != "length"

            if not_truncated and safe_size and not new_issues:
                candidate = cleaned_result
                resolved = True
                print(
                    f"[BlackboardFinalizer] chunk {index + 1}/{len(chunks)} "
                    f"repaired on attempt {attempt} "
                    f"({len(original)} -> {len(candidate)} chars"
                    + (f", plus {cjk_fixed} deterministic CJK fix(es)" if cjk_fixed else "")
                    + ")",
                    flush=True,
                )
                break

            print(
                f"[BlackboardFinalizer] chunk {index + 1}/{len(chunks)} "
                f"attempt {attempt} not clean "
                f"(retention={ratio:.1%}, remaining={new_issues or ['unsafe size/truncation']})",
                flush=True,
            )
            if (
                not_truncated
                and safe_size
                and len(new_issues) <= len(current_issues)
            ):
                candidate = cleaned_result

        if not resolved:
            # One final semantics-free fallback is allowed. If the only thing
            # left is CJK \text{}, fix it and continue. Any content/structural
            # problem still fails closed.
            fallback, count = _apply_safe_format_fallback(candidate)
            fallback_issues = scan_anomalies(fallback)
            if count and not fallback_issues:
                candidate = fallback
                resolved = True
                print(
                    f"[BlackboardFinalizer] chunk {index + 1}/{len(chunks)}: "
                    f"LLM left only a format issue; deterministic fallback "
                    f"fixed {count} CJK \\text{{...}} span(s)",
                    flush=True,
                )

        if not resolved:
            raise RuntimeError(
                "Final blackboard preflight could not repair chunk "
                f"{index + 1}/{len(chunks)}; "
                f"remaining={scan_anomalies(candidate) or issues}"
            )

        chunks[index] = candidate
        repaired_count += 1

    repaired = "\n\n".join(chunks).strip()
    repaired, final_cjk_fixed = _apply_safe_format_fallback(repaired)
    if final_cjk_fixed:
        print(
            f"[BlackboardFinalizer] final deterministic sweep fixed "
            f"{final_cjk_fixed} CJK \\text{{...}} span(s)",
            flush=True,
        )

    remaining = scan_anomalies(repaired)
    if remaining:
        raise RuntimeError(
            "Final blackboard preflight still has unresolved hard issues: "
            + "; ".join(remaining)
        )

    print(
        f"[BlackboardFinalizer] preflight passed after repairing "
        f"{repaired_count} local chunk(s)",
        flush=True,
    )
    return RepairResult(
        text=repaired,
        models=models,
        repaired_chunks=repaired_count,
    )
