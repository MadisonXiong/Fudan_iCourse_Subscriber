"""Normalize FiCS Markdown annotations before Pandoc/LaTeX conversion.

Stored notes contain small HTML provenance markers.  Convert them into Pandoc
semantic structures while preserving real math nodes.  In particular, visual
restoration output from an LLM may contain ``$ formula $`` (spaces immediately
inside the delimiters) or block ``$$...$$``.  Pandoc does not reliably parse
those forms inside an inline bracketed span, which previously turned dollars
and underscores into escaped text and produced invalid LaTeX.  We normalize
inline math and represent display-math visual restorations as fenced Divs.
"""

from __future__ import annotations

import re


_AI_NOTE_RE = re.compile(
    r'<div\s+data-ai-note="true"[^>]*>(.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
_VISUAL_RE = re.compile(
    r'<span\s+data-visual-restored="true"[^>]*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)
_AI_LABEL_RE = re.compile(
    r'<strong[^>]*>\s*AI\s*补充\s*</strong>\s*(?:<br\s*/?>)?',
    re.IGNORECASE,
)
_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
_VIDEO_LINE_RE = re.compile(
    r'^\s*\*\*视频定位：(.+?)\*\*\s*$',
    re.MULTILINE,
)
_VIS_LABEL_RE = re.compile(r'^\s*〔视觉补全〕\s*')
_DISPLAY_MATH_RE = re.compile(r'^\s*\$\$\s*(.*?)\s*\$\$\s*$', re.DOTALL)
_INLINE_MATH_RE = re.compile(
    r'(?<![\\$])\$(?!\$)([^$\n]*?)(?<!\\)\$(?!\$)'
)
_DISPLAY_BLOCK_RE = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
_BRACKETED_VISUAL_RE = re.compile(
    r'\[(.*?)\]\{\.visual-restored\}', re.DOTALL
)
_SAFE_SPLIT_MATH_TEXT_RE = re.compile(
    r'^[\s\u3000\u4e00-\u9fffA-Za-z0-9，。；：、（）()“”‘’·,.!?+\-]+$'
)


def _clean_ai_inner(inner: str) -> str:
    inner = _AI_LABEL_RE.sub("", inner, count=1)
    inner = _BR_RE.sub("\n", inner)
    # The only strong/em tags expected inside these annotation blocks are
    # presentation markup. Preserve their text and let Markdown/LaTeX handle it.
    inner = re.sub(r'</?(?:strong|em)[^>]*>', "", inner, flags=re.IGNORECASE)
    return inner.strip()


def _ai_note(match: re.Match) -> str:
    inner = _clean_ai_inner(match.group(1))
    return f"\n\n::: {{.ai-note}}\n{inner}\n:::\n\n"


def _normalize_inline_math(text: str) -> str:
    """Make ``$ formula $`` parse as Pandoc math without changing the formula.

    Pandoc deliberately treats spaces immediately inside dollar delimiters as
    literal text.  Math-enhanced transcripts produced by an LLM commonly use
    that readable-but-incompatible spelling.  Normalize every *single-dollar*
    inline span before Pandoc sees it, while leaving ``$$`` display math and
    escaped currency dollars untouched.
    """

    def repl(match: re.Match) -> str:
        return "$" + match.group(1).strip() + "$"

    return _INLINE_MATH_RE.sub(repl, text)


def _delimiter_balance(formula: str) -> int:
    """Return the unmatched ``\\left`` count in one TeX fragment."""
    return len(re.findall(r"\\left\b", formula)) - len(
        re.findall(r"\\right\b", formula)
    )


def _plain_gap_as_tex(gap: str) -> str | None:
    """Convert short prose between accidentally split math blocks to TeX text."""
    compact = re.sub(r"\s+", " ", str(gap or "")).strip()
    if not compact:
        return ""
    if len(compact) > 160 or not _SAFE_SPLIT_MATH_TEXT_RE.fullmatch(compact):
        return None
    escaped = (
        compact.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )
    return rf"\text{{{escaped}}}"


def _merge_split_display_delimiters(text: str) -> str:
    """Merge display blocks that accidentally split one left/right formula.

    OCR/LLM output sometimes emits one set-builder expression as three display
    blocks, such as ``$$\\left\\{f$$ 为 $$X$$ 上可测且
    $$...\\right\\}$$``. Pandoc creates three independent displays and
    Tectonic rejects the unmatched delimiters. This deterministic repair keeps
    every math fragment and places only short intervening prose in ``\\text{}``.
    """
    source = str(text or "")
    blocks = list(_DISPLAY_BLOCK_RE.finditer(source))
    if not blocks:
        return source

    out: list[str] = []
    cursor = 0
    i = 0
    while i < len(blocks):
        first = blocks[i]
        balance = _delimiter_balance(first.group(1))
        if balance <= 0:
            i += 1
            continue

        pieces = [first.group(1).strip()]
        j = i
        safe = True
        while balance > 0 and j + 1 < len(blocks):
            following = blocks[j + 1]
            gap_tex = _plain_gap_as_tex(source[blocks[j].end():following.start()])
            if gap_tex is None:
                safe = False
                break
            if gap_tex:
                pieces.append(gap_tex)
            pieces.append(following.group(1).strip())
            balance += _delimiter_balance(following.group(1))
            j += 1

        if not safe or balance != 0:
            i += 1
            continue

        out.append(source[cursor:first.start()])
        out.append("$$\n" + " ".join(piece for piece in pieces if piece) + "\n$$")
        cursor = blocks[j].end()
        i = j + 1

    if not out:
        return source
    out.append(source[cursor:])
    return "".join(out)


def _looks_like_raw_math(text: str) -> bool:
    """Detect model output that forgot math delimiters but is clearly TeX math."""
    return bool(
        re.search(
            r"\\(?:frac|sum|int|lim|forall|exists|Rightarrow|Leftrightarrow|"
            r"langle|rangle|mathbb|ell|infty|varepsilon|begin|left|right|"
            r"subset|in|to|geq|leq|cdot|overline|iff|Leftrightarrow)\b|[_^]",
            text,
        )
    )


def _single_dollar_positions(text: str) -> list[int]:
    """Return unescaped single-dollar delimiters, excluding ``$$`` pairs."""
    positions: list[int] = []
    for i, char in enumerate(text):
        if char != "$":
            continue
        if i and text[i - 1] == "\\":
            continue
        if (i and text[i - 1] == "$") or (i + 1 < len(text) and text[i + 1] == "$"):
            continue
        positions.append(i)
    return positions


def _repair_visual_math_delimiters(inner: str) -> str:
    """Repair a lost edge delimiter in a visual formula without changing it.

    A real cached transcript contained ``[\\Rightarrow ... f(x)$]{...}``:
    the model had dropped only the opening dollar.  Pandoc interpreted the
    remaining dollar as literal text and generated invalid LaTeX.  Visual
    formulas are provenance-bounded, so adding the missing opposite edge is a
    deterministic representation repair, not a mathematical edit.
    """
    stripped = inner.strip()
    dollars = _single_dollar_positions(stripped)
    if len(dollars) % 2 == 0 or not _looks_like_raw_math(stripped):
        return inner
    if dollars == [0]:
        return stripped + "$"
    if dollars == [len(stripped) - 1]:
        return "$" + stripped
    return inner


def _normalize_bracketed_visual(match: re.Match) -> str:
    r"""Add math delimiters to an already-semantic bare visual formula.

    Cached enhanced transcripts can already contain Pandoc bracketed spans,
    bypassing :func:`_visual_restore`.  A bare ``\iff`` or norm expression in
    such a span would otherwise place math-only commands in LaTeX text mode.
    """
    inner = _repair_visual_math_delimiters(match.group(1))
    if "$" not in inner and _looks_like_raw_math(inner):
        return f"[${inner.strip()}$]{{.visual-restored}}"
    return f"[{inner}]{{.visual-restored}}"


def _visual_restore(match: re.Match) -> str:
    """Convert one visual-provenance HTML span into safe Pandoc structures.

    Display math cannot legally live inside Pandoc's inline ``Span`` node, so a
    visual restoration containing exactly one ``$$...$$`` block becomes a
    fenced Div.  Inline formulas remain spans, allowing the Lua filter to color
    them and add the provenance label without rasterizing anything.
    """
    inner = _BR_RE.sub(" ", match.group(1)).strip()
    inner = _VIS_LABEL_RE.sub("", inner, count=1).strip()
    if not inner:
        return ""

    display = _DISPLAY_MATH_RE.match(inner)
    if display:
        formula = display.group(1).strip()
        return (
            "\n\n::: {.visual-restored-block}\n"
            "$$\n"
            f"{formula}\n"
            "$$\n"
            ":::\n\n"
        )

    inner = _normalize_inline_math(inner)
    inner = _repair_visual_math_delimiters(inner)

    # A few model responses violate the contract by returning bare TeX such as
    # ``\Rightarrow ...``.  Treat it as math rather than allowing Pandoc to
    # escape underscores/backslashes into text.  This only adds delimiters; it
    # does not invent or rewrite mathematical content.
    if "$" not in inner and _looks_like_raw_math(inner):
        inner = f"${inner.strip()}$"

    # Do not repeat the human-visible label here.  \visualrestore itself adds
    # 〔视觉补全〕, while the Span contains only the restored source content.
    return f"[{inner}]{{.visual-restored}}"


def _video_location(match: re.Match) -> str:
    label = match.group(1).strip()
    return (
        "::: {.video-location}\n"
        f"**视频定位：{label}**\n"
        ":::"
    )


def preprocess_markdown(text: str) -> str:
    """Convert FiCS-specific HTML annotations into Pandoc semantic classes."""
    text = str(text or "").replace("\r\n", "\n")
    text = _merge_split_display_delimiters(text)
    # This must apply to the whole transcript, not only to purple visual
    # restoration spans.  Otherwise ``$ A \subseteq X $`` becomes literal
    # ``\$ A \subseteq X \$`` in Pandoc's TeX and cannot compile.
    text = _normalize_inline_math(text)
    text = _AI_NOTE_RE.sub(_ai_note, text)
    text = _VISUAL_RE.sub(_visual_restore, text)
    text = _BRACKETED_VISUAL_RE.sub(_normalize_bracketed_visual, text)
    text = _VIDEO_LINE_RE.sub(_video_location, text)
    return text.strip()


def compose_course_markdown(
    summary: str,
    *,
    math_transcript: str = "",
) -> str:
    """Build the Markdown body for one course-note PDF.

    Keep the existing course notes intact, then append the complete readable
    math-enhanced transcript. The conservative proofread transcript remains
    persisted as an audit source, rather than becoming a second email file.
    """
    parts = [preprocess_markdown(summary)]
    if str(math_transcript or "").strip():
        parts.extend(
            [
                "\\newpage",
                preprocess_markdown(math_transcript),
            ]
        )
    return "\n\n".join(part for part in parts if part).strip() + "\n"
