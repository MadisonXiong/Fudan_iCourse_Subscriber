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
_INLINE_MATH_RE = re.compile(r'\$(?!\$)\s*([^$\n]+?)\s*\$(?!\$)')


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
    """Make ``$ formula $`` parse as Pandoc math without changing the formula."""

    def repl(match: re.Match) -> str:
        return "$" + match.group(1).strip() + "$"

    return _INLINE_MATH_RE.sub(repl, text)


def _looks_like_raw_math(text: str) -> bool:
    """Detect model output that forgot math delimiters but is clearly TeX math."""
    return bool(
        re.search(
            r"\\(?:frac|sum|int|lim|forall|exists|Rightarrow|Leftrightarrow|"
            r"langle|rangle|mathbb|ell|infty|varepsilon|begin|left|right|"
            r"subset|in|to|geq|leq|cdot|overline)\b|[_^]",
            text,
        )
    )


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
    text = _AI_NOTE_RE.sub(_ai_note, text)
    text = _VISUAL_RE.sub(_visual_restore, text)
    text = _VIDEO_LINE_RE.sub(_video_location, text)
    return text.strip()


def compose_course_markdown(
    summary: str,
    *,
    math_transcript: str = "",
) -> str:
    """Build the Markdown body for one course-note PDF.

    Keep the existing course notes intact, then append the complete readable
    math-enhanced transcript. The conservative proofread transcript remains a
    separate audit attachment managed by ``Emailer``.
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
