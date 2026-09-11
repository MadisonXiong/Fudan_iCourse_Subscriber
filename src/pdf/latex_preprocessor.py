"""Normalize FiCS Markdown annotations before Pandoc/LaTeX conversion.

The stored notes intentionally contain a small amount of HTML so email/web
renderers can distinguish teacher content from model-authored additions.  Pandoc
can preserve the same semantics more reliably if those HTML markers are first
converted to fenced Divs/Spans with explicit classes.  A Lua filter then maps
those classes to LaTeX environments/macros.
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


def _clean_ai_inner(inner: str) -> str:
    inner = _AI_LABEL_RE.sub("", inner, count=1)
    inner = _BR_RE.sub("\n", inner)
    # The only strong/em tags expected inside these annotation blocks are
    # presentation markup.  Preserve their text and let Markdown/LaTeX handle it.
    inner = re.sub(r'</?(?:strong|em)[^>]*>', "", inner, flags=re.IGNORECASE)
    return inner.strip()


def _ai_note(match: re.Match) -> str:
    inner = _clean_ai_inner(match.group(1))
    return f"\n\n::: {{.ai-note}}\n{inner}\n:::\n\n"


def _visual_restore(match: re.Match) -> str:
    inner = _BR_RE.sub(" ", match.group(1)).strip()
    inner = re.sub(r'^\s*〔视觉补全〕\s*', "", inner)
    if not inner:
        return ""
    return f"[〔视觉补全〕{inner}]{{.visual-restored}}"


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

    The faithful ASR transcript intentionally stays outside the PDF as a separate
    audit attachment.  Only the evidence-constrained math-enhanced transcript is
    appended to the reading PDF.
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
