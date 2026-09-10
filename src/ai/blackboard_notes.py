"""Compile chronological blackboard frame transcriptions into LaTeX notes.

This is deliberately *not* summarization.  The vision model already produced
an evidence transcription for each frame.  Here we only remove material that
is repeated across nearby frames as the lecturer keeps writing on the same
board.  New or changed lines are preserved verbatim, including every visible
proof step and LaTeX formula.
"""

from __future__ import annotations

import re
from collections import deque

from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER


BLACKBOARD_NOTES_MARKER = "### 黑板板书 LaTeX 笔记"
_FRAME_HEADING_RE = re.compile(r"^####\s+([^\n]+)\s*$", re.MULTILINE)


def _normalize(unit: str) -> str:
    """Normalize only whitespace for duplicate detection; never rewrite math."""
    return re.sub(r"\s+", "", unit).strip()


def _logical_units(body: str) -> list[str]:
    """Split one frame into conservative line/formula units.

    Display-math blocks delimited by ``$$`` stay intact.  Ordinary non-empty
    lines are separate units so an added proof line can survive even when the
    rest of the board is unchanged.
    """
    units: list[str] = []
    math_buf: list[str] = []
    in_display = False

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if in_display:
                math_buf.append("")
            continue

        dollar_count = line.count("$$")
        if in_display:
            math_buf.append(line)
            if dollar_count % 2 == 1:
                units.append("\n".join(math_buf).strip())
                math_buf = []
                in_display = False
            continue

        if dollar_count % 2 == 1:
            math_buf = [line]
            in_display = True
            continue

        units.append(line.strip())

    if math_buf:
        # Preserve malformed/incomplete visible transcription rather than drop it.
        units.append("\n".join(math_buf).strip())
    return [u for u in units if u.strip()]


def _parse_frames(markdown: str) -> list[tuple[str, str]]:
    text = (markdown or "").strip()
    if text.startswith(BLACKBOARD_SUMMARY_MARKER):
        text = text[len(BLACKBOARD_SUMMARY_MARKER):].lstrip()

    matches = list(_FRAME_HEADING_RE.finditer(text))
    frames: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        frames.append((match.group(1).strip(), body))
    return frames


def compile_blackboard_notes(markdown: str, recent_frames: int = 8) -> str:
    """Convert full-frame repetitions into faithful chronological notes.

    The algorithm is intentionally conservative:
    * exact content (modulo whitespace) repeated in recent frames is removed;
    * any changed symbol/line is retained;
    * when the visible board changes almost completely, duplicate memory is
      reset so a new board starts as a fresh note block;
    * no mathematical inference, paraphrase, correction, or compression occurs.
    """
    frames = _parse_frames(markdown)
    if not frames:
        return (markdown or "").strip()

    recent: deque[set[str]] = deque(maxlen=max(1, int(recent_frames)))
    previous: set[str] = set()
    out: list[str] = [BLACKBOARD_NOTES_MARKER, ""]
    kept_units = 0

    for timestamp, body in frames:
        units = _logical_units(body)
        current_norms = {_normalize(u) for u in units if _normalize(u)}
        if not current_norms:
            continue

        # A low-overlap state normally means the lecturer erased/replaced the
        # board or the camera moved to another board.  Start a new local memory
        # so repeated notation on the new board is not accidentally suppressed.
        if previous:
            overlap = len(current_norms & previous) / max(1, min(len(current_norms), len(previous)))
            if overlap < 0.12:
                recent.clear()

        recently_seen: set[str] = set().union(*recent) if recent else set()
        new_units = [u for u in units if _normalize(u) not in recently_seen]

        if new_units:
            out.extend([f"#### {timestamp}", "", "\n\n".join(new_units), ""])
            kept_units += len(new_units)

        recent.append(current_norms)
        previous = current_norms

    if not kept_units:
        return (markdown or "").strip()
    return "\n".join(out).strip()
