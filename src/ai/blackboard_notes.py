"""Reconstruct chronological blackboard transcriptions into LaTeX notes.

This module performs deterministic board-state reconstruction.  It does not
summarize, paraphrase, infer mathematics, or call another LLM.  Each vision
frame is already an evidence transcription; here we infer only temporal state:
what stayed on the same board, what was newly added, and when the board was
cleared/replaced.

The output is therefore suitable as a text substitute for replaying a proof-
heavy lecture: repeated full-board snapshots disappear, while new proof lines,
formula changes, corrections, and unreadable markers remain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER


BLACKBOARD_NOTES_MARKER = "### 黑板板书 LaTeX 笔记"
_FRAME_HEADING_RE = re.compile(r"^####\s+([^\n]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class _Frame:
    timestamp: str
    units: tuple[str, ...]


def _normalize(unit: str) -> str:
    """Whitespace-only canonicalization used for identity tests.

    Mathematical punctuation and symbols are deliberately untouched: changing
    ``<`` to ``<=`` or a subscript is mathematically meaningful and must not be
    hidden by fuzzy normalization.
    """
    return re.sub(r"\s+", "", unit).strip()


def _logical_units(body: str) -> list[str]:
    """Split a frame conservatively while keeping display-math blocks intact."""
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
        # Keep incomplete visible evidence rather than silently discarding it.
        units.append("\n".join(math_buf).strip())
    return [u for u in units if u.strip()]


def _parse_frames(markdown: str) -> list[_Frame]:
    text = (markdown or "").strip()
    if text.startswith(BLACKBOARD_SUMMARY_MARKER):
        text = text[len(BLACKBOARD_SUMMARY_MARKER):].lstrip()

    matches = list(_FRAME_HEADING_RE.finditer(text))
    frames: list[_Frame] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        units = tuple(_logical_units(body))
        if units:
            frames.append(_Frame(match.group(1).strip(), units))
    return frames


def _norm_set(frame: _Frame | None) -> set[str]:
    if frame is None:
        return set()
    return {_normalize(u) for u in frame.units if _normalize(u)}


def _retention(old: set[str], new: set[str]) -> float:
    """Fraction of the old visible board that remains exactly present."""
    if not old:
        return 1.0
    return len(old & new) / len(old)


def _overlap_small_side(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _is_board_reset(frames: list[_Frame], index: int) -> bool:
    """Detect a persistent erase/board switch, rejecting one-frame occlusion.

    A reset is accepted only when the following frame supports the new state
    rather than returning to the previous state.  This prevents a lecturer
    standing in front of the board from creating a false section break.
    """
    if index <= 0:
        return False

    previous = _norm_set(frames[index - 1])
    current = _norm_set(frames[index])
    following = _norm_set(frames[index + 1]) if index + 1 < len(frames) else set()
    if not previous or not current:
        return False

    direct_overlap = _overlap_small_side(previous, current)
    old_retained = _retention(previous, current)

    # Last frame cannot be validated by look-ahead; only accept a very strong
    # discontinuity there.
    if not following:
        return direct_overlap < 0.06 and old_retained < 0.12

    new_persists = _retention(current, following)
    old_returns = _retention(previous, following)

    # Near-total replacement: almost none of the old text remains, and the new
    # state persists into the next sampled frame.
    if direct_overlap < 0.10 and old_retained < 0.18:
        return new_persists >= 0.35 and old_returns < 0.30

    # Erasing often leaves a heading or one formula behind.  Detect a large,
    # persistent shrink even when the surviving small set overlaps perfectly.
    large_shrink = len(current) <= max(2, int(len(previous) * 0.55))
    if large_shrink and old_retained < 0.45:
        return new_persists >= 0.50 and old_returns < 0.55

    return False


def _extension_candidate(previous_units: tuple[str, ...], current_unit: str,
                         current_pos: int) -> str | None:
    """Find a nearby previous line that was merely extended while writing.

    We only treat literal containment as an extension; no fuzzy edit-distance
    replacement is used because a one-symbol mathematical edit can change the
    statement.  Position is restricted to the same/adjacent line to avoid
    merging two different proof steps that happen to share a long expression.
    """
    new_norm = _normalize(current_unit)
    if len(new_norm) < 10:
        return None

    lo = max(0, current_pos - 1)
    hi = min(len(previous_units), current_pos + 2)
    candidates: list[str] = []
    for old in previous_units[lo:hi]:
        old_norm = _normalize(old)
        if len(old_norm) < 8 or old_norm == new_norm:
            continue
        if old_norm in new_norm and len(old_norm) / len(new_norm) >= 0.45:
            candidates.append(old)
    if not candidates:
        return None
    return max(candidates, key=lambda u: len(_normalize(u)))


def _merge_frame_into_episode(state: list[str], previous_units: tuple[str, ...],
                              current_units: tuple[str, ...]) -> None:
    """Merge one full-board snapshot into an accumulated board episode."""
    norm_to_index = {_normalize(u): i for i, u in enumerate(state)}

    for pos, unit in enumerate(current_units):
        norm = _normalize(unit)
        if not norm or norm in norm_to_index:
            continue

        # A teacher often writes one formula over several 15-second samples.
        # Replace only a literal shorter prefix/subexpression from the previous
        # snapshot with the longer visible version.  This removes partial-write
        # duplicates without inventing or correcting any mathematics.
        old = _extension_candidate(previous_units, unit, pos)
        old_norm = _normalize(old) if old else ""
        if old_norm and old_norm in norm_to_index:
            idx = norm_to_index.pop(old_norm)
            state[idx] = unit
            norm_to_index[norm] = idx
            continue

        # Otherwise it is genuinely new/changed evidence. Preserve it verbatim.
        norm_to_index[norm] = len(state)
        state.append(unit)


def _render_episode(number: int, start_ts: str, end_ts: str,
                    units: list[str]) -> list[str]:
    if not units:
        return []
    heading = f"#### 板书片段 {number}（{start_ts}–{end_ts}）"
    return [heading, "", "\n\n".join(units), ""]


def compile_blackboard_notes(markdown: str) -> str:
    """Reconstruct full-board snapshots into continuous, faithful LaTeX notes.

    Semantics:
    - exact repeated material is emitted once per physical board episode;
    - incremental writing extends a partial line when literal containment proves
      that it is the same line being completed;
    - changed statements/formulas are preserved as separate evidence;
    - persistent erase/board-switch events start a new board episode;
    - no LLM, mathematical inference, paraphrase, correction, or summarization.
    """
    frames = _parse_frames(markdown)
    if not frames:
        return (markdown or "").strip()

    out: list[str] = [BLACKBOARD_NOTES_MARKER, ""]
    episode_no = 1
    episode_start = frames[0].timestamp
    episode_end = frames[0].timestamp
    episode_state: list[str] = []
    previous_units: tuple[str, ...] = tuple()

    for i, frame in enumerate(frames):
        if i > 0 and _is_board_reset(frames, i):
            out.extend(
                _render_episode(
                    episode_no,
                    episode_start,
                    episode_end,
                    episode_state,
                )
            )
            episode_no += 1
            episode_start = frame.timestamp
            episode_state = []
            previous_units = tuple()

        _merge_frame_into_episode(episode_state, previous_units, frame.units)
        previous_units = frame.units
        episode_end = frame.timestamp

    out.extend(
        _render_episode(
            episode_no,
            episode_start,
            episode_end,
            episode_state,
        )
    )

    # If every parsed frame was somehow empty after normalization, fall back to
    # the raw evidence rather than emitting a misleading blank note.
    if len(out) <= 2:
        return (markdown or "").strip()
    return "\n".join(out).strip()
