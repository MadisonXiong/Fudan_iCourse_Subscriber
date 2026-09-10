"""Stitch chronological blackboard snapshots into non-repetitive LaTeX notes.

The vision model transcribes each selected frame accurately, but a physical
blackboard can move up and down.  The same written material therefore appears
at different vertical positions in many consecutive frames.  This module
stitches those snapshots by CONTENT rather than by line position.

No LLM is called here.  We do not summarize or infer mathematics.  We only:
1. split large ``aligned`` snapshots into logical rows;
2. recognize the same row even when it moved vertically or formatting changed;
3. replace a visibly partial row with its later, longer completion;
4. preserve genuinely new proof steps in first-appearance order; and
5. start a new board segment only after a persistent content discontinuity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER


BLACKBOARD_NOTES_MARKER = "### 黑板板书 LaTeX 笔记"
_FRAME_HEADING_RE = re.compile(r"^####\s+([^\n]+)\s*$", re.MULTILINE)
_ALIGNED_RE = re.compile(
    r"(?:\$\$|\\\[)?\s*\\begin\{aligned\}(.*?)\\end\{aligned\}\s*(?:\$\$|\\\])?",
    re.DOTALL,
)
_TEXT_RE = re.compile(r"\\text\{([^{}]*)\}")


@dataclass(frozen=True)
class _Frame:
    timestamp: str
    units: tuple[str, ...]


def _split_aligned(unit: str) -> list[str]:
    """Split one aligned environment into row-sized display-math units.

    Vision sometimes emits an entire board as a single aligned environment.
    Treating that as one unit defeats de-duplication and creates enormous
    rendering URLs.  Splitting on LaTeX row separators preserves the written
    content while allowing rows to match the same rows in later frames.
    """
    match = _ALIGNED_RE.fullmatch(unit.strip())
    if not match:
        return [unit.strip()]

    body = match.group(1)
    rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", body)
    out: list[str] = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        # '&' is alignment syntax, not mathematical content.  Once a row is
        # taken out of aligned, leaving '&' would make standalone LaTeX invalid.
        row = row.replace("&", "").strip()
        if row:
            out.append(f"$$\n{row}\n$$")
    return out or [unit.strip()]


def _logical_units(body: str) -> list[str]:
    """Split a frame conservatively while keeping display math intact."""
    raw_units: list[str] = []
    math_buf: list[str] = []
    in_display = False
    aligned_buf: list[str] = []
    in_bare_aligned = False

    for raw in body.splitlines():
        line = raw.rstrip()

        # Bare \begin{aligned}...\end{aligned} occasionally arrives without
        # $$ delimiters.  Buffer the whole environment before splitting rows.
        if in_bare_aligned:
            aligned_buf.append(line)
            if "\\end{aligned}" in line:
                raw_units.append("\n".join(aligned_buf).strip())
                aligned_buf = []
                in_bare_aligned = False
            continue
        if not in_display and "\\begin{aligned}" in line and "$$" not in line:
            aligned_buf = [line]
            if "\\end{aligned}" in line:
                raw_units.append(line.strip())
                aligned_buf = []
            else:
                in_bare_aligned = True
            continue

        if not line.strip():
            if in_display:
                math_buf.append("")
            continue

        dollar_count = line.count("$$")
        if in_display:
            math_buf.append(line)
            if dollar_count % 2 == 1:
                raw_units.append("\n".join(math_buf).strip())
                math_buf = []
                in_display = False
            continue

        if dollar_count % 2 == 1:
            math_buf = [line]
            in_display = True
            continue

        raw_units.append(line.strip())

    if math_buf:
        raw_units.append("\n".join(math_buf).strip())
    if aligned_buf:
        raw_units.append("\n".join(aligned_buf).strip())

    units: list[str] = []
    for unit in raw_units:
        if not unit.strip():
            continue
        units.extend(_split_aligned(unit))
    return [unit for unit in units if unit.strip()]


def _parse_frames(markdown: str) -> list[_Frame]:
    text = (markdown or "").strip()
    if text.startswith(BLACKBOARD_SUMMARY_MARKER):
        text = text[len(BLACKBOARD_SUMMARY_MARKER):].lstrip()

    matches = list(_FRAME_HEADING_RE.finditer(text))
    frames: list[_Frame] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        units = tuple(_logical_units(body))
        if units:
            frames.append(_Frame(match.group(1).strip(), units))
    return frames


def _comparison_key(unit: str) -> str:
    """Canonical form used ONLY to detect repeated visible content.

    We remove presentation syntax (math delimiters, alignment markers,
    whitespace, bullets, \text wrappers) but keep mathematical symbols,
    numbers, subscripts, superscripts, operators, and command names.
    """
    text = unit.strip()
    text = text.replace("$$", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    text = text.replace("&", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\(?:quad|qquad|,|;|!|:)", "", text)

    # Unwrap simple \text{...} so a prose line and the same prose inside an
    # aligned environment compare as the same board content.
    previous = None
    while previous != text:
        previous = text
        text = _TEXT_RE.sub(r"\1", text)

    text = re.sub(r"^[\s>*•·\-—–]+", "", text)
    text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    text = text.replace("；", ";").replace("（", "(").replace("）", ")")
    text = text.replace(r"\cdots", "...").replace(r"\ldots", "...")
    text = re.sub(r"\s+", "", text)
    # End punctuation varies harmlessly between otherwise identical frames.
    text = re.sub(r"[.,;:]+$", "", text)
    return text


def _is_math_heavy(unit: str) -> bool:
    return any(
        token in unit
        for token in ("$$", "\\sum", "\\int", "\\forall", "\\infty", "=", "<", ">", "^", "_")
    )


def _same_content(a: str, b: str) -> bool:
    """Conservative content-equivalence test independent of vertical position."""
    ka = _comparison_key(a)
    kb = _comparison_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True

    shorter, longer = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(shorter) >= 10 and shorter in longer:
        # Literal containment is strong evidence of the same row while the
        # lecturer is still completing it.  Use a stricter ratio for math.
        ratio = len(shorter) / len(longer)
        threshold = 0.84 if (_is_math_heavy(a) or _is_math_heavy(b)) else 0.70
        if ratio >= threshold:
            return True

    # Fuzzy matching is allowed only for prose.  For mathematics, even a
    # one-symbol edit may be meaningful, so exact/containment rules above are
    # deliberately the only forms of fuzzy equivalence.
    if _is_math_heavy(a) or _is_math_heavy(b):
        return False
    if min(len(ka), len(kb)) < 12:
        return False
    return SequenceMatcher(None, ka, kb, autojunk=False).ratio() >= 0.93


def _prefer_newer(old: str, new: str) -> bool:
    """Whether a later equivalent row is a visibly fuller transcription."""
    ko = _comparison_key(old)
    kn = _comparison_key(new)
    if not ko or not kn or ko == kn:
        return False
    if ko in kn and len(kn) >= len(ko) * 1.06:
        return True
    if not (_is_math_heavy(old) or _is_math_heavy(new)):
        sim = SequenceMatcher(None, ko, kn, autojunk=False).ratio()
        return sim >= 0.93 and len(kn) >= len(ko) * 1.12
    return False


def _matching_count(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Greedy one-to-one match count between two board snapshots."""
    used: set[int] = set()
    count = 0
    for unit_a in a:
        for j, unit_b in enumerate(b):
            if j in used:
                continue
            if _same_content(unit_a, unit_b):
                used.add(j)
                count += 1
                break
    return count


def _overlap_small_side(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    return _matching_count(a, b) / min(len(a), len(b))


def _retention(old: tuple[str, ...], new: tuple[str, ...]) -> float:
    if not old:
        return 1.0
    return _matching_count(old, new) / len(old)


def _is_board_reset(frames: list[_Frame], index: int) -> bool:
    """Detect persistent replacement, not mere up/down board motion.

    Vertical motion commonly turns a full frame into an overlapping subset;
    overlap on the SMALLER side therefore remains high and must not be treated
    as an erase.  A reset requires both low semantic overlap and confirmation
    from the following frame.
    """
    if index <= 0:
        return False

    previous = frames[index - 1].units
    current = frames[index].units
    following = frames[index + 1].units if index + 1 < len(frames) else tuple()
    if not previous or not current:
        return False

    direct_overlap = _overlap_small_side(previous, current)
    old_retained = _retention(previous, current)

    if not following:
        return direct_overlap < 0.05 and old_retained < 0.10

    new_persists = _retention(current, following)
    old_returns = _retention(previous, following)

    # A genuine board replacement has almost no common content and the new
    # content remains visible in the next sample.  A shifted board usually has
    # strong overlap on the smaller side even if many old lines leave view.
    return (
        direct_overlap < 0.08
        and old_retained < 0.14
        and new_persists >= 0.35
        and old_returns < 0.25
    )


def _find_equivalent(state: list[str], unit: str) -> int | None:
    # Search newest-first because a repeated board line is normally close in
    # time, but position is intentionally ignored.
    for index in range(len(state) - 1, -1, -1):
        if _same_content(state[index], unit):
            return index
    return None


def _merge_units(state: list[str], current_units: tuple[str, ...]) -> int:
    """Merge a snapshot into a segment; return number of genuinely new rows."""
    added = 0
    for unit in current_units:
        index = _find_equivalent(state, unit)
        if index is not None:
            if _prefer_newer(state[index], unit):
                state[index] = unit
            continue
        state.append(unit)
        added += 1
    return added


def _seen_elsewhere(history: list[str], unit: str) -> bool:
    """Suppress long rows that reappear after a physical board move/reset.

    Short labels such as 'Proof' or '例' may legitimately recur, so global
    suppression is restricted to substantive rows.
    """
    key = _comparison_key(unit)
    if len(key) < 14:
        return False
    return _find_equivalent(history, unit) is not None


def _render_episode(number: int, start_ts: str, end_ts: str,
                    units: list[str]) -> list[str]:
    if not units:
        return []
    heading = f"#### 板书片段 {number}（{start_ts}–{end_ts}）"
    return [heading, "", "\n\n".join(units), ""]


def compile_blackboard_notes(markdown: str) -> str:
    """Compile frame snapshots into content-stitched, non-repetitive notes."""
    frames = _parse_frames(markdown)
    if not frames:
        return (markdown or "").strip()

    out: list[str] = [BLACKBOARD_NOTES_MARKER, ""]
    history: list[str] = []
    episode_no = 1
    episode_start = frames[0].timestamp
    episode_end = frames[0].timestamp
    episode_state: list[str] = []

    for index, frame in enumerate(frames):
        if index > 0 and _is_board_reset(frames, index):
            out.extend(
                _render_episode(
                    episode_no,
                    episode_start,
                    episode_end,
                    episode_state,
                )
            )
            history.extend(episode_state)
            episode_no += 1
            episode_start = frame.timestamp
            episode_state = []

        # If a board was moved away and later moved back, long rows can cross a
        # reset boundary.  Do not print those rows twice; only merge genuinely
        # new content from the returning board.
        filtered = tuple(
            unit for unit in frame.units
            if not _seen_elsewhere(history, unit)
        )
        _merge_units(episode_state, filtered)
        episode_end = frame.timestamp

    out.extend(
        _render_episode(
            episode_no,
            episode_start,
            episode_end,
            episode_state,
        )
    )

    if len(out) <= 2:
        return (markdown or "").strip()
    return "\n".join(out).strip()
