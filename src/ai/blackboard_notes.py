"""Stitch chronological blackboard snapshots as a sliding long canvas.

The vision model already transcribes selected frames accurately.  The hard part is
that a physical classroom blackboard can move vertically, so the same written
material re-enters the camera at a different screen position and is transcribed
again with small formatting/OCR differences.

This compiler therefore does NOT deduplicate rows independently.  It aligns
whole ordered snapshots.  Repeated rows should form a coherent diagonal between
two snapshots (the mathematical analogue of matching two overlapping windows
of one long scroll).  This lets us tolerate small LaTeX spelling differences
inside an otherwise clearly identical block while keeping genuinely new rows.

No LLM is called here and no mathematical step is inferred.  The raw
frame-by-frame transcription remains the source of truth in the database.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from src.ai.blackboard_vision import BLACKBOARD_SUMMARY_MARKER


BLACKBOARD_NOTES_MARKER = "### 黑板板书 LaTeX 笔记"
_FRAME_HEADING_RE = re.compile(r"^####\s+([^\n]+)\s*$", re.MULTILINE)
_ALIGNED_RE = re.compile(
    r"(?:\$\$|\\\[)?\s*\\begin\{aligned\}(.*?)\\end\{aligned\}\s*(?:\$\$|\\\])?",
    re.DOTALL,
)
_SIMPLE_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}")


@dataclass(frozen=True)
class _Unit:
    text: str
    key: str
    math_heavy: bool


@dataclass(frozen=True)
class _Frame:
    timestamp: str
    units: tuple[_Unit, ...]


@dataclass
class _Variant:
    text: str
    count: int
    last_seq: int


@dataclass
class _Note:
    """One logical row on the stitched long-canvas transcript."""

    episode: int
    first_ts: str
    last_ts: str
    variants: dict[str, _Variant] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        unit: _Unit,
        episode: int,
        timestamp: str,
        seq: int,
    ) -> "_Note":
        note = cls(episode=episode, first_ts=timestamp, last_ts=timestamp)
        note.variants[unit.key] = _Variant(unit.text, 1, seq)
        return note

    def observe(self, unit: _Unit, timestamp: str, seq: int) -> None:
        """Record another sighting of this logical row.

        Literal partial -> full completion is collapsed immediately.  Other
        small fuzzy variants are retained internally and the most repeatedly
        observed version is selected at render time.  This is useful when the
        same board row is recognized as ``\\leq`` in one frame and
        ``\\leqslant`` (or with one OCR-character difference) in another.
        """
        self.last_ts = timestamp

        if unit.key in self.variants:
            variant = self.variants[unit.key]
            variant.count += 1
            variant.last_seq = seq
            if len(unit.text) > len(variant.text):
                variant.text = unit.text
            return

        # If one version is literally a substantial prefix/subsequence of the
        # other, treat the longer one as the completed row and transfer the
        # observations to it.
        for key, variant in list(self.variants.items()):
            shorter, longer = (
                (key, unit.key) if len(key) <= len(unit.key) else (unit.key, key)
            )
            if (
                len(shorter) >= 8
                and shorter in longer
                and len(shorter) / len(longer) >= 0.58
            ):
                if len(unit.key) > len(key):
                    self.variants.pop(key)
                    self.variants[unit.key] = _Variant(
                        unit.text,
                        variant.count + 1,
                        seq,
                    )
                else:
                    variant.count += 1
                    variant.last_seq = seq
                return

        # A context-supported fuzzy match is kept as a separate variant rather
        # than silently rewriting symbols.  Repeated sightings decide which
        # transcription is rendered.
        self.variants[unit.key] = _Variant(unit.text, 1, seq)

    def representative(self) -> str:
        if not self.variants:
            return ""
        _, variant = max(
            self.variants.items(),
            key=lambda kv: (
                kv[1].count,
                len(kv[0]),
                kv[1].last_seq,
            ),
        )
        return variant.text


@dataclass(frozen=True)
class _PairInfo:
    similarity: float
    exact: bool
    containment: bool
    containment_ratio: float
    min_len: int
    math_heavy: bool


@dataclass(frozen=True)
class _Alignment:
    pairs: tuple[tuple[int, int], ...]
    ref_len: int
    cur_len: int

    @property
    def ref_coverage(self) -> float:
        return len(self.pairs) / self.ref_len if self.ref_len else 0.0

    @property
    def cur_coverage(self) -> float:
        return len(self.pairs) / self.cur_len if self.cur_len else 0.0

    @property
    def small_coverage(self) -> float:
        denominator = min(self.ref_len, self.cur_len)
        return len(self.pairs) / denominator if denominator else 0.0


def _split_aligned(unit: str) -> list[str]:
    """Split one aligned environment into independent display-math rows."""
    match = _ALIGNED_RE.fullmatch(unit.strip())
    if not match:
        return [unit.strip()]

    rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", match.group(1))
    out: list[str] = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        # '&' is alignment syntax and is invalid after the row is removed from
        # its aligned environment.
        row = row.replace("&", "").strip()
        if row:
            out.append(f"$$\n{row}\n$$")
    return out or [unit.strip()]


def _logical_units(body: str) -> list[str]:
    """Split one frame conservatively while keeping display math intact."""
    raw_units: list[str] = []
    math_buf: list[str] = []
    aligned_buf: list[str] = []
    in_display = False
    in_bare_aligned = False

    for raw in body.splitlines():
        line = raw.rstrip()

        # Vision occasionally emits a bare \begin{aligned}...\end{aligned}
        # without $$ delimiters.  Buffer it before splitting its rows.
        if in_bare_aligned:
            aligned_buf.append(line)
            if r"\end{aligned}" in line:
                raw_units.append("\n".join(aligned_buf).strip())
                aligned_buf = []
                in_bare_aligned = False
            continue

        if not in_display and r"\begin{aligned}" in line and "$$" not in line:
            aligned_buf = [line]
            if r"\end{aligned}" in line:
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
        if unit.strip():
            units.extend(_split_aligned(unit))
    return [unit for unit in units if unit.strip()]


def _comparison_key(unit: str) -> str:
    """Canonical key used only for visual-content matching.

    Only presentation-level LaTeX differences are normalized.  The rendered
    text is never rewritten from this key.
    """
    text = unit.strip()
    text = text.replace("$$", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    text = text.replace("&", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\displaystyle", "")

    # Common visually equivalent LaTeX spellings produced inconsistently by
    # vision/OCR.  Use command-boundary regexes so \le does not corrupt \leq.
    text = re.sub(r"\\leqslant\b", r"\\leq", text)
    text = re.sub(r"\\le(?![A-Za-z])", r"\\leq", text)
    text = re.sub(r"\\geqslant\b", r"\\geq", text)
    text = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", text)
    text = re.sub(r"\\Longrightarrow\b", r"\\Rightarrow", text)
    text = re.sub(r"\\longrightarrow\b", r"\\to", text)
    text = re.sub(r"\\rightarrow\b", r"\\to", text)
    text = re.sub(r"\\Longleftrightarrow\b", r"\\Leftrightarrow", text)
    text = re.sub(r"\\(?:c|l)?dots\b", r"\\dots", text)
    text = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\b", "", text)
    text = text.replace(r"\lVert", r"\|").replace(r"\rVert", r"\|")
    text = text.replace(r"\Vert", r"\|")
    text = re.sub(r"\\(?:quad|qquad|,|;|!|:)", "", text)

    previous = None
    while previous != text:
        previous = text
        text = _SIMPLE_WRAPPER_RE.sub(r"\1", text)

    text = re.sub(r"^[\s>*•·\-—–]+", "", text)
    text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    text = text.replace("；", ";").replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.,;:]+$", "", text)
    return text


def _is_math_heavy(text: str) -> bool:
    return any(
        token in text
        for token in (
            "$$",
            r"\sum",
            r"\int",
            r"\forall",
            r"\exists",
            r"\infty",
            r"\mathbb",
            r"\mathcal",
            "=",
            "<",
            ">",
            "^",
            "_",
        )
    )


def _make_unit(text: str) -> _Unit:
    text = text.strip()
    return _Unit(
        text=text,
        key=_comparison_key(text),
        math_heavy=_is_math_heavy(text),
    )


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

        units: list[_Unit] = []
        for raw_unit in _logical_units(body):
            unit = _make_unit(raw_unit)
            if unit.key:
                units.append(unit)

        if units:
            frames.append(_Frame(match.group(1).strip(), tuple(units)))
    return frames


@lru_cache(maxsize=200_000)
def _sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _pair_info(a: _Unit, b: _Unit) -> _PairInfo:
    ka, kb = a.key, b.key
    math_heavy = a.math_heavy or b.math_heavy
    if not ka or not kb:
        return _PairInfo(0.0, False, False, 0.0, 0, math_heavy)

    min_len = min(len(ka), len(kb))
    if ka == kb:
        return _PairInfo(1.0, True, True, 1.0, min_len, math_heavy)

    shorter, longer = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    containment = len(shorter) >= 6 and shorter in longer
    containment_ratio = len(shorter) / len(longer) if containment else 0.0
    similarity = _sequence_ratio(ka, kb)

    return _PairInfo(
        similarity,
        False,
        containment,
        containment_ratio,
        min_len,
        math_heavy,
    )


def _anchor_weight(info: _PairInfo) -> float:
    """Weight a row pair as evidence for the board's vertical displacement."""
    if info.min_len < 4:
        return 0.0
    if info.exact:
        return 1.15 if info.min_len >= 10 else 0.75
    if (
        info.containment
        and info.min_len >= 8
        and info.containment_ratio >= 0.62
    ):
        return 0.95 + 0.15 * info.containment_ratio
    if info.min_len >= 12 and info.similarity >= 0.93:
        return info.similarity
    if info.min_len >= 28 and info.similarity >= 0.89:
        return 0.85 * info.similarity
    return 0.0


def _match_gain(info: _PairInfo) -> float:
    """Score a candidate pair inside an already identified diagonal band."""
    if info.exact:
        return 4.0
    if (
        info.containment
        and info.min_len >= 8
        and info.containment_ratio >= 0.58
    ):
        return 2.8 + info.containment_ratio
    if info.min_len < 8:
        return 0.0
    if info.similarity >= 0.95:
        return 3.4
    if info.similarity >= 0.88:
        return 2.2
    if info.similarity >= 0.80:
        return 1.0
    return 0.0


def _estimate_offset(
    reference: tuple[_Unit, ...] | list[_Unit],
    current: tuple[_Unit, ...] | list[_Unit],
) -> int | None:
    """Estimate vertical row displacement by voting on strong pair diagonals."""
    anchors: list[tuple[int, float, int, int]] = []
    for i, old in enumerate(reference):
        for j, new in enumerate(current):
            info = _pair_info(old, new)
            weight = _anchor_weight(info)
            if weight:
                anchors.append((i - j, weight, i, j))

    if not anchors:
        return None

    best_offset: int | None = None
    best_key: tuple[int, float, int] | None = None
    for offset in {item[0] for item in anchors}:
        cluster = [
            item
            for item in anchors
            if abs(item[0] - offset) <= 2
        ]
        current_rows = len({item[3] for item in cluster})
        reference_rows = len({item[2] for item in cluster})
        score = sum(item[1] for item in cluster)

        # Prefer a coherent multi-row overlap.  If two regions are equally
        # convincing in the full history, the later one is usually the board
        # currently visible.
        key = (min(current_rows, reference_rows), score, offset)
        if best_key is None or key > best_key:
            best_key = key
            best_offset = offset

    return best_offset


def _run_ids(
    pairs: list[tuple[int, int, _PairInfo]],
) -> list[int]:
    """Group nearly consecutive aligned pairs into local context runs."""
    out: list[int] = []
    run_id = 0
    previous: tuple[int, int, _PairInfo] | None = None

    for pair in pairs:
        if previous is not None:
            pi, pj, _ = previous
            i, j, _ = pair
            if i - pi > 3 or j - pj > 3:
                run_id += 1
        out.append(run_id)
        previous = pair
    return out


def _is_strong(info: _PairInfo) -> bool:
    if info.exact:
        return True
    if (
        info.containment
        and info.min_len >= 8
        and info.containment_ratio >= 0.68
    ):
        return True
    return info.min_len >= 10 and info.similarity >= 0.95


def _accept_aligned_pairs(
    pairs: list[tuple[int, int, _PairInfo]],
) -> list[tuple[int, int]]:
    """Accept fuzzy math only when neighboring rows corroborate the match."""
    if not pairs:
        return []

    run_ids = _run_ids(pairs)
    run_members: dict[int, list[int]] = defaultdict(list)
    for position, run_id in enumerate(run_ids):
        run_members[run_id].append(position)

    accepted: list[tuple[int, int]] = []
    for position, (i, j, info) in enumerate(pairs):
        if info.min_len < 4:
            continue

        if _is_strong(info):
            accepted.append((i, j))
            continue

        members = run_members[run_ids[position]]
        strong_count = sum(
            _is_strong(pairs[member][2])
            for member in members
        )

        threshold = 0.82 if info.math_heavy else 0.80
        if info.min_len < 10:
            threshold = max(threshold, 0.94)
        if info.similarity < threshold:
            continue

        # This is the central safety rule: a merely similar mathematical row
        # is never merged in isolation.  It must lie inside a coherent ordered
        # overlap whose neighboring rows also match.
        if (
            (len(members) >= 2 and strong_count >= 1)
            or len(members) >= 3
        ):
            accepted.append((i, j))

    return accepted


def _align_sequences(
    reference: tuple[_Unit, ...] | list[_Unit],
    current: tuple[_Unit, ...] | list[_Unit],
    band: int = 6,
) -> _Alignment:
    """Align two ordered snapshots around their dominant vertical offset."""
    ref = list(reference)
    cur = list(current)
    n, m = len(ref), len(cur)
    if not n or not m:
        return _Alignment(tuple(), n, m)

    offset = _estimate_offset(ref, cur)
    if offset is None:
        return _Alignment(tuple(), n, m)

    # Weighted monotone alignment in a narrow band around the voted offset.
    # Skips are free; they represent rows leaving/entering the camera window.
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice = [[0] * (m + 1) for _ in range(n + 1)]
    infos: dict[tuple[int, int], _PairInfo] = {}

    for i in range(1, n + 1):
        old = ref[i - 1]
        for j in range(1, m + 1):
            best = dp[i - 1][j]
            selected = 1  # skip reference row

            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                selected = 2  # skip current row

            if abs(((i - 1) - (j - 1)) - offset) <= band:
                info = _pair_info(old, cur[j - 1])
                infos[(i - 1, j - 1)] = info
                gain = _match_gain(info)
                if gain > 0 and dp[i - 1][j - 1] + gain > best:
                    best = dp[i - 1][j - 1] + gain
                    selected = 3

            dp[i][j] = best
            choice[i][j] = selected

    raw_pairs: list[tuple[int, int, _PairInfo]] = []
    i, j = n, m
    while i > 0 and j > 0:
        selected = choice[i][j]
        if selected == 3:
            info = infos[(i - 1, j - 1)]
            if _match_gain(info) > 0:
                raw_pairs.append((i - 1, j - 1, info))
            i -= 1
            j -= 1
        elif selected == 1:
            i -= 1
        else:
            j -= 1

    raw_pairs.reverse()
    return _Alignment(
        tuple(_accept_aligned_pairs(raw_pairs)),
        n,
        m,
    )


def _should_start_episode(
    frames: list[_Frame],
    index: int,
    previous_alignment: _Alignment,
) -> bool:
    """Detect a persistent camera/board discontinuity for presentation only."""
    if index <= 0:
        return False
    if previous_alignment.small_coverage >= 0.14:
        return False

    if index + 1 >= len(frames):
        return previous_alignment.small_coverage < 0.05

    following_alignment = _align_sequences(
        frames[index].units,
        frames[index + 1].units,
    )
    return following_alignment.small_coverage >= 0.30


def _representative_units(notes: list[_Note]) -> tuple[_Unit, ...]:
    return tuple(
        _make_unit(note.representative())
        for note in notes
        if note.representative()
    )


def compile_blackboard_notes(markdown: str) -> str:
    """Compile full-frame board snapshots into a sliding-canvas transcript."""
    frames = _parse_frames(markdown)
    if not frames:
        return (markdown or "").strip()

    notes: list[_Note] = []
    key_index: dict[str, list[int]] = defaultdict(list)
    previous_frame: _Frame | None = None
    previous_note_ids: list[int | None] = []
    episode = 1
    episode_bounds: dict[int, list[str]] = {}
    seq = 0

    for frame_index, frame in enumerate(frames):
        seq += 1
        current = frame.units

        if previous_frame is None:
            previous_alignment = _Alignment(tuple(), 0, len(current))
        else:
            previous_alignment = _align_sequences(
                previous_frame.units,
                current,
            )

        history_units = _representative_units(notes)
        need_history_alignment = bool(notes) and (
            previous_frame is None
            or previous_alignment.cur_coverage < 0.70
            or previous_alignment.small_coverage < 0.50
        )
        history_alignment = (
            _align_sequences(history_units, current)
            if need_history_alignment
            else _Alignment(tuple(), len(history_units), len(current))
        )

        if _should_start_episode(
            frames,
            frame_index,
            previous_alignment,
        ):
            episode += 1

        if episode not in episode_bounds:
            episode_bounds[episode] = [frame.timestamp, frame.timestamp]
        else:
            episode_bounds[episode][1] = frame.timestamp

        current_note_ids: list[int | None] = [None] * len(current)
        used_note_ids: set[int] = set()

        # First preference: correspondence to the immediately previous frame.
        # This is the strongest evidence that a row merely moved on screen.
        for old_i, current_j in previous_alignment.pairs:
            if old_i >= len(previous_note_ids):
                continue
            note_id = previous_note_ids[old_i]
            if note_id is None or note_id in used_note_ids:
                continue
            current_note_ids[current_j] = note_id
            used_note_ids.add(note_id)

        # If overlap with the adjacent frame is incomplete, align the whole
        # current snapshot against the stitched history.  A returning physical
        # board then matches one coherent diagonal in the old long canvas,
        # instead of being appended as duplicate independent rows.
        for history_i, current_j in history_alignment.pairs:
            if (
                current_note_ids[current_j] is not None
                or history_i in used_note_ids
            ):
                continue
            current_note_ids[current_j] = history_i
            used_note_ids.add(history_i)

        # Exact/style-normalized matches are safe even outside the dominant
        # diagonal.  Restrict this fallback to substantive rows so recurring
        # labels such as "Pf." are not globally collapsed.
        for current_j, unit in enumerate(current):
            if current_note_ids[current_j] is not None or len(unit.key) < 8:
                continue
            for note_id in reversed(key_index.get(unit.key, [])):
                if note_id not in used_note_ids:
                    current_note_ids[current_j] = note_id
                    used_note_ids.add(note_id)
                    break

        # Only rows with no correspondence anywhere above are genuinely new
        # board content.  They are appended in first-appearance order.
        for current_j, unit in enumerate(current):
            note_id = current_note_ids[current_j]
            if note_id is None:
                note_id = len(notes)
                notes.append(
                    _Note.create(
                        unit,
                        episode,
                        frame.timestamp,
                        seq,
                    )
                )
                current_note_ids[current_j] = note_id
            else:
                notes[note_id].observe(unit, frame.timestamp, seq)

            if note_id not in key_index[unit.key]:
                key_index[unit.key].append(note_id)

        previous_frame = frame
        previous_note_ids = current_note_ids

    if not notes:
        return (markdown or "").strip()

    notes_by_episode: dict[int, list[_Note]] = defaultdict(list)
    for note in notes:
        notes_by_episode[note.episode].append(note)

    out: list[str] = [BLACKBOARD_NOTES_MARKER, ""]
    section_number = 1
    for episode_id in sorted(notes_by_episode):
        group = notes_by_episode[episode_id]
        if not group:
            continue

        start_ts, end_ts = episode_bounds.get(
            episode_id,
            [group[0].first_ts, group[-1].last_ts],
        )
        out.append(
            f"#### 板书片段 {section_number}（{start_ts}–{end_ts}）"
        )
        out.append("")
        out.append(
            "\n\n".join(
                note.representative()
                for note in group
                if note.representative()
            )
        )
        out.append("")
        section_number += 1

    return "\n".join(out).strip()
