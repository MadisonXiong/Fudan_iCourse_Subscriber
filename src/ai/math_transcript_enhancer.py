"""Readable math transcript built on top of the faithful audit transcript.

This optional second layer may make high-confidence contextual ASR corrections
and restore formulas from same-window PPT/blackboard evidence.  The faithful
AI-proofread transcript is never modified and remains the audit trail.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from openai import OpenAI

from src.ai.ppt_dedup import clean_ppt_text
from src.runtime import config

_TIMEOUT = int(os.environ.get("MATH_TRANSCRIPT_TIMEOUT", "300"))
_BATCH = max(1, int(os.environ.get("MATH_TRANSCRIPT_BATCH_SIZE", "5")))
_MIN_RATIO, _MAX_RATIO = 0.90, 2.20
_MIN_SOURCE_COVERAGE = 0.88
_MIN_CLAUSE_COVERAGE = 0.65
_BOARD_MAX = int(os.environ.get("MATH_TRANSCRIPT_BOARD_CHARS", "4000"))
_BOARD_CONTEXT_MAX = int(os.environ.get("MATH_TRANSCRIPT_BOARD_CONTEXT_CHARS", "900"))
_PPT_MAX = int(os.environ.get("MATH_TRANSCRIPT_PPT_CHARS", "1400"))
_CONTEXT_MAX = int(os.environ.get("MATH_TRANSCRIPT_CONTEXT_CHARS", "700"))
_BOARD_RE = re.compile(r"^####\s+(\d{1,2}):(\d{2})(?::(\d{2}))?.*$", re.M)
_OUT_RE = re.compile(
    r"<<<CHUNK\s+(\d+)>>>\s*(.*?)\s*<<<END\s+CHUNK\s+\1>>>", re.I | re.S
)
_VIS_OPEN = '<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕'
_VIS_CLOSE = "</span>"

# Course-scoped replacements whose meaning is unambiguous even if the model is
# unavailable.  They are applied only to the readable enhanced copy, never to
# the faithful proofread attachment.  Keep this list deliberately small.
_HIGH_CONFIDENCE_ASR = {
    "泛函分析": (
        ("办案分析", "泛函分析"),
        ("十遍函数", "实变函数"),
        ("十变函数", "实变函数"),
    ),
}

_SYSTEM = rf"""
你是数学课堂“增强语音转写器”。忠实 AI 校订稿是不可改写的审计底稿；你的输出是另一个更适合阅读的增强版本。

对每个 CHUNK 严格按两个阶段处理：

第一阶段——上下文 ASR 纠错（正常黑色文本）：
1. 可依据课程名、当前 CHUNK 的完整句意及相邻语音上下文，修正高置信度的同音/近音误识别、专业术语、英文词和断句。例如泛函分析课程中的“办案分析”应修为“泛函分析”，“十遍函数”应修为“实变函数”。
2. ASR 纠错是在还原老师实际说出的语音，不得添加“视觉补全”标记，也不得改变老师的意思、论述顺序或信息量。
3. 只修能确定的局部。若“十遍函数的文物课程”中只能确定“十遍函数”是“实变函数”，则只修这一处；无法确定的其余原话保留，必要时紧随可疑片段标 `[语音存疑]`，绝不能自行补写。
4. 相邻上下文只用于判断术语和句意，严禁把相邻 CHUNK 的内容复制进当前 CHUNK，尤其严禁据此补公式。

第二阶段——视觉数学恢复（紫色“视觉补全”）：
5. 每个 CHUNK 的公式证据完全独立。只有标为“本 CHUNK”的 PPT/黑板直接给出对应对象时，才可恢复原语音没有说清楚的变量、集合、映射、量词、等式、不等式或公式。绝不能跨 CHUNK 借公式，也不能凭学科常识补公式。
6. 原语音已经明确说出的数学表达可直接规范成 `$...$`，保持正常黑色；原语音没有、仅由本 CHUNK 视觉证据恢复的内容，才必须完整放在：
{_VIS_OPEN}$...${_VIS_CLOSE}
若是极短数学短语也同样标记。只补语音缺口，不抄整段板书/PPT；证据不足时保留原文并可标 `[公式存疑]`。

共同约束：
7. 必须逐句保留原转写的顺序、口语语义和全部已有信息，包括课程介绍、通知、作业、评分、答疑等非数学内容；不得删除句子，不得摘要，不得润色成教材，不得用 PPT/板书文字替换老师的讲述，也不得扩写证明或补背景知识。
8. 数学用 `$...$` 或 `$$...$$`；不要使用 aligned/array/cases/gathered/split。
9. 输出严格为下列格式，一个输入 CHUNK 对应一个输出 CHUNK，编号一致且不得遗漏：
<<<CHUNK 1>>>
增强后的正文
<<<END CHUNK 1>>>
不要在 CHUNK 块之外输出任何文字。
""".strip()

_RETRY = rf"""
上一轮存在缺块、删句或改动过大。重新处理整个批次：逐句保留原文，包括所有课程介绍、通知、作业和评分信息；只做高置信度的局部 ASR 纠错和最小必要公式恢复；不得摘要或用板书替换语音。ASR 纠错保持黑色，只有本 CHUNK 视觉证据新增的内容才放在 {_VIS_OPEN}...{_VIS_CLOSE} 中；禁止使用相邻上下文补公式，禁止搬运完整定义/定理/证明；剔除紫色视觉补全后，黑色正文的信息量和长度必须与原转写基本一致；必须输出全部编号。
""".strip()


@dataclass(frozen=True)
class MathTranscriptResult:
    markdown: str
    segments: list[dict]
    model_label: str


def _fmt(ms: int) -> str:
    sec = max(0, int(ms) // 1000)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _board_time(m: re.Match) -> int:
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return a * 60 + b if m.group(3) is None else a * 3600 + b * 60 + c


def _trim(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    left = int(limit * .62)
    right = limit - left
    return text[:left] + f"\n[{label}中段截断]\n" + text[-right:]


def _correct_high_confidence_asr(text: str, course_title: str) -> str:
    """Apply a tiny course-scoped allowlist to the enhanced copy only."""
    corrected = str(text or "")
    title = str(course_title or "")
    for keyword, replacements in _HIGH_CONFIDENCE_ASR.items():
        if keyword not in title:
            continue
        for wrong, right in replacements:
            corrected = corrected.replace(wrong, right)
    return corrected


def _adjacent_context(source: list[dict], index: int) -> tuple[str, str]:
    """Return small, read-only tails/heads from the neighbouring chunks."""
    before = str(source[index - 1].get("text") or "") if index > 0 else ""
    after = (
        str(source[index + 1].get("text") or "")
        if index + 1 < len(source)
        else ""
    )
    return before[-_CONTEXT_MAX:], after[:_CONTEXT_MAX]


def _board(raw: str, start: int, end: int) -> str:
    if not raw:
        return ""
    matches = list(_BOARD_RE.finditer(raw))
    out = []
    for i, m in enumerate(matches):
        sec = _board_time(m)
        if sec < start or sec > end:
            continue
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        block = raw[m.start():stop].strip()
        if block:
            out.append(block)
    return _trim("\n\n".join(out), _BOARD_MAX, "黑板证据")


def _previous_board_context(raw: str, start: int) -> str:
    """Return the nearest earlier board block for terminology context only."""
    if not raw:
        return ""
    matches = list(_BOARD_RE.finditer(raw))
    previous = [
        (i, match)
        for i, match in enumerate(matches)
        if _board_time(match) < start
    ]
    if not previous:
        return ""
    i, match = previous[-1]
    stop = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
    return _trim(raw[match.start():stop].strip(), _BOARD_CONTEXT_MAX, "前序黑板上下文")


def _ppt(pages: list[dict], start: int, end: int) -> str:
    mid = (start + end) // 2
    found = []
    for page in pages or []:
        sec = int(page.get("created_sec", 0) or 0)
        if sec < start or sec > end:
            continue
        text = clean_ppt_text(page.get("text") or "").strip()
        if text:
            found.append((abs(sec - mid), f"[PPT @ {sec//60:02d}:{sec%60:02d}]\n{text}"))
    found.sort(key=lambda x: x[0])
    return _trim("\n\n".join(text for _, text in found), _PPT_MAX, "PPT证据")


def _safe(source: str, candidate: str) -> bool:
    if not candidate:
        return False
    ratio = len(candidate) / max(1, len(source))
    return _MIN_RATIO <= ratio <= _MAX_RATIO


def _spoken_text(candidate: str) -> str:
    """Remove visual-only additions before checking preservation of speech."""
    value = re.sub(
        re.escape(_VIS_OPEN) + r".*?" + re.escape(_VIS_CLOSE),
        "",
        str(candidate or ""),
        flags=re.S,
    )
    value = re.sub(r"<[^>]+>", "", value)
    return value


def _comparison_text(text: str) -> str:
    """Normalize layout punctuation while retaining spoken content characters."""
    value = re.sub(r"[`$*_#]", "", str(text or ""))
    value = re.sub(r"[\s，。！？；：、,.!?;:（）()【】\[\]《》<>“”‘’]+", "", value)
    return value


def _source_coverage(source: str, candidate: str) -> float:
    """Return how much source speech survives in the non-visual candidate."""
    expected = _comparison_text(source)
    actual = _comparison_text(_spoken_text(candidate))
    if not expected or not actual:
        return 0.0
    matched = sum(
        block.size
        for block in SequenceMatcher(None, expected, actual, autojunk=False).get_matching_blocks()
    )
    return matched / len(expected)


def _clauses_preserved(source: str, candidate: str) -> bool:
    """Prevent a locally high overall score from hiding one deleted sentence."""
    actual = _comparison_text(_spoken_text(candidate))
    clauses = [
        _comparison_text(part)
        for part in re.split(r"[。！？；\n]+", str(source or ""))
    ]
    for clause in clauses:
        if len(clause) < 8:
            continue
        matched = sum(
            block.size
            for block in SequenceMatcher(None, clause, actual, autojunk=False).get_matching_blocks()
        )
        if matched / len(clause) < _MIN_CLAUSE_COVERAGE:
            return False
    return True


def _valid_candidate(
    source: str,
    candidate: str,
    *,
    course_title: str,
    has_visual_evidence: bool,
) -> bool:
    """Reject unsafe rewrites and impossible/malformed visual additions."""
    if not _safe(source, candidate):
        return False
    if _source_coverage(source, candidate) < _MIN_SOURCE_COVERAGE:
        return False
    if not _clauses_preserved(source, candidate):
        return False
    open_count = candidate.count(_VIS_OPEN)
    close_count = candidate.count(_VIS_CLOSE)
    label_count = candidate.count("〔视觉补全〕")
    if open_count != close_count or label_count != open_count:
        return False
    if open_count and not has_visual_evidence:
        return False

    # Deterministic high-confidence corrections are spoken-word corrections,
    # so the model must preserve them and must never relabel them as visual.
    title = str(course_title or "")
    targets: list[str] = []
    for keyword, replacements in _HIGH_CONFIDENCE_ASR.items():
        if keyword in title:
            targets.extend(right for _, right in replacements if right in source)
    spans = re.findall(re.escape(_VIS_OPEN) + r"(.*?)" + re.escape(_VIS_CLOSE), candidate, re.S)
    for target in targets:
        if candidate.count(target) < source.count(target):
            return False
        if any(target in span for span in spans):
            return False
    return True


def _parse(text: str, n: int) -> dict[int, str]:
    out = {}
    for m in _OUT_RE.finditer(text or ""):
        idx = int(m.group(1))
        body = m.group(2).strip()
        if 1 <= idx <= n and body and idx not in out:
            out[idx] = body
    return out


class MathTranscriptEnhancer:
    def __init__(self):
        override = [
            x.strip() for x in os.environ.get("MATH_TRANSCRIPT_MODELS", "").split(",")
            if x.strip()
        ]
        self.providers = []
        for p in config.resolve_model_providers():
            models = list(p["models"])
            if p["name"] == "modelscope":
                models = override or ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
            self.providers.append(
                (p["name"], OpenAI(api_key=p["api_key"], base_url=p["base_url"]), tuple(models))
            )
        if not self.providers:
            raise ValueError("No model provider available for math transcript enhancement")

    def _call(self, prompt: str) -> tuple[str, str]:
        errors = []
        for provider, client, models in self.providers:
            for model in models:
                model_id = f"{provider}/{model}"
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=.03,
                        max_tokens=9000,
                        timeout=_TIMEOUT,
                    )
                    choice = resp.choices[0]
                    text = (choice.message.content or "").strip()
                    if not text:
                        raise RuntimeError("empty response")
                    if str(getattr(choice, "finish_reason", "") or "").lower() == "length":
                        raise RuntimeError("truncated response")
                    print(
                        f"[MathTranscript] {model_id}: {len(prompt)} -> {len(text)} chars "
                        f"in {time.time()-t0:.0f}s", flush=True
                    )
                    return text, model_id
                except Exception as exc:
                    msg = f"{model_id}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"[MathTranscript] {msg}", flush=True)
        raise RuntimeError("All math transcript models failed: " + " | ".join(errors))

    @staticmethod
    def _normalise(seg: dict) -> dict | None:
        start = int(seg.get("start_ms", 0) or 0)
        end = int(seg.get("end_ms", start) or start)
        if end < start:
            start, end = end, start
        text = str(seg.get("text") or "").strip()
        return {"start_ms": start, "end_ms": end, "text": text} if text else None

    def _prompt(
        self,
        batch: list[dict],
        pages: list[dict],
        raw_board: str,
        *,
        course_title: str,
        source: list[dict],
        batch_start: int,
    ) -> str:
        blocks = []
        for i, seg in enumerate(batch, 1):
            before, after = _adjacent_context(source, batch_start + i - 1)
            s, e = seg["start_ms"] // 1000, seg["end_ms"] // 1000
            blocks.append(
                f"===== INPUT CHUNK {i} =====\n"
                f"【课程名称】{course_title or '（未知）'}\n"
                f"【视频时段】{_fmt(seg['start_ms'])}–{_fmt(seg['end_ms'])}\n"
                f"【相邻上文语音（只用于术语判断，禁止复制或补公式）】\n"
                f"{before or '（无）'}\n\n"
                f"【忠实 AI 校订语音转写】\n{seg['text']}\n\n"
                f"【相邻下文语音（只用于术语判断，禁止复制或补公式）】\n"
                f"{after or '（无）'}\n\n"
                f"【前一时段黑板上下文（只用于 ASR 术语判断，禁止据此补公式）】\n"
                f"{_previous_board_context(raw_board, s) or '（无）'}\n\n"
                f"【本 CHUNK 的 PPT OCR 证据】\n{_ppt(pages, s, e) or '（无）'}\n\n"
                f"【本 CHUNK 的黑板视觉证据】\n{_board(raw_board, s, e) or '（无）'}"
            )
        return "\n\n".join(blocks)

    def _batch(
        self,
        batch: list[dict],
        pages: list[dict],
        raw_board: str,
        *,
        course_title: str,
        source: list[dict],
        batch_start: int,
    ):
        prompt = self._prompt(
            batch,
            pages,
            raw_board,
            course_title=course_title,
            source=source,
            batch_start=batch_start,
        )
        models, bad = [], set()
        try:
            text, model = self._call(prompt)
            models.append(model)
            parsed = _parse(text, len(batch))
        except Exception as exc:
            print(
                f"[MathTranscript] batch unavailable; preserving safe ASR-corrected base: "
                f"{type(exc).__name__}: {exc}", flush=True
            )
            return [
                {**seg, "math_enhance_status": "safe_base_api_fallback"} for seg in batch
            ], models, len(batch)

        for i, seg in enumerate(batch, 1):
            s, e = seg["start_ms"] // 1000, seg["end_ms"] // 1000
            has_visual = bool(_ppt(pages, s, e) or _board(raw_board, s, e))
            if not _valid_candidate(
                seg["text"],
                parsed.get(i, ""),
                course_title=course_title,
                has_visual_evidence=has_visual,
            ):
                bad.add(i)
        if bad:
            try:
                retry, model = self._call(prompt + "\n\n" + _RETRY)
                models.append(model)
                retry_parsed = _parse(retry, len(batch))
                for i in list(bad):
                    seg = batch[i - 1]
                    s, e = seg["start_ms"] // 1000, seg["end_ms"] // 1000
                    has_visual = bool(_ppt(pages, s, e) or _board(raw_board, s, e))
                    if _valid_candidate(
                        seg["text"],
                        retry_parsed.get(i, ""),
                        course_title=course_title,
                        has_visual_evidence=has_visual,
                    ):
                        parsed[i] = retry_parsed[i]
                        bad.remove(i)
            except Exception as exc:
                print(
                    f"[MathTranscript] batch retry unavailable: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        out = []
        for i, seg in enumerate(batch, 1):
            if i in bad or not parsed.get(i):
                out.append({**seg, "math_enhance_status": "safe_base_safety_fallback"})
            else:
                out.append(
                    {**seg, "text": parsed[i].strip(), "math_enhance_status": "visual_evidence_enhanced"}
                )
        return out, models, len(bad)

    def enhance(
        self,
        proofread_segments,
        ppt_pages,
        *,
        course_title: str = "",
        raw_blackboard: str = "",
    ):
        source = [
            x for x in (self._normalise(s) for s in proofread_segments or []) if x is not None
        ]
        if not source:
            raise ValueError("AI-proofread timed segments are required")
        source = [
            {**seg, "text": _correct_high_confidence_asr(seg["text"], course_title)}
            for seg in source
        ]
        pages = ppt_pages or []
        enhanced, models, fallbacks = [], [], 0
        batches = (len(source) + _BATCH - 1) // _BATCH
        for start in range(0, len(source), _BATCH):
            batch = source[start:start + _BATCH]
            print(
                f"[MathTranscript] batch {start//_BATCH+1}/{batches}: {len(batch)} chunk(s)",
                flush=True,
            )
            out, used, failed = self._batch(
                batch,
                pages,
                raw_blackboard,
                course_title=course_title,
                source=source,
                batch_start=start,
            )
            enhanced.extend(out)
            models.extend(used)
            fallbacks += failed

        md = [
            "# 数学增强语音转写", "",
            "> 该版本完整保留老师的语音讲述，并以忠实 AI 校订转写为底稿：高置信度的上下文 ASR 纠错以正常黑色显示；只有原语音未说清且同时段 PPT/黑板有直接证据的数学内容才以紫色“视觉补全”显示。忠实校订稿另作为独立附件保留，便于回查。", "",
        ]
        for seg in enhanced:
            md += [
                f"## {_fmt(seg['start_ms'])}–{_fmt(seg['end_ms'])}", "",
                str(seg.get("text") or "").strip(), "",
            ]
        unique = []
        for model in models:
            if model and model not in unique:
                unique.append(model)
        label = "math-transcript-v4/" + ("+".join(unique) if unique else "no-llm")
        if fallbacks:
            label += f"|safe-base-fallback[{fallbacks}]"
        return MathTranscriptResult("\n".join(md).strip() + "\n", enhanced, label)
