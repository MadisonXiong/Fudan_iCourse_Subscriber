"""Evidence-constrained restoration of ASR-lost mathematical notation.

The faithful AI-proofread transcript remains the audit trail. This optional
second layer may restore formulas only from the same timed PPT/blackboard
window. Requests are batched to keep model usage bounded on long lectures.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from src.ai.ppt_dedup import clean_ppt_text
from src.runtime import config

_TIMEOUT = int(os.environ.get("MATH_TRANSCRIPT_TIMEOUT", "300"))
_BATCH = max(1, int(os.environ.get("MATH_TRANSCRIPT_BATCH_SIZE", "5")))
_MIN_RATIO, _MAX_RATIO = 0.65, 2.20
_BOARD_MAX = int(os.environ.get("MATH_TRANSCRIPT_BOARD_CHARS", "4000"))
_PPT_MAX = int(os.environ.get("MATH_TRANSCRIPT_PPT_CHARS", "1400"))
_BOARD_RE = re.compile(r"^####\s+(\d{1,2}):(\d{2})(?::(\d{2}))?.*$", re.M)
_OUT_RE = re.compile(
    r"<<<CHUNK\s+(\d+)>>>\s*(.*?)\s*<<<END\s+CHUNK\s+\1>>>", re.I | re.S
)
_VIS_OPEN = '<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕'
_VIS_CLOSE = "</span>"

_SYSTEM = rf"""
你是数学课堂“视觉证据增强转写器”。输入包含多个独立 CHUNK；每个 CHUNK 都有自己的忠实 AI 校订语音、PPT OCR 和黑板视觉证据。

只修复 ASR 在数学符号处留下的空洞。严格规则：
1. CHUNK 必须独立处理，绝不能跨 CHUNK 借公式。
2. 保留原转写的顺序、口语语义和全部已有信息；不得总结、润色成教材、扩写证明或补背景知识。
3. 只有该 CHUNK 的 PPT/黑板直接给出对应对象时，才可恢复变量、集合、映射、量词、等式/不等式或公式。不能凭“数学上应该如此”补全；不确定时保留原文并可写 `[公式存疑]`。
4. 原语音已经明确说出的数学表达可直接规范成 `$...$`。
5. 原转写没有、仅由视觉证据恢复的内容必须完整放在：
{_VIS_OPEN}$...${_VIS_CLOSE}
若是极短数学短语也同样标记。视觉补全只补缺口，不抄整段板书/PPT。
6. 数学用 `$...$` 或 `$$...$$`；不要使用 aligned/array/cases/gathered/split。
7. 输出严格为下列格式，一个输入 CHUNK 对应一个输出 CHUNK，编号一致且不得遗漏：
<<<CHUNK 1>>>
增强后的正文
<<<END CHUNK 1>>>
不要在 CHUNK 块之外输出任何文字。
""".strip()

_RETRY = rf"""
上一轮存在缺块或改动过大。重新处理整个批次：逐句保留原文，只插入最小必要公式；禁止搬运完整定义/定理/证明；所有视觉新增内容必须放在 {_VIS_OPEN}...{_VIS_CLOSE} 中；每个 CHUNK 长度应接近其原转写；必须输出全部编号。
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


def _board(raw: str, start: int, end: int) -> str:
    if not raw:
        return ""
    matches = list(_BOARD_RE.finditer(raw))
    out = []
    for i, m in enumerate(matches):
        sec = _board_time(m)
        if sec < start - 20 or sec > end + 20:
            continue
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        block = raw[m.start():stop].strip()
        if block:
            out.append(block)
    return _trim("\n\n".join(out), _BOARD_MAX, "黑板证据")


def _ppt(pages: list[dict], start: int, end: int) -> str:
    mid = (start + end) // 2
    found = []
    for page in pages or []:
        sec = int(page.get("created_sec", 0) or 0)
        if sec < start - 45 or sec > end + 45:
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

    def _prompt(self, batch: list[dict], pages: list[dict], raw_board: str) -> str:
        blocks = []
        for i, seg in enumerate(batch, 1):
            s, e = seg["start_ms"] // 1000, seg["end_ms"] // 1000
            blocks.append(
                f"===== INPUT CHUNK {i} =====\n"
                f"【视频时段】{_fmt(seg['start_ms'])}–{_fmt(seg['end_ms'])}\n"
                f"【忠实 AI 校订语音转写】\n{seg['text']}\n\n"
                f"【本 CHUNK 的 PPT OCR 证据】\n{_ppt(pages, s, e) or '（无）'}\n\n"
                f"【本 CHUNK 的黑板视觉证据】\n{_board(raw_board, s, e) or '（无）'}"
            )
        return "\n\n".join(blocks)

    def _batch(self, batch: list[dict], pages: list[dict], raw_board: str):
        prompt = self._prompt(batch, pages, raw_board)
        models, bad = [], set()
        try:
            text, model = self._call(prompt)
            models.append(model)
            parsed = _parse(text, len(batch))
        except Exception as exc:
            print(
                f"[MathTranscript] batch unavailable; preserving faithful text: "
                f"{type(exc).__name__}: {exc}", flush=True
            )
            return [
                {**seg, "math_enhance_status": "faithful_api_fallback"} for seg in batch
            ], models, len(batch)

        for i, seg in enumerate(batch, 1):
            if not _safe(seg["text"], parsed.get(i, "")):
                bad.add(i)
        if bad:
            try:
                retry, model = self._call(prompt + "\n\n" + _RETRY)
                models.append(model)
                retry_parsed = _parse(retry, len(batch))
                for i in list(bad):
                    if _safe(batch[i - 1]["text"], retry_parsed.get(i, "")):
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
                out.append({**seg, "math_enhance_status": "faithful_safety_fallback"})
            else:
                out.append(
                    {**seg, "text": parsed[i].strip(), "math_enhance_status": "visual_evidence_enhanced"}
                )
        return out, models, len(bad)

    def enhance(self, proofread_segments, ppt_pages, *, raw_blackboard: str = ""):
        source = [
            x for x in (self._normalise(s) for s in proofread_segments or []) if x is not None
        ]
        if not source:
            raise ValueError("AI-proofread timed segments are required")
        pages = ppt_pages or []
        enhanced, models, fallbacks = [], [], 0
        batches = (len(source) + _BATCH - 1) // _BATCH
        for start in range(0, len(source), _BATCH):
            batch = source[start:start + _BATCH]
            if not any(
                _ppt(pages, s["start_ms"]//1000, s["end_ms"]//1000)
                or _board(raw_blackboard, s["start_ms"]//1000, s["end_ms"]//1000)
                for s in batch
            ):
                enhanced.extend({**s, "math_enhance_status": "no_visual_evidence"} for s in batch)
                continue
            print(
                f"[MathTranscript] batch {start//_BATCH+1}/{batches}: {len(batch)} chunk(s)",
                flush=True,
            )
            out, used, failed = self._batch(batch, pages, raw_blackboard)
            enhanced.extend(out)
            models.extend(used)
            fallbacks += failed

        md = [
            "# 数学增强语音转写", "",
            "> 该版本以忠实 AI 校订转写为底稿，仅在同时段 PPT/黑板有直接证据时恢复 ASR 丢失的数学符号。紫色“视觉补全”不是逐字语音，而是可追溯的视觉证据恢复；忠实逐字稿仍作为独立附件保留。", "",
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
        label = "math-transcript-v2/" + ("+".join(unique) if unique else "no-llm")
        if fallbacks:
            label += f"|faithful-fallback[{fallbacks}]"
        return MathTranscriptResult("\n".join(md).strip() + "\n", enhanced, label)
