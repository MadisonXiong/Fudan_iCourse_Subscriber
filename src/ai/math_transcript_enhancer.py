"""Restore mathematically meaningful notation lost by speech recognition.

This is deliberately separate from the faithful ASR proofreader.  The faithful
transcript remains the audit trail.  This enhancer may restore formulas only when
the same video window has direct PPT/blackboard evidence, and every visually
restored insertion is explicitly marked so it cannot be mistaken for literal
speech.
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
_MIN_RATIO = 0.65
_MAX_RATIO = 2.20
_BOARD_HEADING_RE = re.compile(
    r"^####\s+(\d{1,2}):(\d{2})(?::(\d{2}))?.*$", re.MULTILINE
)
_VIS_OPEN = (
    '<span data-visual-restored="true" style="color:#7c3aed;">'
    '〔视觉补全〕'
)
_VIS_CLOSE = "</span>"

_SYSTEM_PROMPT = rf"""
你是“数学课堂视觉证据增强转写器”。输入包含：
1. 某个真实视频时段的【忠实 AI 校订语音转写】；
2. 同时段附近的 PPT OCR；
3. 同时段附近的黑板视觉转写。

目标不是重新写讲义，而是解决语音识别在数学符号处出现的空洞，例如：
“对任意，有。”、“当且仅当。”、“设为的 Fourier 展开。”。

严格规则：
1. 保留忠实转写的论述顺序、口语语义和所有已有内容。不得总结、润色成教材、扩写证明或补充背景知识。
2. 只有当 PPT/黑板在同一时段直接给出对应数学对象时，才允许恢复 ASR 丢失的变量、映射、集合、等式、不等式、量词或公式。
3. 不得因为“数学上应该如此”而补公式；视觉证据不清楚时宁可保留原来的空洞，并加 `[公式存疑]`。
4. 如果只是把原转写中已经明确说出的数学表达改写为 LaTeX，不算新增，可以直接写 `$...$`。
5. 任何“原转写里没有、仅根据视觉证据恢复”的内容，都必须完整放在下面的紫色标记中：
{_VIS_OPEN}$...${_VIS_CLOSE}
若补全的是一段短语，也仍放在该标记中。不要修改 data-visual-restored 属性和颜色。
6. 视觉补全只补缺口，不要把整块板书/PPT抄进逐字稿。通常每个缺口只补一个公式或极短数学短语。
7. 数学统一使用可渲染 LaTeX：行内 `$...$`，必要时块级 `$$...$$`；不要使用 aligned/array/cases/gathered/split。
8. 不输出标题、时间戳、修改说明或列表化的“修改记录”；只输出这个视频时段的增强转写正文。
""".strip()

_RETRY_PROMPT = rf"""
上一轮输出改动过大。请重新处理，并执行更严格的限制：
- 逐句保留原转写，只在明显缺少数学对象的位置插入最小公式；
- 视觉证据中的定义、定理、证明步骤不能整体搬进来；
- 所有由视觉证据新增的字符必须位于 {_VIS_OPEN}...{_VIS_CLOSE} 中；
- 输出长度应接近原转写，除必要公式外不要增加任何解释。
""".strip()


@dataclass(frozen=True)
class MathTranscriptResult:
    markdown: str
    segments: list[dict]
    model_label: str


def _fmt_ms(ms: int) -> str:
    sec = max(0, int(ms) // 1000)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_board_time(match: re.Match) -> int:
    a = int(match.group(1))
    b = int(match.group(2))
    c = int(match.group(3) or 0)
    if match.group(3) is None:
        return a * 60 + b
    return a * 3600 + b * 60 + c


def _board_in_window(raw_board: str, start_sec: int, end_sec: int) -> str:
    if not raw_board:
        return ""
    matches = list(_BOARD_HEADING_RE.finditer(raw_board))
    if not matches:
        return ""
    out: list[str] = []
    for i, match in enumerate(matches):
        sec = _parse_board_time(match)
        if sec < start_sec - 75 or sec > end_sec + 75:
            continue
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_board)
        block = raw_board[match.start():block_end].strip()
        if block:
            out.append(block)
    joined = "\n\n".join(out)
    if len(joined) > 7500:
        joined = joined[:7500] + "\n[黑板证据截断]"
    return joined


def _ppt_in_window(pages: list[dict], start_sec: int, end_sec: int) -> str:
    out: list[str] = []
    for page in pages or []:
        sec = int(page.get("created_sec", 0) or 0)
        if sec < start_sec - 90 or sec > end_sec + 90:
            continue
        text = clean_ppt_text(page.get("text") or "").strip()
        if text:
            out.append(f"[PPT @ {sec // 60:02d}:{sec % 60:02d}]\n{text}")
    joined = "\n\n".join(out)
    if len(joined) > 6000:
        joined = joined[:6000] + "\n[PPT 证据截断]"
    return joined


def _safe_ratio(source: str, candidate: str) -> tuple[bool, float]:
    ratio = len(candidate) / max(1, len(source))
    return _MIN_RATIO <= ratio <= _MAX_RATIO, ratio


class MathTranscriptEnhancer:
    def __init__(self):
        resolved = config.resolve_model_providers()
        self.providers: list[tuple[str, OpenAI, tuple[str, ...]]] = []
        override = os.environ.get("MATH_TRANSCRIPT_MODELS", "").strip()
        preferred = [m.strip() for m in override.split(",") if m.strip()]
        for provider in resolved:
            models = list(provider["models"])
            if provider["name"] == "modelscope" and preferred:
                models = preferred
            elif provider["name"] == "modelscope":
                # Text evidence is enough here; use the cheaper text model first.
                models = ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
            self.providers.append(
                (
                    provider["name"],
                    OpenAI(
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                    ),
                    tuple(models),
                )
            )
        if not self.providers:
            raise ValueError("No model provider available for math transcript enhancement")

    def _call(self, prompt: str) -> tuple[str, str]:
        errors: list[str] = []
        for provider_name, client, models in self.providers:
            for model in models:
                model_id = f"{provider_name}/{model}"
                t0 = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.03,
                        max_tokens=5000,
                        timeout=_TIMEOUT,
                    )
                    choice = response.choices[0]
                    text = (choice.message.content or "").strip()
                    if not text:
                        raise RuntimeError("empty response")
                    if str(getattr(choice, "finish_reason", "") or "").lower() == "length":
                        raise RuntimeError("truncated response")
                    print(
                        f"[MathTranscript] {model_id}: {len(prompt)} -> {len(text)} chars "
                        f"in {time.time() - t0:.0f}s",
                        flush=True,
                    )
                    return text, model_id
                except Exception as exc:
                    msg = f"{model_id}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"[MathTranscript] {msg}", flush=True)
        raise RuntimeError("All math transcript models failed: " + " | ".join(errors))

    def enhance(
        self,
        proofread_segments: list[dict],
        ppt_pages: list[dict] | None,
        *,
        raw_blackboard: str = "",
    ) -> MathTranscriptResult:
        if not proofread_segments:
            raise ValueError("AI-proofread timed segments are required")

        enhanced: list[dict] = []
        models: list[str] = []
        fallback_count = 0

        for index, seg in enumerate(proofread_segments, start=1):
            start_ms = int(seg.get("start_ms", 0) or 0)
            end_ms = int(seg.get("end_ms", start_ms) or start_ms)
            if end_ms < start_ms:
                start_ms, end_ms = end_ms, start_ms
            current = str(seg.get("text") or "").strip()
            if not current:
                continue

            start_sec, end_sec = start_ms // 1000, end_ms // 1000
            ppt = _ppt_in_window(ppt_pages or [], start_sec, end_sec)
            board = _board_in_window(raw_blackboard, start_sec, end_sec)

            # No visual source means there is no legitimate basis for restoration.
            if not ppt and not board:
                enhanced.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": current,
                        "math_enhance_status": "no_visual_evidence",
                    }
                )
                continue

            prompt = (
                f"【视频时段】{_fmt_ms(start_ms)}–{_fmt_ms(end_ms)}\n\n"
                f"【忠实 AI 校订语音转写】\n{current}\n\n"
                f"【同时段 PPT OCR 证据】\n{ppt or '（无）'}\n\n"
                f"【同时段黑板视觉证据】\n{board or '（无）'}"
            )
            candidate, model = self._call(prompt)
            safe, ratio = _safe_ratio(current, candidate)
            used_model = model

            if not safe:
                print(
                    f"[MathTranscript] unsafe expansion {index}/{len(proofread_segments)} "
                    f"{_fmt_ms(start_ms)}–{_fmt_ms(end_ms)}: "
                    f"{len(current)} -> {len(candidate)} ({ratio:.1%}); retrying.",
                    flush=True,
                )
                retry, retry_model = self._call(prompt + "\n\n" + _RETRY_PROMPT)
                retry_safe, retry_ratio = _safe_ratio(current, retry)
                models.extend([model, retry_model])
                if retry_safe:
                    candidate = retry
                    used_model = retry_model
                    ratio = retry_ratio
                else:
                    fallback_count += 1
                    candidate = current
                    used_model = "raw-proofread-fallback"
                    print(
                        f"[MathTranscript] retry still unsafe; preserving faithful "
                        f"transcript for {_fmt_ms(start_ms)}–{_fmt_ms(end_ms)}.",
                        flush=True,
                    )
            else:
                models.append(model)

            enhanced.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": candidate.strip(),
                    "math_enhance_status": (
                        "faithful_fallback"
                        if used_model == "raw-proofread-fallback"
                        else "visual_evidence_enhanced"
                    ),
                }
            )

        if not enhanced:
            raise RuntimeError("Math transcript enhancer produced no timed chunks")

        markdown_parts = [
            "# 数学增强语音转写",
            "",
            "> 该版本以忠实 AI 校订转写为底稿，仅在同时段 PPT/黑板有直接证据时恢复 ASR 丢失的数学符号。紫色“视觉补全”不是逐字语音，而是可追溯的视觉证据恢复；忠实逐字稿仍作为独立附件保留。",
            "",
        ]
        for seg in enhanced:
            markdown_parts.extend(
                [
                    f"## {_fmt_ms(seg['start_ms'])}–{_fmt_ms(seg['end_ms'])}",
                    "",
                    str(seg.get("text") or "").strip(),
                    "",
                ]
            )

        unique_models = []
        for model in models:
            if model and model not in unique_models:
                unique_models.append(model)
        model_label = "math-transcript-v1/" + (
            "+".join(unique_models) if unique_models else "no-llm"
        )
        if fallback_count:
            model_label += f"|faithful-fallback[{fallback_count}]"

        return MathTranscriptResult(
            markdown="\n".join(markdown_parts).strip() + "\n",
            segments=enhanced,
            model_label=model_label,
        )
