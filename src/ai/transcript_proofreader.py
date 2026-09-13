"""Evidence-constrained AI proofreading for timed ASR transcripts.

The goal is not to rewrite a lecture into polished prose. It is to correct
obvious ASR mistakes while preserving what the lecturer actually said and the
video timeline. Each 3-minute window is proofread independently with nearby PPT
OCR; Functional Analysis can additionally supply timestamped raw blackboard
frames as evidence.

A single unsafe model rewrite must not abort an entire lecture. If a first
proofreading result changes a window too aggressively, that window is retried
once with a stricter anti-expansion prompt. If the retry is still unsafe, the
original ASR is retained verbatim and explicitly marked as an unverified/raw
fallback rather than being silently labelled as AI-proofread.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from src.ai.ppt_dedup import clean_ppt_text
from src.data.transcript_store import (
    PROOFREAD_BOARD_MODEL_PREFIX,
    PROOFREAD_MODEL_PREFIX,
)
from src.runtime import config


_CHUNK_SEC = int(os.environ.get("TRANSCRIPT_PROOFREAD_CHUNK_SEC", "180"))
_CONTEXT_SEC = 45
_TIMEOUT = int(os.environ.get("TRANSCRIPT_PROOFREAD_TIMEOUT", "300"))
_MIN_RATIO = 0.55
_MAX_RATIO = 1.55

_MODELSCOPE_MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Instruct",
]

_BOARD_HEADING_RE = re.compile(
    r"^####\s+(\d{1,2}):(\d{2})(?::(\d{2}))?.*$",
    re.MULTILINE,
)


SYSTEM_PROMPT = r"""
你是“课堂语音转写校订器”，不是总结助手，也不是改写助手。

输入包含某个连续视频时段的原始 ASR、前后少量语音上下文，以及同时段附近的 PPT OCR；某些课程还会提供同时段黑板视觉转写。

任务：把【当前时段原始 ASR】校订成尽可能忠实于老师实际讲话的可读逐字稿。

硬性规则：
1. 保留老师原本的信息、论述顺序和口语语义，不做总结，不删掉实质内容，不把它改写成教材。
2. 可以修正明显的同音字、数学/专业术语、英文词、人名、断句和 ASR 幻觉。
3. PPT/黑板只是校对证据：它们可以帮助确认术语和公式，但不能把老师没有说过的板书内容硬塞进逐字稿。
4. 前后语音上下文只用于判断当前时段，不要把上下文重复输出。
5. 如果某处确实无法可靠判断，保留最接近原 ASR 的表达并标注 `[转写存疑]`；禁止凭学科常识补出一整句。
6. 不要添加解释、评价、标题、时间戳、Markdown 列表或“AI 补充”。
7. 数学符号可使用简洁 LaTeX `$...$`，但不必把所有普通口语强行公式化。
8. 直接输出校订后的【当前时段】正文，不要说明修改了什么。
""".strip()


STRICT_RETRY_SYSTEM_PROMPT = r"""
你是“极保守的课堂 ASR 校对器”。上一轮校订因为对原始语音改动过大而被安全检查拒绝。

这一次必须严格执行：
1. 只修改【当前时段原始 ASR】中明确可由上下文、PPT 或黑板证据确认的错字、术语、英文、人名和标点。
2. 严禁把 PPT、黑板上的完整定义、定理、公式、证明步骤补进逐字稿，除非原始 ASR 中已经明显说到了对应内容。
3. 严禁扩写、解释、总结、润色成教材语言；严禁重复前后文。
4. 尽量保持原始 ASR 的句子数量、信息量和长度。若不确定，宁可保留原 ASR，并在局部加 `[转写存疑]`。
5. 输出长度原则上应与原始 ASR 接近，不得明显变长或明显变短。
6. 不要输出标题、时间戳、说明、列表或修改理由，只输出当前时段正文。
""".strip()


@dataclass(frozen=True)
class ProofreadResult:
    markdown: str
    segments: list[dict]
    model_label: str


def _fmt(sec: int | float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _segment_bounds(seg: dict) -> tuple[int, int]:
    start = int(seg.get("start_ms", 0)) // 1000
    end = int(seg.get("end_ms", seg.get("start_ms", 0))) // 1000
    return start, max(start, end)


def _text_in_window(segments: list[dict], start: int, end: int) -> str:
    texts: list[str] = []
    for seg in segments:
        s, e = _segment_bounds(seg)
        if e < start or s > end:
            continue
        text = str(seg.get("text") or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts).strip()


def _ppt_in_window(pages: list[dict], start: int, end: int) -> str:
    out: list[str] = []
    for page in pages or []:
        sec = int(page.get("created_sec", 0))
        if sec < start - 90 or sec > end + 90:
            continue
        text = clean_ppt_text(page.get("text") or "").strip()
        if text:
            out.append(f"[PPT @ {_fmt(sec)}]\n{text}")
    return "\n\n".join(out) or "（无可用 PPT OCR）"


def _parse_board_time(match: re.Match) -> int:
    a = int(match.group(1))
    b = int(match.group(2))
    c = int(match.group(3) or 0)
    if match.group(3) is None:
        return a * 60 + b
    return a * 3600 + b * 60 + c


def _board_in_window(raw_board: str, start: int, end: int) -> str:
    if not raw_board:
        return "（无黑板视觉证据）"
    matches = list(_BOARD_HEADING_RE.finditer(raw_board))
    if not matches:
        return "（无可定位黑板视觉证据）"
    out: list[str] = []
    for i, match in enumerate(matches):
        sec = _parse_board_time(match)
        if sec < start - 75 or sec > end + 75:
            continue
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_board)
        block = raw_board[match.start():block_end].strip()
        if block:
            out.append(block)
    joined = "\n\n".join(out)
    if len(joined) > 9000:
        joined = joined[:9000] + "\n[黑板证据截断]"
    return joined or "（该时段无可用黑板视觉证据）"


def _safe_ratio(source: str, candidate: str) -> tuple[bool, float]:
    ratio = len(candidate) / max(1, len(source))
    return _MIN_RATIO <= ratio <= _MAX_RATIO, ratio


class TranscriptProofreader:
    def __init__(self):
        resolved = config.resolve_model_providers()
        self.providers: list[tuple[str, OpenAI, tuple[str, ...]]] = []
        for provider in resolved:
            models = list(provider["models"])
            if provider["name"] == "modelscope":
                override = os.environ.get("TRANSCRIPT_PROOFREAD_MODELS", "").strip()
                models = (
                    [m.strip() for m in override.split(",") if m.strip()]
                    if override else _MODELSCOPE_MODELS
                )
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
            raise ValueError("No model provider available for transcript proofreading")

    def _call(self, prompt: str, *, system_prompt: str = SYSTEM_PROMPT) -> tuple[str, str]:
        errors: list[str] = []
        for provider_name, client, models in self.providers:
            for model in models:
                model_id = f"{provider_name}/{model}"
                t0 = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.05,
                        max_tokens=6000,
                        timeout=_TIMEOUT,
                    )
                    choice = response.choices[0]
                    text = (choice.message.content or "").strip()
                    if not text:
                        raise RuntimeError("empty response")
                    if str(getattr(choice, "finish_reason", "") or "").lower() == "length":
                        raise RuntimeError("truncated response")
                    print(
                        f"[TranscriptProofreader] {model_id}: "
                        f"{len(prompt)} -> {len(text)} chars in {time.time()-t0:.0f}s",
                        flush=True,
                    )
                    return text, model_id
                except Exception as exc:
                    errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
                    print(f"[TranscriptProofreader] {errors[-1]}", flush=True)
        raise RuntimeError(
            "All transcript proofreading models failed: " + " | ".join(errors)
        )

    def proofread(
        self,
        segments: list[dict],
        ppt_pages: list[dict] | None,
        *,
        raw_blackboard: str = "",
    ) -> ProofreadResult:
        if not segments:
            raise ValueError("Timestamped ASR segments are required for proofreading")

        max_end = max(_segment_bounds(seg)[1] for seg in segments)
        chunks: list[dict] = []
        models: list[str] = []
        fallback_windows = 0

        for start in range(0, max_end + 1, _CHUNK_SEC):
            end = min(max_end, start + _CHUNK_SEC)
            current = _text_in_window(segments, start, end)
            if not current:
                continue
            before = _text_in_window(
                segments,
                max(0, start - _CONTEXT_SEC),
                start - 1,
            )
            after = _text_in_window(
                segments,
                end + 1,
                end + _CONTEXT_SEC,
            )
            ppt = _ppt_in_window(ppt_pages or [], start, end)
            board = (
                _board_in_window(raw_blackboard, start, end)
                if raw_blackboard
                else "（本课程不使用黑板视觉证据）"
            )

            prompt = (
                f"【视频时段】{_fmt(start)}–{_fmt(end)}\n\n"
                f"【前文语音上下文，只读】\n{before or '（无）'}\n\n"
                f"【当前时段原始 ASR】\n{current}\n\n"
                f"【后文语音上下文，只读】\n{after or '（无）'}\n\n"
                f"【同时段 PPT OCR 校对证据】\n{ppt}\n\n"
                f"【同时段黑板视觉校对证据】\n{board}"
            )

            corrected, model = self._call(prompt)
            safe, ratio = _safe_ratio(current, corrected)
            fallback = False

            if not safe:
                print(
                    "[TranscriptProofreader] unsafe first pass for "
                    f"{_fmt(start)}–{_fmt(end)}: {len(current)} -> "
                    f"{len(corrected)} ({ratio:.1%}); retrying conservatively.",
                    flush=True,
                )
                retry_prompt = (
                    prompt
                    + "\n\n【安全重试要求】\n"
                    + f"原始 ASR 长度为 {len(current)} 个字符。"
                    + "上一轮输出因改动过大被拒绝。请只做最小必要校正，"
                    + "不要补充任何板书/PPT 中但原 ASR 没有说出的内容。"
                )
                retried, retry_model = self._call(
                    retry_prompt,
                    system_prompt=STRICT_RETRY_SYSTEM_PROMPT,
                )
                retry_safe, retry_ratio = _safe_ratio(current, retried)
                models.extend([model, retry_model])

                if retry_safe:
                    corrected = retried
                    model = retry_model
                    ratio = retry_ratio
                    print(
                        "[TranscriptProofreader] conservative retry accepted for "
                        f"{_fmt(start)}–{_fmt(end)}: {len(current)} -> "
                        f"{len(corrected)} ({ratio:.1%})",
                        flush=True,
                    )
                else:
                    fallback = True
                    fallback_windows += 1
                    corrected = current
                    print(
                        "[TranscriptProofreader] conservative retry still unsafe for "
                        f"{_fmt(start)}–{_fmt(end)}: {len(current)} -> "
                        f"{len(retried)} ({retry_ratio:.1%}); preserving raw ASR "
                        "and marking this window as unverified instead of aborting lecture.",
                        flush=True,
                    )
            else:
                models.append(model)

            chunks.append(
                {
                    "start_ms": start * 1000,
                    "end_ms": end * 1000,
                    "text": corrected.strip(),
                    "proofread_status": "raw_fallback" if fallback else "ai_proofread",
                }
            )

        if not chunks:
            raise RuntimeError("Transcript proofreader produced no timed chunks")

        markdown_parts = [
            "# AI 校订语音转写",
            "",
            "> 本附件以原始 ASR 为底稿，由 AI 结合同时段课件/板书证据校订。AI 只用于纠正明显识别错误；无法可靠判断处尽量保留原文。若某个时段的 AI 校订未通过安全检查，该时段会明确标为“原始 ASR 回退”，不会把可疑扩写伪装成已校订内容。",
            "",
        ]
        for chunk in chunks:
            s = chunk["start_ms"] // 1000
            e = chunk["end_ms"] // 1000
            markdown_parts.extend([f"### {_fmt(s)}–{_fmt(e)}", ""])
            if chunk.get("proofread_status") == "raw_fallback":
                markdown_parts.extend(
                    [
                        "> ⚠️ 本时段 AI 校订未通过安全检查，以下保留原始 ASR；内容可能存在识别错误。",
                        "",
                    ]
                )
            markdown_parts.extend([chunk["text"], ""])

        prefix = (
            PROOFREAD_BOARD_MODEL_PREFIX
            if raw_blackboard
            else PROOFREAD_MODEL_PREFIX
        )
        model_label = "+".join(dict.fromkeys(models))
        if fallback_windows:
            model_label += f"+raw-fallback[{fallback_windows}]"
            print(
                f"[TranscriptProofreader] completed with {fallback_windows} raw-ASR "
                "fallback window(s); all other windows passed AI proofreading.",
                flush=True,
            )
        return ProofreadResult(
            markdown="\n".join(markdown_parts).strip(),
            segments=chunks,
            model_label=prefix + model_label,
        )
