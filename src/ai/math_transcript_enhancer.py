"""Readable math transcript built on top of the faithful audit transcript.

The first pass may make high-confidence contextual ASR corrections and restore
formulas from same-window PPT/blackboard evidence.  After every chunk exists, a
second model pass reads the complete lecture for terminology consistency and
then reviews every chunk again.  The faithful AI-proofread transcript is never
modified and remains the audit trail.
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
_FINAL_BATCH = max(1, int(os.environ.get("MATH_TRANSCRIPT_FINAL_BATCH_SIZE", "3")))
_MIN_RATIO, _MAX_RATIO = 0.90, 2.20
_MIN_SOURCE_COVERAGE = 0.88
_MIN_CLAUSE_COVERAGE = 0.65
_BOARD_MAX = int(os.environ.get("MATH_TRANSCRIPT_BOARD_CHARS", "4000"))
_BOARD_CONTEXT_MAX = int(os.environ.get("MATH_TRANSCRIPT_BOARD_CONTEXT_CHARS", "900"))
_PPT_MAX = int(os.environ.get("MATH_TRANSCRIPT_PPT_CHARS", "1400"))
_CONTEXT_MAX = int(os.environ.get("MATH_TRANSCRIPT_CONTEXT_CHARS", "700"))
_FINAL_GUIDE_MAX = int(os.environ.get("MATH_TRANSCRIPT_FINAL_GUIDE_CHARS", "6000"))
_FINAL_GUIDE_INPUT_MAX = int(
    os.environ.get("MATH_TRANSCRIPT_FINAL_GUIDE_INPUT_CHARS", "18000")
)
_SUMMARY_REFERENCE_MAX = int(
    os.environ.get("MATH_TRANSCRIPT_SUMMARY_REFERENCE_CHARS", "24000")
)
_BOARD_RE = re.compile(r"^####\s+(\d{1,2}):(\d{2})(?::(\d{2}))?.*$", re.M)
_OUT_RE = re.compile(
    r"<<<CHUNK\s+(\d+)>>>\s*(.*?)\s*<<<END\s+CHUNK\s+\1>>>", re.I | re.S
)
_PREVIOUS_VISUAL_REF_RE = re.compile(
    r"(?:这个|那个|上面|前面|刚才|之前)(?:式子|公式|等式|不等式|定义|表达式|板书)"
    r"|(?:上式|下式|该式|此式|如图|板书上)"
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
5. 优先使用“本 CHUNK”的 PPT/黑板证据恢复原语音没有说清楚的变量、集合、映射、量词、等式、不等式或公式。若老师在当前语音中明确说了“这个式子”“上式”“刚才写的”等指代，才可使用“最近前序板书证据”恢复被指代的公式；没有明确语音指代时，绝不能从前序板书搬运公式，也不能凭学科常识补公式。
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
上一轮存在缺块、删句或改动过大。重新处理整个批次：逐句保留原文，包括所有课程介绍、通知、作业和评分信息；只做高置信度的局部 ASR 纠错和最小必要公式恢复；不得摘要或用板书替换语音。ASR 纠错保持黑色，只有视觉证据新增的内容才放在 {_VIS_OPEN}...{_VIS_CLOSE} 中；相邻语音只能辅助纠错；最近前序板书只有在当前语音明确指代上式时才能用于恢复公式；禁止搬运完整定义/定理/证明；剔除紫色视觉补全后，黑色正文的信息量和长度必须与原转写基本一致；必须输出全部编号。
""".strip()

_FINAL_GUIDE_SYSTEM = """
你是数学课堂转写的全稿内容审校员。你会读到“AI 课程总结”参考以及按时间排列的完整增强转写。

你的任务不是重写课程，而是提取供第二轮逐段终审使用的全局一致性指南：
- 从 AI 课程总结中提取课程主线、数学术语、定义、公式记法、人物名、教材名和英文词；
- 对照转写中反复出现的内容，判断明显的 ASR 错词、断裂病句和数学语言错误；
- 根据全课重复语境可以高置信度确认的 ASR 同音/近音错误及“错误 → 正确”映射；
- 前后不一致但能够依据重复出现内容确定的称呼或记号。

课程总结是校对参考而不是逐字稿：不得用总结取代课堂讲述，不得补充两份材料都未出现的知识。
只输出简短但具体的项目列表；证据不足的项目不要写。不要输出改写后的转写。
""".strip()

_FINAL_REVIEW_SYSTEM = rf"""
你是数学课堂“完整转写编辑与终审员”。第一轮增强已经完成；现在依据 AI 课程总结、全课内容指南和相邻段落，把 ASR 口语稿校订为句意通顺、数学语言准确的课堂文字稿。

允许：
1. 增、删、改文字：删除“呃、啊、这个”等无意义口头填充、重复起句和明显 ASR 噪声；合并重复句；拆分或重组病句；补足被 ASR 吞掉但能由上下文确定的主语、谓语和连接词；
2. 修正同音/近音错词、数学术语、人物名、教材名、英文词、断句和标点，使每句话自然通顺；
3. 依据全课内容指南修正已有数学表达与 `$...$` 记法，补回总结和上下文能够明确支持的必要符号，使定义、推导和结论在数学上准确；
4. 保留老师讲授的实质信息、论证顺序、课程通知和例子，但可将口语整理为清晰书面语。输出应当能让学生直接阅读，而不是保留错误的 ASR 原貌。

禁止：
1. 不得改变老师的核心观点、数学结论和讲授顺序，不得删除任何实质知识点、课程要求或例子；
2. 不得把课程总结整段复制到转写中，不得凭学科常识编造课堂未讲的证明或知识；确实无法判断的内容标 `[语音存疑]`；
3. 输入中所有从 `{_VIS_OPEN}` 开始到 `{_VIS_CLOSE}` 结束的紫色视觉补全必须逐字、逐符号保留，不能删除、复制或无依据改写；
4. 相邻段落只用于理解当前 CHUNK，不得把别的时间段内容复制进来；
5. 一个输入 CHUNK 对应一个输出 CHUNK，编号一致，不得遗漏。必须对每段做实际编辑，不要原样照抄仍然明显错误的 ASR 句子。

严格输出：
<<<CHUNK 1>>>
终审后的完整正文
<<<END CHUNK 1>>>
不得在 CHUNK 块之外输出任何内容。
""".strip()

_FINAL_RETRY = """
上一轮输出缺块、过短或破坏了视觉公式。请重新编辑全部编号：保留每段的实质知识点、通知、例子和论证顺序，但主动删除口头赘词与重复、修复 ASR 病句、纠正数学术语和公式，使文字明显比原稿通顺准确。所有紫色视觉补全必须逐字保留；不得用课程总结替换老师讲述；输出全部编号。
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


def _has_formula_evidence(
    source: str,
    pages: list[dict],
    raw_board: str,
    start: int,
    end: int,
) -> bool:
    """Accept prior-board formula evidence only when speech points back to it."""
    if _ppt(pages, start, end) or _board(raw_board, start, end):
        return True
    return bool(
        _previous_board_context(raw_board, start)
        and _PREVIOUS_VISUAL_REF_RE.search(str(source or ""))
    )


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


def _visual_fragments(text: str) -> list[str]:
    """Return complete visual-restoration fragments in source order."""
    return re.findall(
        re.escape(_VIS_OPEN) + r".*?" + re.escape(_VIS_CLOSE),
        str(text or ""),
        flags=re.S,
    )


def _valid_final_candidate(source: str, candidate: str, *, course_title: str) -> bool:
    """Accept a substantive edit while guarding against loss of whole chunks."""
    source_spoken = _spoken_text(source).strip()
    candidate_spoken = _spoken_text(candidate).strip()
    if not source_spoken or not candidate_spoken:
        return False
    # Editing may legitimately remove fillers/repetitions or repair swallowed
    # words, so character-level similarity is intentionally not required here.
    # The broad ratio still rejects summaries and runaway textbook expansion.
    ratio = len(candidate_spoken) / max(1, len(source_spoken))
    if not 0.45 <= ratio <= 2.50:
        return False
    # The final pass has no visual evidence and therefore may neither add nor
    # alter formula restorations accepted by the evidence-aware first pass.
    if _visual_fragments(candidate) != _visual_fragments(source):
        return False
    corrected_source = _correct_high_confidence_asr(source_spoken, course_title)
    corrected_candidate = candidate_spoken
    title = str(course_title or "")
    for keyword, replacements in _HIGH_CONFIDENCE_ASR.items():
        if keyword not in title:
            continue
        for wrong, right in replacements:
            if (
                right in corrected_source
                and corrected_candidate.count(right) < corrected_source.count(right)
            ):
                return False
            if wrong not in corrected_source and wrong in corrected_candidate:
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
                models = override or models
            self.providers.append(
                (p["name"], OpenAI(api_key=p["api_key"], base_url=p["base_url"]), tuple(models))
            )
        if not self.providers:
            raise ValueError("No model provider available for math transcript enhancement")

    def _call(self, prompt: str) -> tuple[str, str]:
        return self._call_with_system(
            prompt,
            system=_SYSTEM,
            max_tokens=9000,
            stage="enhance",
        )

    def _call_with_system(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        stage: str,
    ) -> tuple[str, str]:
        errors = []
        for provider, client, models in self.providers:
            for model in models:
                model_id = f"{provider}/{model}"
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=.03,
                        max_tokens=max_tokens,
                        timeout=_TIMEOUT,
                    )
                    choice = resp.choices[0]
                    text = (choice.message.content or "").strip()
                    if not text:
                        raise RuntimeError("empty response")
                    if str(getattr(choice, "finish_reason", "") or "").lower() == "length":
                        raise RuntimeError("truncated response")
                    print(
                        f"[MathTranscript:{stage}] {model_id}: "
                        f"{len(prompt)} -> {len(text)} chars "
                        f"in {time.time()-t0:.0f}s", flush=True
                    )
                    return text, model_id
                except Exception as exc:
                    msg = f"{model_id}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"[MathTranscript] {msg}", flush=True)
        raise RuntimeError("All math transcript models failed: " + " | ".join(errors))

    def _global_review_guide(
        self,
        enhanced: list[dict],
        *,
        course_title: str,
        summary_reference: str,
    ) -> tuple[str, list[str]]:
        """Read every assembled chunk, then synthesize a compact global guide.

        Sending a long lecture in one request can make the provider truncate
        its response.  We therefore extract guides from contiguous, complete
        groups of chunks and merge those guides once.  Every chunk is still
        read after first-pass assembly; no transcript text is sampled away.
        """
        models: list[str] = []
        summary = _trim(
            str(summary_reference or "").strip(),
            _SUMMARY_REFERENCE_MAX,
            "AI课程总结参考",
        )
        summary_guide = "（AI课程总结不可用）"
        if summary:
            try:
                summary_guide, model = self._call_with_system(
                    f"【课程名称】{course_title or '（未知）'}\n"
                    f"【AI 课程总结参考】\n{summary}\n\n"
                    "请提取用于校订课堂逐字稿的课程主线、数学术语、定义和公式记法。",
                    system=_FINAL_GUIDE_SYSTEM,
                    max_tokens=2200,
                    stage="summary-reference-guide",
                )
                models.append(model)
            except Exception as exc:
                print(
                    f"[MathTranscript:summary-reference-guide] unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        blocks = [
            f"## {_fmt(seg['start_ms'])}–{_fmt(seg['end_ms'])}\n{seg['text']}"
            for seg in enhanced
        ]
        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for block in blocks:
            extra = len(block) + (2 if current else 0)
            if current and current_len + extra > _FINAL_GUIDE_INPUT_MAX:
                groups.append(current)
                current, current_len = [], 0
            current.append(block)
            current_len += extra
        if current:
            groups.append(current)

        partials: list[str] = []
        for index, group in enumerate(groups, 1):
            prompt = (
                f"【课程名称】{course_title or '（未知）'}\n"
                f"【由 AI 课程总结提取的校对参考】\n{summary_guide}\n\n"
                f"【全课术语审校分段 {index}/{len(groups)}；连续完整 CHUNK】\n\n"
                + "\n\n".join(group)
                + "\n\n请提取本分段供全课终审使用的术语与一致性指南。"
            )
            try:
                guide, model = self._call_with_system(
                    prompt,
                    system=_FINAL_GUIDE_SYSTEM,
                    max_tokens=1800,
                    stage=f"global-guide-{index}",
                )
                partials.append(guide)
                models.append(model)
            except Exception as exc:
                print(
                    f"[MathTranscript:global-guide] segment {index}/{len(groups)} "
                    f"unavailable: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        if not partials:
            return "（全课指南不可用；仅进行保守逐段终审）", models
        if len(partials) == 1:
            return _trim(partials[0], _FINAL_GUIDE_MAX, "全课术语指南"), models

        merge_prompt = (
            f"【课程名称】{course_title or '（未知）'}\n"
            "【按全课顺序提取的分段术语指南】\n\n"
            + "\n\n".join(
                f"### 分段 {i}\n{guide}" for i, guide in enumerate(partials, 1)
            )
            + "\n\n请去重、合并为一份保守的全课术语与一致性指南。"
        )
        try:
            merged, model = self._call_with_system(
                merge_prompt,
                system=_FINAL_GUIDE_SYSTEM,
                max_tokens=2400,
                stage="global-guide-merge",
            )
            models.append(model)
            guide = merged
        except Exception as exc:
            print(
                f"[MathTranscript:global-guide] merge unavailable; using all "
                f"partial guides: {type(exc).__name__}: {exc}",
                flush=True,
            )
            guide = "\n".join(partials)
        return _trim(guide, _FINAL_GUIDE_MAX, "全课术语指南"), models

    def _final_prompt(
        self,
        batch: list[dict],
        *,
        course_title: str,
        source: list[dict],
        batch_start: int,
        guide: str,
    ) -> str:
        blocks = [
            f"【课程名称】{course_title or '（未知）'}",
            f"【参考 AI 课程总结并通读完整转写后形成的全课内容与校对指南】\n{guide}",
        ]
        for i, seg in enumerate(batch, 1):
            before, after = _adjacent_context(source, batch_start + i - 1)
            blocks.append(
                f"===== FINAL INPUT CHUNK {i} =====\n"
                f"【视频时段】{_fmt(seg['start_ms'])}–{_fmt(seg['end_ms'])}\n"
                f"【相邻上文（只用于判断术语）】\n{before or '（无）'}\n\n"
                f"【第一轮增强完整正文】\n{seg['text']}\n\n"
                f"【相邻下文（只用于判断术语）】\n{after or '（无）'}"
            )
        return "\n\n".join(blocks)

    def _final_batch(
        self,
        batch: list[dict],
        *,
        course_title: str,
        source: list[dict],
        batch_start: int,
        guide: str,
    ) -> tuple[list[dict], list[str], int]:
        prompt = self._final_prompt(
            batch,
            course_title=course_title,
            source=source,
            batch_start=batch_start,
            guide=guide,
        )
        models: list[str] = []
        try:
            text, model = self._call_with_system(
                prompt,
                system=_FINAL_REVIEW_SYSTEM,
                max_tokens=9000,
                stage="final-review",
            )
            models.append(model)
            parsed = _parse(text, len(batch))
        except Exception as exc:
            print(
                f"[MathTranscript:final-review] batch unavailable; preserving first pass: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return [
                {**seg, "final_review_status": "first_pass_api_fallback"}
                for seg in batch
            ], models, len(batch)

        bad = {
            i
            for i, seg in enumerate(batch, 1)
            if not _valid_final_candidate(
                seg["text"], parsed.get(i, ""), course_title=course_title
            )
        }
        if bad:
            try:
                retry, model = self._call_with_system(
                    prompt + "\n\n" + _FINAL_RETRY,
                    system=_FINAL_REVIEW_SYSTEM,
                    max_tokens=9000,
                    stage="final-review-retry",
                )
                models.append(model)
                retry_parsed = _parse(retry, len(batch))
                for i in list(bad):
                    seg = batch[i - 1]
                    if _valid_final_candidate(
                        seg["text"], retry_parsed.get(i, ""), course_title=course_title
                    ):
                        parsed[i] = retry_parsed[i]
                        bad.remove(i)
            except Exception as exc:
                print(
                    f"[MathTranscript:final-review] retry unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        out = []
        for i, seg in enumerate(batch, 1):
            if i in bad or not parsed.get(i):
                out.append({**seg, "final_review_status": "first_pass_safety_fallback"})
            else:
                out.append(
                    {
                        **seg,
                        "text": parsed[i].strip(),
                        "final_review_status": "ai_final_reviewed",
                    }
                )
        return out, models, len(bad)

    def _final_review(
        self,
        enhanced: list[dict],
        *,
        course_title: str,
        summary_reference: str,
    ) -> tuple[list[dict], list[str], int]:
        """Run a complete second pass after all first-pass chunks exist."""
        guide, models = self._global_review_guide(
            enhanced,
            course_title=course_title,
            summary_reference=summary_reference,
        )
        reviewed: list[dict] = []
        fallbacks = 0
        batches = (len(enhanced) + _FINAL_BATCH - 1) // _FINAL_BATCH
        for start in range(0, len(enhanced), _FINAL_BATCH):
            batch = enhanced[start:start + _FINAL_BATCH]
            print(
                f"[MathTranscript:final-review] batch {start//_FINAL_BATCH+1}/{batches}: "
                f"{len(batch)} chunk(s)",
                flush=True,
            )
            out, used, failed = self._final_batch(
                batch,
                course_title=course_title,
                source=enhanced,
                batch_start=start,
                guide=guide,
            )
            reviewed.extend(out)
            models.extend(used)
            fallbacks += failed
        return reviewed, models, fallbacks

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
                f"【最近前序板书证据（用于术语判断；仅在当前语音明确指代上式时可恢复公式）】\n"
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
            has_visual = _has_formula_evidence(
                seg["text"], pages, raw_board, s, e
            )
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
                    has_visual = _has_formula_evidence(
                        seg["text"], pages, raw_board, s, e
                    )
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
        summary_reference: str = "",
        first_pass_segments=None,
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
        seeded = [
            x for x in (self._normalise(s) for s in first_pass_segments or [])
            if x is not None
        ]
        seed_matches = len(seeded) == len(source) and all(
            a["start_ms"] == b["start_ms"] and a["end_ms"] == b["end_ms"]
            for a, b in zip(seeded, source)
        )
        enhanced, models, fallbacks = [], [], 0
        if seed_matches:
            enhanced = seeded
            models.append("cached-evidence-pass-v7")
            print(
                f"[MathTranscript] Reusing {len(seeded)} v7 evidence-aware chunks; "
                "reserving model capacity for required editorial review.",
                flush=True,
            )
        else:
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

        enhanced, final_models, final_fallbacks = self._final_review(
            enhanced,
            course_title=course_title,
            summary_reference=summary_reference,
        )
        models.extend(final_models)
        if final_fallbacks:
            raise RuntimeError(
                f"editorial review incomplete: {final_fallbacks}/{len(enhanced)} "
                "chunks were not accepted"
            )

        md = [
            "# 完整课堂语音转写（AI 校订与公式补全）", "",
            "> 本附录以老师的完整讲授内容为基础。第一轮依据课程上下文和视觉证据校正 ASR、补全公式；全部内容生成后，第二轮大模型参考前文“AI 课程总结”通读全课，删除口头赘词与重复、重组病句，并校正数学术语和公式。实质知识点、通知、例子与讲授顺序保持不变；有直接视觉证据恢复的公式以紫色“视觉补全”显示，忠实校订底稿仍保存在系统中，便于回查。", "",
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
        label = "math-transcript-v9/" + ("+".join(unique) if unique else "no-llm")
        if fallbacks:
            label += f"|safe-base-fallback[{fallbacks}]"
        if final_fallbacks:
            label += f"|final-review-fallback[{final_fallbacks}]"
        return MathTranscriptResult("\n".join(md).strip() + "\n", enhanced, label)
