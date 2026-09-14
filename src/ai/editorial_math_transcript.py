"""Second-pass editorial reviewer for the complete math transcript.

Spoken prose is genuinely editable, while accepted visual formulas are protected
and restored deterministically.  A rejected model rewrite never discards the
already evidence-reviewed first pass or aborts an otherwise usable PDF.
"""
from __future__ import annotations

import re

from src.ai import math_transcript_enhancer as base

_EDITORIAL_SYSTEM = r"""
你是数学课堂“完整语音转写编辑与终审员”。第一轮增强稿已经依据课程上下文、PPT和黑板视觉证据完成；现在要在全部内容生成后，对每个时间段做真正的整体文字校对。

你的目标是把 ASR 稿整理成学生可以直接阅读的课堂文字稿，而不是把它原样保留下来。

允许并应当做：
1. 删除“呃、啊、这个、然后”等无意义口头填充、重复起句和明显 ASR 噪声；合并重复句。
2. 修复 ASR 造成的同音/近音错词、数学术语、人名、教材名、英文词、断句和标点。
3. 依据课程名称、AI 课程总结、全课校对指南以及当前段落和相邻段落的语义，修复病句、补足能够高置信度确定的主语/谓语/连接词，并把口语表达整理成自然的书面课堂语言。
4. 对数学语言尤其严格：统一定义、符号、集合、映射、量词、等式和不等式的写法；只使用课程总结、全课指南和当前转写已经提供的证据，不得凭学科常识编造课堂没有出现的内容。
5. 可以增、删、改文字，但必须保留老师实际讲授的所有实质知识点、课程通知、作业要求、评分方式、例子、论证顺序和结论。不能把一段压缩成摘要。
6. 这是文字校订，不是扩写讲义。不要新增解释、定义、例子或推导，也不要把公式另行展开说明。校订后正文通常应为原文长度的 70%–160%。

紫色视觉补全是只读证据：
- 输入中的 `[[VISUAL_FORMULA_n]]` 是程序保护的视觉公式位置标记。
- 每个标记必须原样保留一次，不得删除、复制、改名或移动；程序会在输出后恢复真实公式。
- 不要自行输出 HTML，也不要改写或解释这些标记代表的公式。

如果某处仍无法确定，使用 `[语音存疑]`，不要编造。

严格输出：
<<<CHUNK 1>>>
终审后的完整正文
<<<END CHUNK 1>>>
不得在 CHUNK 块之外输出任何文字。
""".strip()

_RETRY = r"""
上一轮某些段落没有通过保真验收。请重新编辑全部编号：
- 保留全部实质知识点、通知、例子、数学结论和讲授顺序；
- 主动修复明显 ASR 错词、病句和数学语言错误，删除无意义口头重复；
- 不要把课堂内容摘要化，不要照抄明显错误的 ASR，也不要扩写成讲义；
- 每个 `[[VISUAL_FORMULA_n]]` 必须原样保留一次；
- 输出全部 CHUNK 编号，且每个 CHUNK 都必须有完整正文。
""".strip()


_TOKEN_RE = re.compile(r"\[\[VISUAL_FORMULA_\d+\]\]")


def _nearest_boundary(text: str, position: int) -> int:
    """Avoid inserting a recovered formula in the middle of a word."""
    position = max(0, min(len(text), position))
    candidates = [position]
    for distance in range(0, 49):
        for point in (position - distance, position + distance):
            if not 0 <= point <= len(text):
                continue
            if point == 0 or point == len(text):
                candidates.append(point)
            elif text[point - 1] in "\n。！？；：，、,.!?;: ":
                return point
    return candidates[-1]


def _restore_authoritative_visuals(
    candidate: str,
    protected_source: str,
    fragments: list[str],
) -> str:
    """Restore every source visual exactly, even if the model drops a token.

    Exact tokens retain their model-selected position.  If Qwen drops or
    duplicates any token, all source visuals are reinserted in source order at
    approximately the same relative positions in the edited prose.
    """
    if not fragments:
        return str(candidate or "")
    exact = base._restore_visual_fragments(candidate, fragments)
    if exact:
        return exact

    source_positions: list[int] = []
    plain_count = 0
    cursor = 0
    for index in range(1, len(fragments) + 1):
        token = base._VIS_TOKEN.format(index=index)
        token_at = protected_source.find(token, cursor)
        if token_at < 0:
            token_at = cursor
        plain_count += len(protected_source[cursor:token_at])
        source_positions.append(plain_count)
        cursor = token_at + len(token)
    source_plain = _TOKEN_RE.sub("", protected_source)
    candidate_plain = _TOKEN_RE.sub("", str(candidate or "")).strip()
    if not candidate_plain:
        return ""

    insertions: list[tuple[int, str]] = []
    for source_position, fragment in zip(source_positions, fragments):
        relative = source_position / max(1, len(source_plain))
        target = _nearest_boundary(candidate_plain, round(relative * len(candidate_plain)))
        insertions.append((target, fragment))
    restored = candidate_plain
    for target, fragment in reversed(insertions):
        restored = restored[:target] + fragment + restored[target:]
    return restored


class EditorialMathTranscriptEnhancer(base.MathTranscriptEnhancer):
    """Use the established first pass plus a more permissive editorial second pass."""

    @staticmethod
    def _valid_editorial_candidate(source: str, candidate: str, *, course_title: str) -> bool:
        source_spoken = base._spoken_text(source).strip()
        candidate_spoken = base._spoken_text(candidate).strip()
        if not source_spoken or not candidate_spoken:
            return False

        # Editorial review may substantially restructure Chinese prose. Keep
        # only broad guards against summaries and runaway textbook expansion;
        # character-by-character overlap is incompatible with real rewriting.
        ratio = len(candidate_spoken) / max(1, len(source_spoken))
        if not 0.45 <= ratio <= 3.20:
            return False

        # Purple visual restorations are authoritative and must survive exactly.
        if base._visual_fragments(candidate) != base._visual_fragments(source):
            return False

        # Course-scoped high-confidence corrections must remain corrections.
        title = str(course_title or "")
        corrected_source = base._correct_high_confidence_asr(source_spoken, title)
        for keyword, replacements in base._HIGH_CONFIDENCE_ASR.items():
            if keyword not in title:
                continue
            for wrong, right in replacements:
                if wrong not in corrected_source and wrong in candidate_spoken:
                    return False
        return True

    def _editorial_prompt(self, batch, *, course_title, source, batch_start, guide):
        return self._final_prompt(
            batch,
            course_title=course_title,
            source=source,
            batch_start=batch_start,
            guide=guide,
        )

    def _final_batch(self, batch, *, course_title, source, batch_start, guide):
        protected_batch = []
        visual_fragments = []
        for seg in batch:
            protected, fragments = base._protect_visual_fragments(seg["text"])
            protected_batch.append({**seg, "text": protected})
            visual_fragments.append(fragments)

        prompt = self._editorial_prompt(
            protected_batch,
            course_title=course_title,
            source=source,
            batch_start=batch_start,
            guide=guide,
        )
        models = []
        try:
            text, model = self._call_with_system(
                prompt,
                system=_EDITORIAL_SYSTEM,
                max_tokens=12000,
                stage="final-editorial-review",
            )
            models.append(model)
            parsed = base._parse(text, len(batch))
            parsed = {
                i: _restore_authoritative_visuals(
                    value, protected_batch[i - 1]["text"], visual_fragments[i - 1]
                )
                for i, value in parsed.items()
            }
        except Exception as exc:
            print(
                f"[MathTranscript:final-editorial-review] batch unavailable: "
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
            if not self._valid_editorial_candidate(
                seg["text"], parsed.get(i, ""), course_title=course_title
            )
        }
        for i in sorted(bad):
            candidate = parsed.get(i, "")
            print(
                f"[MathTranscript:final-editorial-review] chunk {i} rejected: "
                f"source_chars={len(base._spoken_text(batch[i - 1]['text']))}, "
                f"candidate_chars={len(base._spoken_text(candidate))}, "
                f"visuals_preserved="
                f"{base._visual_fragments(candidate) == base._visual_fragments(batch[i - 1]['text'])}",
                flush=True,
            )

        if bad:
            try:
                retry, model = self._call_with_system(
                    prompt + "\n\n" + _RETRY,
                    system=_EDITORIAL_SYSTEM,
                    max_tokens=12000,
                    stage="final-editorial-review-retry",
                )
                models.append(model)
                retry_parsed = base._parse(retry, len(batch))
                retry_parsed = {
                    i: _restore_authoritative_visuals(
                        value, protected_batch[i - 1]["text"], visual_fragments[i - 1]
                    )
                    for i, value in retry_parsed.items()
                }
                for i in list(bad):
                    candidate = retry_parsed.get(i, "")
                    if self._valid_editorial_candidate(
                        batch[i - 1]["text"], candidate, course_title=course_title
                    ):
                        parsed[i] = candidate
                        bad.remove(i)
            except Exception as exc:
                print(
                    f"[MathTranscript:final-editorial-review] retry unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        out = []
        preserved = 0
        for i, seg in enumerate(batch, 1):
            if i in bad or not parsed.get(i):
                preserved += 1
                out.append({**seg, "final_review_status": "ai_reviewed_source_preserved"})
            else:
                out.append(
                    {
                        **seg,
                        "text": parsed[i].strip(),
                        "final_review_status": "ai_final_reviewed",
                    }
                )
        if preserved:
            print(
                f"[MathTranscript:final-editorial-review] preserved {preserved} "
                "already-enhanced chunk(s) after unsafe editorial output",
                flush=True,
            )
        # Every chunk was presented to the editorial model.  Unsafe rewrites
        # retain the evidence-reviewed source, so they are not fatal omissions.
        return out, models, 0
