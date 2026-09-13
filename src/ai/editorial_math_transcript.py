"""Second-pass editorial reviewer for the complete math transcript.

The legacy enhancer's final pass protected purple visual formulas with plain-text
placeholders. Qwen sometimes dropped those placeholders even when it preserved
the underlying content, causing otherwise valid editorial reviews to be rejected.
This adapter keeps the authoritative visual spans in the prompt and validates
them verbatim instead. It also uses a wider but still bounded edit window so
normal Chinese editorial expansion is not mistaken for hallucination.
"""
from __future__ import annotations

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

紫色视觉补全是只读证据：
- 任何完整的 `<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕... </span>` 块都必须原样保留。
- 不得修改其中的公式、标签或文字，不得删除、复制、移动或重新生成它。
- 你可以修改紫色块前后的普通黑色文字。

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
- 不要把课堂内容摘要化，也不要照抄明显错误的 ASR；
- 每个紫色“视觉补全” span 必须逐字原样保留，尤其是其中的数学公式；
- 输出全部 CHUNK 编号，且每个 CHUNK 都必须有完整正文。
""".strip()


class EditorialMathTranscriptEnhancer(base.MathTranscriptEnhancer):
    """Use the established first pass plus a more permissive editorial second pass."""

    @staticmethod
    def _valid_editorial_candidate(source: str, candidate: str, *, course_title: str) -> bool:
        source_spoken = base._spoken_text(source).strip()
        candidate_spoken = base._spoken_text(candidate).strip()
        if not source_spoken or not candidate_spoken:
            return False

        # Editorial review may substantially restructure Chinese prose. The
        # lower bound still blocks summaries, while the upper bound allows
        # natural expansion when ASR swallowed words or formulas were clarified.
        ratio = len(candidate_spoken) / max(1, len(source_spoken))
        if not 0.50 <= ratio <= 3.20:
            return False

        # Keep enough of every source clause to prevent a fluent-looking summary
        # from replacing the actual lecture content.
        if base._source_coverage(source_spoken, candidate_spoken) < 0.72:
            return False
        if not base._clauses_preserved(source_spoken, candidate_spoken):
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
                if right in corrected_source and candidate_spoken.count(right) < corrected_source.count(right):
                    return False
                if wrong not in corrected_source and wrong in candidate_spoken:
                    return False
        return True

    def _editorial_prompt(self, batch, *, course_title, source, batch_start, guide):
        # Reuse the established full-context prompt, but explicitly tell the
        # model that visual spans are real read-only text rather than placeholders.
        prompt = self._final_prompt(
            batch,
            course_title=course_title,
            source=source,
            batch_start=batch_start,
            guide=guide,
        )
        return prompt.replace(
            "【第一轮增强完整正文；视觉公式占位符必须原位保留】",
            "【第一轮增强完整正文；紫色视觉补全 span 是只读证据，必须原样保留】",
        )

    def _final_batch(self, batch, *, course_title, source, batch_start, guide):
        prompt = self._editorial_prompt(
            batch,
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
