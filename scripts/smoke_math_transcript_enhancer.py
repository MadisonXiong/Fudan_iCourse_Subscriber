"""Offline smoke checks for course-aware math transcript enhancement."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai.math_transcript_enhancer import (
    MathTranscriptEnhancer,
    _clauses_preserved,
    _comparison_text,
    _correct_high_confidence_asr,
    _has_formula_evidence,
    _protect_visual_fragments,
    _restore_visual_fragments,
    _source_coverage,
    _valid_final_candidate,
    _valid_candidate,
)
from src.api.emailer import Emailer
from src.runtime.config import MODEL_PROVIDERS
from src.data.math_transcript_store import (
    MATH_TRANSCRIPT_VERSION,
    source_fingerprint,
)
from scripts.resend_processed import (
    _estimated_duration,
    _raw_transcript_markdown,
    _reconstruct_timed_segments,
)


class _UnavailableEnhancer(MathTranscriptEnhancer):
    def __init__(self):
        pass

    def _call(self, prompt: str):
        raise RuntimeError("offline smoke test")


class _BrokenEnhancer:
    def enhance(self, *args, **kwargs):
        raise RuntimeError("enhancement unavailable")


class _FinalReviewEnhancer(MathTranscriptEnhancer):
    """Deterministic model double covering the full-transcript second pass."""

    def __init__(self):
        self.saw_complete_transcript = False
        self.saw_summary_reference = False

    def _call_with_system(self, prompt, *, system, max_tokens, stage):
        if stage == "summary-reference-guide":
            self.saw_summary_reference = "赋范空间是核心概念" in prompt
            return "- 课程主线：赋范空间", "test/summary-reference"
        if stage.startswith("global-guide"):
            self.saw_complete_transcript = (
                "赋饭空间" in prompt and "全课术语审校分段" in prompt
            )
            return "- 赋饭空间 → 赋范空间（全课重复语境确认）", "test/global-guide"
        if stage == "final-review":
            return (
                "<<<CHUNK 1>>>\n"
                "这里继续完整讨论赋范空间的定义和性质，老师提醒大家结合实变函数复习。\n"
                "<<<END CHUNK 1>>>",
                "test/final-review",
            )
        raise AssertionError(f"unexpected stage: {stage}")


class _FakeDb:
    def get_done_ppt_pages(self, sub_id: str):
        return []


class _FakeCompletions:
    def __init__(self, *, text="", error=None):
        self.text = text
        self.error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self.text),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])


def _fake_client(*, text="", error=None):
    return SimpleNamespace(chat=SimpleNamespace(
        completions=_FakeCompletions(text=text, error=error)
    ))


def main() -> None:
    assert MODEL_PROVIDERS[0]["name"] == "modelscope"
    assert MODEL_PROVIDERS[0]["models"][0].startswith("Qwen/")

    preferred = _fake_client(text="ModelScope result")
    unused_gemini = _fake_client(text="Gemini result")
    provider_test = object.__new__(MathTranscriptEnhancer)
    provider_test.providers = [
        ("modelscope", preferred, ("Qwen/test",)),
        ("gemini", unused_gemini, ("gemini-test",)),
    ]
    text, model = provider_test._call_with_system(
        "prompt", system="system", max_tokens=100, stage="provider-order-test"
    )
    assert (text, model) == ("ModelScope result", "modelscope/Qwen/test")
    assert unused_gemini.chat.completions.calls == 0

    limited_modelscope = _fake_client(
        error=RuntimeError("429 rate limit; please retry in 57s")
    )
    fallback_gemini = _fake_client(text="Gemini fallback")
    provider_test.providers = [
        ("modelscope", limited_modelscope, ("Qwen/test",)),
        ("gemini", fallback_gemini, ("gemini-test",)),
    ]
    with patch("src.ai.math_transcript_enhancer.time.sleep") as sleep:
        text, model = provider_test._call_with_system(
            "prompt", system="system", max_tokens=100, stage="fast-fallback-test"
        )
    assert (text, model) == ("Gemini fallback", "gemini/gemini-test")
    sleep.assert_not_called()

    original = "我们办案分析会用到十遍函数的文物课程。"
    corrected = _correct_high_confidence_asr(original, "泛函分析")
    assert corrected == "我们泛函分析会用到实变函数的文物课程。"
    assert _correct_high_confidence_asr(original, "高等数理统计") == original

    source = [
        {"start_ms": 0, "end_ms": 1000, "text": "前一段提到实变函数。"},
        {"start_ms": 1000, "end_ms": 2000, "text": corrected},
        {"start_ms": 2000, "end_ms": 3000, "text": "后一段继续讨论赋范空间。"},
    ]
    enhancer = object.__new__(MathTranscriptEnhancer)
    prompt = enhancer._prompt(
        [source[1]],
        [],
        "#### 00:00\n泛函分析与实变函数\n\n#### 00:01\n本段板书",
        course_title="泛函分析",
        source=source,
        batch_start=1,
    )
    assert "【课程名称】泛函分析" in prompt
    assert "前一段提到实变函数" in prompt
    assert "后一段继续讨论赋范空间" in prompt
    assert "最近前序板书证据" in prompt
    assert "泛函分析与实变函数" in prompt
    assert "本 CHUNK 的 PPT OCR 证据" in prompt
    prior_board = "#### 00:00\n$x+y=0$"
    assert not _has_formula_evidence("下面继续讲。", [], prior_board, 1, 2)
    assert _has_formula_evidence("刚才这个式子很重要。", [], prior_board, 1, 2)
    assert MATH_TRANSCRIPT_VERSION == 9
    fp_a = source_fingerprint("泛函分析", "总结甲", "稿", source, [], "")
    fp_b = source_fingerprint("泛函分析", "总结乙", "稿", source, [], "")
    assert fp_a != fp_b
    assert _valid_candidate(
        corrected,
        corrected,
        course_title="泛函分析",
        has_visual_evidence=False,
    )
    false_visual = (
        '<span data-visual-restored="true" style="color:#7c3aed;">'
        "〔视觉补全〕$泛函分析$</span>"
    )
    assert not _valid_candidate(
        corrected,
        false_visual,
        course_title="泛函分析",
        has_visual_evidence=True,
    )

    visual = (
        '<span data-visual-restored="true" style="color:#7c3aed;">'
        '〔视觉补全〕$d(x,y)=d(y,x)$</span>'
    )
    protected_visual, fragments = _protect_visual_fragments(
        "公式如下：" + visual + "，请继续。"
    )
    assert protected_visual == "公式如下：[[VISUAL_FORMULA_1]]，请继续。"
    assert _restore_visual_fragments(protected_visual, fragments) == (
        "公式如下：" + visual + "，请继续。"
    )
    assert _restore_visual_fragments("占位符已被删除", fragments) == ""
    final_source = "这里办案分析继续讨论度量。" + visual
    final_good = "这里泛函分析继续讨论度量。" + visual
    assert _valid_final_candidate(
        final_source,
        final_good,
        course_title="泛函分析",
    )
    assert not _valid_final_candidate(
        final_source,
        final_good.replace("d(x,y)", "d(x,z)"),
        course_title="泛函分析",
    )
    # The post-assembly pass is editorial, not a second conservative ASR copy:
    # filler and repeated starts may be removed while the mathematical point
    # and accepted visual evidence remain intact.
    editorial_source = "好，那么，呃，我们现在来看这个这个赋范空间的定义。" + visual
    editorial_good = "下面讨论赋范空间的定义。" + visual
    assert _valid_final_candidate(
        editorial_source,
        editorial_good,
        course_title="泛函分析",
    )

    second_pass = _FinalReviewEnhancer()
    reviewed, review_models, review_fallbacks = second_pass._final_review(
        [
            {
                "start_ms": 0,
                "end_ms": 180000,
                "text": "这里继续完整讨论赋饭空间的定义和性质，老师提醒大家结合实变函数复习。",
            }
        ],
        course_title="泛函分析",
        summary_reference="赋范空间是核心概念。",
    )
    assert second_pass.saw_complete_transcript
    assert second_pass.saw_summary_reference
    assert reviewed[0]["text"].startswith("这里继续完整讨论赋范空间")
    assert reviewed[0]["final_review_status"] == "ai_final_reviewed"
    assert review_models == [
        "test/summary-reference",
        "test/global-guide",
        "test/final-review",
    ]
    assert review_fallbacks == 0

    full_speech = (
        "首先欢迎大家选修泛函分析课程。作业每周一收发，平时成绩占百分之三十，"
        "期末成绩占百分之七十。大家有问题可以先发邮件，再到办公室当面交流。"
    )
    summarized = "欢迎选修泛函分析。课程包含作业、考试和答疑。"
    assert _source_coverage(full_speech, summarized) < 0.88
    assert not _valid_candidate(
        full_speech,
        summarized,
        course_title="泛函分析",
        has_visual_evidence=False,
    )
    long_source = (
        "这一部分完整保留老师关于课程内容和学习方法的详细说明，不进行任何概括或重写。"
        "办公室答疑前请先发送邮件。"
        "接下来继续完整保留老师关于泛函分析历史、实变函数基础和后续安排的详细讲述。"
    )
    missing_one_sentence = long_source.replace("办公室答疑前请先发送邮件。", "")
    assert not _clauses_preserved(long_source, missing_one_sentence)
    visual_padding = (
        summarized
        + '<span data-visual-restored="true" style="color:#7c3aed;">'
        + "〔视觉补全〕$X\\to Y$</span>"
    )
    assert not _valid_candidate(
        full_speech,
        visual_padding,
        course_title="泛函分析",
        has_visual_evidence=True,
    )
    corrected_full = full_speech.replace("泛函分析", "泛函分析学")
    assert _valid_candidate(
        full_speech,
        corrected_full,
        course_title="泛函分析",
        has_visual_evidence=False,
    )

    # Contextual ASR cleanup must still run when no visual evidence exists or
    # the model API is unavailable; it stays black (no visual-restored span).
    try:
        _UnavailableEnhancer().enhance(
            [{"start_ms": 0, "end_ms": 1000, "text": original}],
            [],
            course_title="泛函分析",
        )
    except RuntimeError as exc:
        assert "required" in str(exc) or "editorial review incomplete" in str(exc)
    else:
        raise AssertionError("an unreviewed transcript was accepted")

    faithful_markdown = "# AI 校订语音转写\n\n完整老师讲述。"
    emailer = object.__new__(Emailer)
    emailer._db = _FakeDb()
    emailer._math_enhancer = _BrokenEnhancer()
    with (
        patch(
            "src.api.emailer.load_proofread",
            return_value=(faithful_markdown, source, "proofread-v2"),
        ),
        patch("src.api.emailer.course_requires_blackboard", return_value=True),
        patch("src.api.emailer.get_blackboard", return_value=None),
        patch("src.api.emailer.load_math_transcript", return_value=None),
        patch("src.api.emailer.load_first_pass_seed", return_value=None),
    ):
        try:
            emailer._complete_transcript(
                {"sub_id": "lecture-1", "course_title": "泛函分析"}
            )
        except RuntimeError as exc:
            assert "unreviewed transcript" in str(exc)
        else:
            raise AssertionError("emailer accepted an unreviewed transcript")

    legacy = (
        "首先介绍课程要求。作业每周提交一次。"
        + "这一段旧版转写没有任何标点但仍然必须完整保存" * 20
    )
    duration = _estimated_duration(
        legacy,
        "**视频定位：01:36:00–01:45:00**",
        [{"created_sec": 7200}],
        "#### 02:30:00\n板书",
    )
    assert duration == 9000
    rebuilt = _reconstruct_timed_segments(legacy, duration)
    assert rebuilt
    assert max(len(segment["text"]) for segment in rebuilt) <= 160
    assert rebuilt[0]["start_ms"] == 0
    assert rebuilt[-1]["end_ms"] == duration * 1000
    assert all(
        left["end_ms"] < right["start_ms"]
        for left, right in zip(rebuilt, rebuilt[1:])
    )
    assert _comparison_text("".join(x["text"] for x in rebuilt)) == _comparison_text(legacy)
    fallback = _raw_transcript_markdown(rebuilt)
    assert all(segment["text"] in fallback for segment in rebuilt)
    print("math transcript v9 required-editorial-review smoke checks passed")


if __name__ == "__main__":
    main()
