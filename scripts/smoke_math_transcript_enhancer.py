"""Offline smoke checks for course-aware math transcript enhancement."""

from src.ai.math_transcript_enhancer import (
    MathTranscriptEnhancer,
    _clauses_preserved,
    _correct_high_confidence_asr,
    _source_coverage,
    _valid_candidate,
)
from src.data.math_transcript_store import (
    MATH_TRANSCRIPT_VERSION,
    source_fingerprint,
)


class _UnavailableEnhancer(MathTranscriptEnhancer):
    def __init__(self):
        pass

    def _call(self, prompt: str):
        raise RuntimeError("offline smoke test")


def main() -> None:
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
    assert "前一时段黑板上下文" in prompt
    assert "泛函分析与实变函数" in prompt
    assert "本 CHUNK 的 PPT OCR 证据" in prompt
    assert MATH_TRANSCRIPT_VERSION == 4
    fp_a = source_fingerprint("泛函分析", "稿", source, [], "")
    fp_b = source_fingerprint("高等数理统计", "稿", source, [], "")
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
    result = _UnavailableEnhancer().enhance(
        [{"start_ms": 0, "end_ms": 1000, "text": original}],
        [],
        course_title="泛函分析",
    )
    assert "泛函分析" in result.markdown
    assert "实变函数的文物课程" in result.markdown
    assert "办案分析" not in result.markdown
    assert "十遍函数" not in result.markdown
    assert "data-visual-restored" not in result.markdown
    assert result.model_label.startswith("math-transcript-v4/")
    print("math transcript v4 smoke checks passed")


if __name__ == "__main__":
    main()
