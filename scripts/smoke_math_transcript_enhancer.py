"""Offline smoke checks for course-aware math transcript enhancement."""

from src.ai.math_transcript_enhancer import (
    MathTranscriptEnhancer,
    _correct_high_confidence_asr,
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
        "",
        course_title="泛函分析",
        source=source,
        batch_start=1,
    )
    assert "【课程名称】泛函分析" in prompt
    assert "前一段提到实变函数" in prompt
    assert "后一段继续讨论赋范空间" in prompt
    assert "本 CHUNK 的 PPT OCR 证据" in prompt
    assert MATH_TRANSCRIPT_VERSION == 3
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
    assert result.model_label.startswith("math-transcript-v3/")
    print("math transcript v3 smoke checks passed")


if __name__ == "__main__":
    main()
