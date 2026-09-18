#!/usr/bin/env python3
"""Offline smoke tests for traceable-summary repair and graceful fallback."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.summarizer import (
    Summarizer,
    render_traceable_summary,
    render_traceable_summary_lenient,
)


WINDOWS = {
    "T00": (0, 179),
    "T01": (180, 359),
}


class OfflineSummarizer(Summarizer):
    def __init__(self, *, repair_fails: bool):
        self.providers = [{"name": "fake", "models": ["fake-model"]}]
        self._clients = {"fake": object()}
        self.repair_fails = repair_fails
        self.repair_calls = 0

    def _call_llm(
        self,
        client,
        model: str,
        title: str,
        content: str,
        *,
        traceable: bool = False,
    ) -> str:
        assert traceable
        return (
            "### 有可靠定位的部分\n"
            "〔VIDEO:T00〕\n\n"
            "这部分正文必须保留。\n\n"
            "#### 模型漏掉定位的部分\n\n"
            "这部分正文也必须保留。"
        )

    def _repair_traceability(
        self,
        client,
        model: str,
        title: str,
        source_content: str,
        summary: str,
        validation_error: Exception,
    ) -> str:
        self.repair_calls += 1
        assert "heading lacks video provenance" in str(validation_error)
        if self.repair_fails:
            raise RuntimeError("simulated repair outage")
        return summary.replace(
            "#### 模型漏掉定位的部分\n\n",
            "#### 模型漏掉定位的部分\n〔VIDEO:T01〕\n\n",
        )


def main() -> None:
    strict = render_traceable_summary(
        "### 第一部分\n〔VIDEO:T00-T01〕\n\n正文。",
        WINDOWS,
    )
    assert "**视频定位：00:00–05:59**" in strict
    assert "〔VIDEO:" not in strict

    degraded = render_traceable_summary_lenient(
        "### 第一部分\n〔VIDEO:T00〕\n\n正文一。\n\n"
        "#### 第二部分\n\n正文二。\n\n"
        "##### 第三部分\n〔VIDEO:T99〕\n\n正文三。",
        WINDOWS,
    )
    assert "**视频定位：00:00–02:59**" in degraded
    assert "部分标题缺少可靠的视频定位" in degraded
    assert degraded.count("本部分缺少可靠的视频定位") == 2
    assert "正文一。" in degraded and "正文二。" in degraded and "正文三。" in degraded
    assert "〔VIDEO:" not in degraded

    repaired = OfflineSummarizer(repair_fails=False)
    repaired_text, repaired_model = repaired.summarize(
        "测试课程",
        "=== [T00] 视频 00:00–02:59 ===\n材料一\n"
        "=== [T01] 视频 03:00–05:59 ===\n材料二",
        video_windows=WINDOWS,
    )
    assert repaired.repair_calls == 1
    assert repaired_model == "fake/fake-model+provenance-repair"
    assert "**视频定位：03:00–05:59**" in repaired_text
    assert "缺少可靠的视频定位" not in repaired_text

    fallback = OfflineSummarizer(repair_fails=True)
    fallback_text, fallback_model = fallback.summarize(
        "测试课程",
        "=== [T00] 视频 00:00–02:59 ===\n材料一\n"
        "=== [T01] 视频 03:00–05:59 ===\n材料二",
        video_windows=WINDOWS,
    )
    assert fallback.repair_calls == 1
    assert fallback_model == "fake/fake-model+provenance-fallback"
    assert "模型漏掉定位的部分" in fallback_text
    assert "这部分正文也必须保留。" in fallback_text
    assert "部分标题缺少可靠的视频定位" in fallback_text
    assert "〔VIDEO:" not in fallback_text

    print("Traceable summary repair and fallback smoke test passed.")


if __name__ == "__main__":
    main()
