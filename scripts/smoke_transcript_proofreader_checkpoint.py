#!/usr/bin/env python3
"""Deterministic smoke test for resumable, failure-tolerant proofreading."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.transcript_proofreader import TranscriptProofreader
from src.data.transcript_store import (
    clear_proofread_checkpoint,
    load_proofread_checkpoint,
)


class APITimeoutError(Exception):
    pass


class RateLimitError(Exception):
    status_code = 429


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        choice = SimpleNamespace(
            message=SimpleNamespace(content=outcome),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


def circuit_breaker_proofreader(providers) -> TranscriptProofreader:
    proofreader = object.__new__(TranscriptProofreader)
    proofreader.providers = providers
    proofreader._model_failure_counts = {}
    proofreader._disabled_models = {}
    proofreader._circuit_failures = 2
    return proofreader


class MemoryDB:
    def __init__(self):
        self.meta: dict[str, str] = {}

    def read_meta(self, key: str) -> str | None:
        return self.meta.get(key)

    def write_meta(self, key: str, value: str) -> None:
        self.meta[key] = value


class ScriptedProofreader(TranscriptProofreader):
    def __init__(self, *, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def _call(self, prompt: str, *, system_prompt: str = "") -> tuple[str, str]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated provider interruption")
        marker = "【当前时段原始 ASR】\n"
        current = prompt.split(marker, 1)[1].split("\n\n【后文", 1)[0]
        return current, "fake/current-model"


def main() -> None:
    db = MemoryDB()
    segments = [
        {"start_ms": 0, "end_ms": 10_000, "text": "第一段课堂语音。"},
        {"start_ms": 190_000, "end_ms": 200_000, "text": "第二段课堂语音。"},
    ]

    degraded = ScriptedProofreader(fail_on_call=2)
    result = degraded.proofread(
        segments,
        [],
        checkpoint_db=db,
        checkpoint_sub_id="lecture-1",
    )
    assert degraded.calls == 2
    assert len(result.segments) == 2
    assert [item["proofread_status"] for item in result.segments] == [
        "ai_proofread",
        "raw_fallback",
    ]
    assert "raw-asr/provider-unavailable" in result.model_label
    assert result.model_label.endswith("+raw-fallback[1]")

    checkpoint = load_proofread_checkpoint(db, "lecture-1")
    assert checkpoint is not None
    assert len(checkpoint["chunks"]) == 2

    resumed = ScriptedProofreader()
    resumed_result = resumed.proofread(
        segments,
        [],
        checkpoint_db=db,
        checkpoint_sub_id="lecture-1",
    )
    assert resumed.calls == 1, "raw fallback windows must be retried"
    assert [item["proofread_status"] for item in resumed_result.segments] == [
        "ai_proofread",
        "ai_proofread",
    ]
    assert "raw-asr/provider-unavailable" not in resumed_result.model_label
    assert "raw-fallback" not in resumed_result.model_label

    clear_proofread_checkpoint(db, "lecture-1")
    assert load_proofread_checkpoint(db, "lecture-1") is None

    slow = FakeClient([
        APITimeoutError("Request timed out."),
        APITimeoutError("Request timed out."),
    ])
    healthy = FakeClient(["校订一", "校订二", "校订三"])
    breaker = circuit_breaker_proofreader([
        ("modelscope", slow, ("slow-model",)),
        ("fallback", healthy, ("healthy-model",)),
    ])
    for _ in range(3):
        text, model = breaker._call("测试提示")
        assert text.startswith("校订")
        assert model == "fallback/healthy-model"
    assert slow.chat.completions.calls == 2
    assert healthy.chat.completions.calls == 3
    assert "modelscope/slow-model" in breaker._disabled_models

    limited = FakeClient([RateLimitError("429 quota exhausted")])
    backup = FakeClient(["回退一", "回退二"])
    quota_breaker = circuit_breaker_proofreader([
        ("gemini", limited, ("limited-model",)),
        ("fallback", backup, ("healthy-model",)),
    ])
    quota_breaker._call("第一次")
    quota_breaker._call("第二次")
    assert limited.chat.completions.calls == 1
    assert backup.chat.completions.calls == 2
    assert "gemini/limited-model" in quota_breaker._disabled_models

    print("Transcript proofreading checkpoint and circuit-breaker smoke tests passed.")


if __name__ == "__main__":
    main()
