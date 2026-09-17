#!/usr/bin/env python3
"""Deterministic smoke test for resumable, failure-tolerant proofreading."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.transcript_proofreader import TranscriptProofreader
from src.data.transcript_store import (
    clear_proofread_checkpoint,
    load_proofread_checkpoint,
)


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
    assert resumed.calls == 0, "checkpointed windows should not call the API again"
    assert [item["proofread_status"] for item in resumed_result.segments] == [
        "ai_proofread",
        "raw_fallback",
    ]

    clear_proofread_checkpoint(db, "lecture-1")
    assert load_proofread_checkpoint(db, "lecture-1") is None
    print("Transcript proofreading checkpoint smoke test passed.")


if __name__ == "__main__":
    main()
