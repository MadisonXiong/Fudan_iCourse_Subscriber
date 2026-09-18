#!/usr/bin/env python3
"""Deterministic smoke test for raw-board fallback after editor rejection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.blackboard_lecture_runner import BlackboardLectureRunner


class Reporter:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class UnsafeEditor:
    def __init__(self, **kwargs):
        pass

    def edit(self, raw: str) -> tuple[str, str]:
        raise RuntimeError("simulated suspicious compression")


def main() -> None:
    runner = object.__new__(BlackboardLectureRunner)
    runner._db = object()
    runner._reporter = Reporter()
    raw = "#### 00:03\n\n老师板书：$x_n \\rightharpoonup x$"

    with patch(
        "src.pipeline.blackboard_lecture_runner.BlackboardEditor",
        UnsafeEditor,
    ):
        notes, model = runner._edit_blackboard_or_fallback("lecture-1", raw)

    assert notes.startswith("### 黑板板书整理稿")
    assert raw in notes
    assert "raw-blackboard-fallback/RuntimeError" == model
    assert any("summary and email delivery can continue" in item for item in runner._reporter.messages)
    print("Blackboard editor raw-fallback smoke test passed.")


if __name__ == "__main__":
    main()
