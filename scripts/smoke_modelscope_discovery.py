#!/usr/bin/env python3
"""Offline smoke test for ModelScope live-model filtering."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime import config


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "data": [
                    {"id": "Qwen/Qwen3.8-Flash-Next"},
                    {"id": "OpenGVLab/InternVL3_5-241B-A28B"},
                ]
            }
        ).encode("utf-8")


def main() -> None:
    base_url = "https://api-inference.modelscope.cn/v1/"
    config._MODELSCOPE_DISCOVERY_CACHE.clear()
    with patch.object(config.urllib.request, "urlopen", return_value=FakeResponse()):
        selected = config.filter_modelscope_models(
            [
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "Qwen/Qwen3.8-Flash-Next",
            ],
            base_url,
        )
    assert selected == ["Qwen/Qwen3.8-Flash-Next"]
    print("ModelScope live-model discovery smoke test passed.")


if __name__ == "__main__":
    main()
