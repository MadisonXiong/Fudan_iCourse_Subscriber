#!/usr/bin/env python3
"""CI smoke test for the native Pandoc + Tectonic PDF pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pdf.latex_preprocessor import compose_course_markdown
from src.pdf.latex_renderer import render_markdown_pdf
from src.pdf.latex_repairer import compiler_error_line


def _smoke_repair_diagnostic() -> None:
    log = "warning: something\nerror: notes.tex:296: Missing \\right. inserted\n"
    assert compiler_error_line(log) == 296


def main() -> int:
    _smoke_repair_diagnostic()

    summary = r'''# 泛函分析测试

设 $X$ 为非空集合，若

$$
d(x,z) \le d(x,y)+d(y,z),
$$

则满足三角不等式。

<div data-ai-note="true" style="color:#6d28d9;"><strong>AI 补充</strong><br>
这里的公式应由 Tectonic 原生排版，而不是 PNG。
</div>

**视频定位：00:03:00–00:06:00**
'''
    math_transcript = r'''# 数学增强语音转写

## 03:00–06:00

对称性：<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕$d(x,y)=d(y,x)$</span>。
'''
    markdown = compose_course_markdown(
        summary,
        math_transcript=math_transcript,
    )
    pdf = render_markdown_pdf(
        markdown,
        title="泛函分析",
        subtitle="CI 测试",
        date="2026-09-11",
        debug_id="ci-native-latex",
    )
    if not pdf.startswith(b"%PDF") or len(pdf) <= 5000:
        raise RuntimeError(f"invalid PDF output: {len(pdf)} bytes")
    print(f"native LaTeX PDF smoke test: {len(pdf)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
