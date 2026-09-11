#!/usr/bin/env python3
"""CI smoke test for the native Pandoc + Tectonic PDF pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pdf.latex_layout_repairer import (
    _deterministic_relation_reflow,
    _validate_replacement,
    overflow_score,
    overfull_issues,
)
from src.pdf.latex_preprocessor import compose_course_markdown, preprocess_markdown
from src.pdf.latex_renderer import render_markdown_pdf
from src.pdf.latex_repairer import compiler_error_line


def _smoke_repair_helpers() -> None:
    """Validate compiler-local and layout-only repair guards without API calls."""
    compiler_log = "error: notes.tex:296: Missing \\right. inserted\n"
    assert compiler_error_line(compiler_log) == 296

    overflow_log = "\n".join(
        [
            "warning: notes.tex:115: Overfull \\hbox (194.91815pt too wide) detected at line 115",
            "warning: notes.tex:746: Overfull \\hbox (278.6047pt too wide) detected at line 746",
            "warning: notes.tex:780: Underfull \\hbox (badness 3557) in paragraph at lines 779--780",
        ]
    )
    issues = overfull_issues(overflow_log, min_pt=24)
    assert [round(item.width_pt, 3) for item in issues] == [278.605, 194.918]
    assert round(overflow_score(overflow_log), 3) == 473.523

    real_tex_log = "Overfull \\hbox (194.91815pt too wide) detected at line 122"
    real_issues = overfull_issues(real_tex_log, min_pt=24)
    assert [(item.line, round(item.width_pt, 3)) for item in real_issues] == [
        (122, 194.918)
    ]

    fourier = r'''\[
\frac{1}{2\pi}\int_0^{2\pi}e^{-int}\left(\int_0^{2\pi}K(s,t)x(s)\,ds\right)dt
\Rightarrow 2\pi C_n=\int_0^{2\pi}\int_0^{2\pi}K(s,t)e^{-int}\left(\sum_{p\in\mathbb Z}a_pe^{ipt}\right)ds\,dt
=\sum_{p\in\mathbb Z}a_p\left(\int_0^{2\pi}\int_0^{2\pi}K(s,t)e^{ipt-int}ds\,dt\right)
\]'''
    reflowed = _deterministic_relation_reflow(fourier)
    assert reflowed is not None
    assert r"\begin{multlined}" in reflowed
    _validate_replacement(fourier, reflowed)

    before = r'''\[
A=B+C+D+E+F+G
\]'''
    safe_after = r'''\[
\begin{aligned}
A&=B+C+D\\
&+E+F+G
\end{aligned}
\]'''
    _validate_replacement(before, safe_after)

    changed = r'''\[
\begin{aligned}
A&=B+C+D\\
&+E+F+H
\end{aligned}
\]'''
    try:
        _validate_replacement(before, changed)
    except Exception:
        pass
    else:
        raise AssertionError("layout guard accepted changed mathematical content")


def _smoke_visual_preprocessor() -> None:
    """Cover visual-math forms seen in real model output."""
    raw = r'''inline: <span data-visual-restored="true">〔视觉补全〕$ d(x_n, x_m) < \varepsilon $</span>

block: <span data-visual-restored="true">〔视觉补全〕$$
\text{泛函分析} \\
\int_X |f|^p\,d\mu < \infty
$$</span>

bare: <span data-visual-restored="true">〔视觉补全〕\Rightarrow \forall n\in\mathbb{N}, x_n\to 0</span>
'''
    cooked = preprocess_markdown(raw)
    assert "〔视觉补全〕" not in cooked
    assert "$d(x_n, x_m) < \\varepsilon$" in cooked
    assert "::: {.visual-restored-block}" in cooked
    assert "$\\Rightarrow \\forall n\\in\\mathbb{N}, x_n\\to 0$" in cooked


def main() -> int:
    _smoke_repair_helpers()
    _smoke_visual_preprocessor()

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

对称性：<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕$ d(x,y)=d(y,x) $</span>。

<span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕$$
\int_X |f|^p\,d\mu < \infty
$$</span>

于是 <span data-visual-restored="true" style="color:#7c3aed;">〔视觉补全〕\Rightarrow \forall n\in\mathbb{N}, x_n\to 0</span>。
'''
    markdown = compose_course_markdown(
        summary,
        math_transcript=math_transcript,
    )
    assert markdown.index("数学增强语音转写") > markdown.index("视频定位")
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
