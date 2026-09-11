"""True LaTeX PDF pipeline: Markdown -> Pandoc -> .tex -> Tectonic -> PDF.

No formula is rasterized.  Mathematics stays as LaTeX until Tectonic typesets the
whole document into a native vector PDF, matching the workflow used by ordinary
LaTeX/Overleaf documents while remaining fully local to GitHub Actions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from src.pdf.latex_preprocessor import compose_course_markdown


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO_ROOT / "templates" / "course_notes.tex"
_FILTER = _REPO_ROOT / "templates" / "fics_filter.lua"
_DEBUG_ROOT = Path(
    os.environ.get(
        "FICS_LATEX_DEBUG_DIR",
        str(_REPO_ROOT / "artifacts" / "latex-debug"),
    )
)


class LatexPdfError(RuntimeError):
    """Raised when Pandoc or Tectonic cannot build a course PDF."""


def _safe_name(text: str, fallback: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", str(text or ""))
    cleaned = cleaned.strip(" .-")
    return cleaned or fallback


def pdf_filename(item: dict) -> str:
    return _safe_name(
        f"{item.get('course_title','课程')}-{item.get('sub_title','课堂')}-完整课程笔记.pdf",
        "完整课程笔记.pdf",
    )


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise LatexPdfError(
            f"required document tool '{name}' is not installed; "
            "GitHub Actions must install Pandoc and Tectonic before sending mail"
        )
    return path


def _run(command: list[str], *, cwd: Path, label: str) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        tail = output[-12000:]
        raise LatexPdfError(
            f"{label} failed with exit code {proc.returncode}\n"
            f"command: {' '.join(command)}\n\n{tail}"
        )
    return output


def _persist_debug(workdir: Path, debug_id: str, error: Exception) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = _DEBUG_ROOT / f"{_safe_name(debug_id, 'lecture')}-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("notes.md", "notes.tex", "notes.log"):
        source = workdir / name
        if source.exists():
            shutil.copy2(source, target / name)
    (target / "error.txt").write_text(str(error), encoding="utf-8")
    return target


def render_markdown_pdf(
    markdown_text: str,
    *,
    title: str,
    subtitle: str = "",
    date: str = "",
    debug_id: str = "course",
) -> bytes:
    """Render already-preprocessed course Markdown as a native LaTeX PDF."""
    pandoc = _require_tool("pandoc")
    tectonic = _require_tool("tectonic")
    if not _TEMPLATE.exists() or not _FILTER.exists():
        raise LatexPdfError("LaTeX template/filter files are missing from the repository")

    with tempfile.TemporaryDirectory(prefix="fics-latex-") as tmp_raw:
        workdir = Path(tmp_raw)
        md_path = workdir / "notes.md"
        tex_path = workdir / "notes.tex"
        pdf_path = workdir / "notes.pdf"
        md_path.write_text(str(markdown_text or ""), encoding="utf-8")

        pandoc_cmd = [
            pandoc,
            str(md_path),
            "--from=markdown+fenced_divs+bracketed_spans+raw_tex+tex_math_dollars",
            "--to=latex",
            "--standalone",
            "--no-highlight",
            "--wrap=none",
            f"--template={_TEMPLATE}",
            f"--lua-filter={_FILTER}",
            "--metadata",
            f"title:{title}",
            "--metadata",
            f"subtitle:{subtitle}",
            "--metadata",
            f"date:{date}",
            "--output",
            str(tex_path),
        ]

        try:
            pandoc_output = _run(pandoc_cmd, cwd=workdir, label="Pandoc")
            tectonic_cmd = [
                tectonic,
                "-X",
                "compile",
                "--keep-logs",
                "--keep-intermediates",
                "--outdir",
                str(workdir),
                str(tex_path),
            ]
            tectonic_output = _run(
                tectonic_cmd,
                cwd=workdir,
                label="Tectonic",
            )
            if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
                raise LatexPdfError(
                    "Tectonic reported success but did not produce a valid notes.pdf"
                )
            print(
                f"[LaTeX PDF] Built native PDF: {len(markdown_text)} markdown chars -> "
                f"{pdf_path.stat().st_size} bytes; "
                f"pandoc_log={len(pandoc_output)}, tectonic_log={len(tectonic_output)}",
                flush=True,
            )
            return pdf_path.read_bytes()
        except Exception as exc:
            debug_dir = _persist_debug(workdir, debug_id, exc)
            raise LatexPdfError(
                f"{exc}\nLaTeX debug files saved under {debug_dir}"
            ) from exc


def build_course_pdf(item: dict, *, math_transcript: str = "") -> bytes:
    """Build one complete reading PDF while keeping faithful ASR separate."""
    course = str(item.get("course_title") or "课程")
    sub = str(item.get("sub_title") or "课堂")
    date = str(item.get("date") or "")
    summary = str(item.get("summary") or "").strip()
    if not summary:
        raise LatexPdfError("course summary is empty")

    markdown_text = compose_course_markdown(
        summary,
        math_transcript=str(math_transcript or ""),
    )
    return render_markdown_pdf(
        markdown_text,
        title=course,
        subtitle=sub,
        date=date,
        debug_id=str(item.get("sub_id") or f"{course}-{sub}"),
    )
