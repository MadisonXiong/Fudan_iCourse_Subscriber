"""True LaTeX PDF pipeline: Markdown -> Pandoc -> .tex -> Tectonic -> PDF.

No formula is rasterized. Mathematics stays as LaTeX until Tectonic typesets the
whole document into a native vector PDF. Compiler errors may be repaired by a
tightly-scoped LLM syntax pass. Once the document compiles, severe over-wide
display equations may receive a second, semantic-preserving layout-only pass.
Tectonic remains the final authority for both stages.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from src.pdf.latex_layout_repairer import (
    LatexLayoutRepairError,
    overflow_score,
    repair_overfull_math,
)
from src.pdf.latex_preprocessor import compose_course_markdown
from src.pdf.latex_repairer import LatexRepairError, repair_latex


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO_ROOT / "templates" / "course_notes.tex"
_FILTER = _REPO_ROOT / "templates" / "fics_filter.lua"
_DEBUG_ROOT = Path(
    os.environ.get(
        "FICS_LATEX_DEBUG_DIR",
        str(_REPO_ROOT / "artifacts" / "latex-debug"),
    )
)
_REPAIR_ENABLED = os.environ.get("FICS_LATEX_REPAIR_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
# Real lecture notes can contain several independent malformed math fragments.
_REPAIR_ATTEMPTS = max(0, int(os.environ.get("FICS_LATEX_REPAIR_ATTEMPTS", "12")))
_LAYOUT_ENABLED = os.environ.get("FICS_LATEX_LAYOUT_REPAIR_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
# Layout repair is a quality pass, not a correctness pass.  Keep the budget
# modest and always fall back to the last already-valid PDF if it cannot help.
_LAYOUT_ATTEMPTS = max(0, int(os.environ.get("FICS_LATEX_LAYOUT_REPAIR_ATTEMPTS", "4")))


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


def _run_capture(command: list[str], *, cwd: Path) -> tuple[int, str]:
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
    return int(proc.returncode), proc.stdout or ""


def _with_tex_log(output: str, workdir: Path) -> str:
    """Combine Tectonic stdout with the real TeX log diagnostics.

    Tectonic's process output can omit non-fatal ``Overfull \\hbox`` warnings;
    they are still present in ``notes.log``.  Layout repair must inspect both.
    """
    log_path = workdir / "notes.log"
    if not log_path.exists():
        return str(output or "")
    try:
        tex_log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return str(output or "")
    return str(output or "") + "\n\n--- notes.log ---\n" + tex_log


def _run(command: list[str], *, cwd: Path, label: str) -> str:
    returncode, output = _run_capture(command, cwd=cwd)
    if returncode != 0:
        tail = output[-12000:]
        raise LatexPdfError(
            f"{label} failed with exit code {returncode}\n"
            f"command: {' '.join(command)}\n\n{tail}"
        )
    return output


def _write_repair_state(
    workdir: Path,
    *,
    original_tex: str,
    current_tex: str,
    audits: list[dict],
    compiler_outputs: list[str],
) -> None:
    (workdir / "notes.original.tex").write_text(original_tex, encoding="utf-8")
    (workdir / "notes.repaired.tex").write_text(current_tex, encoding="utf-8")
    payload = {
        "repair_count": len(audits),
        "repairs": audits,
        "compiler_attempts": [
            {"attempt": i + 1, "tail": text[-12000:]}
            for i, text in enumerate(compiler_outputs)
        ],
    }
    (workdir / "repair-audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_layout_state(
    workdir: Path,
    *,
    audits: list[dict],
    compiler_outputs: list[str],
) -> None:
    payload = {
        "layout_repair_count": len(audits),
        "repairs": audits,
        "compiler_attempts": [
            {
                "attempt": i + 1,
                "overflow_score": overflow_score(text),
                "tail": text[-12000:],
            }
            for i, text in enumerate(compiler_outputs)
        ],
    }
    (workdir / "layout-audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _persist_debug(workdir: Path, debug_id: str, error: Exception, *, suffix: str = "") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tail = f"-{suffix}" if suffix else ""
    target = _DEBUG_ROOT / f"{_safe_name(debug_id, 'lecture')}-{stamp}{tail}"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "notes.md",
        "notes.tex",
        "notes.log",
        "notes.original.tex",
        "notes.repaired.tex",
        "repair-audit.json",
        "layout-audit.json",
    ):
        source = workdir / name
        if source.exists():
            shutil.copy2(source, target / name)
    (target / "error.txt").write_text(str(error), encoding="utf-8")
    return target


def _persist_successful_repair(workdir: Path, debug_id: str) -> Path:
    """Keep an audit trail when syntax and/or layout repair changed the TeX."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = _DEBUG_ROOT / f"{_safe_name(debug_id, 'lecture')}-{stamp}-repair-success"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "notes.md",
        "notes.tex",
        "notes.original.tex",
        "notes.repaired.tex",
        "repair-audit.json",
        "layout-audit.json",
        "notes.log",
    ):
        source = workdir / name
        if source.exists():
            shutil.copy2(source, target / name)
    return target


def _apply_layout_repairs(
    *,
    workdir: Path,
    tex_path: Path,
    pdf_path: Path,
    tectonic_cmd: list[str],
    current_tex: str,
    successful_output: str,
) -> tuple[str, bytes, str, list[dict], list[str]]:
    """Improve severe display-math overflows without risking PDF delivery.

    The input PDF has already compiled successfully.  Every candidate layout is
    compiled again and accepted only if the aggregate severe-overflow score
    decreases.  Any model/compile/quality failure simply returns the last valid
    PDF rather than failing the email.
    """
    if not pdf_path.exists():
        raise LatexPdfError("layout pass requires an already-valid notes.pdf")

    valid_tex = current_tex
    valid_pdf = pdf_path.read_bytes()
    valid_output = successful_output
    audits: list[dict] = []
    outputs: list[str] = [successful_output]

    if not _LAYOUT_ENABLED or _LAYOUT_ATTEMPTS <= 0:
        return valid_tex, valid_pdf, valid_output, audits, outputs

    score = overflow_score(valid_output)
    if score <= 0:
        return valid_tex, valid_pdf, valid_output, audits, outputs

    for _ in range(_LAYOUT_ATTEMPTS):
        try:
            proposal = repair_overfull_math(valid_tex, valid_output)
        except LatexLayoutRepairError as exc:
            print(f"[LaTeX Layout] stop: {exc}", flush=True)
            break

        candidate_tex = proposal.tex
        tex_path.write_text(candidate_tex, encoding="utf-8")
        if pdf_path.exists():
            pdf_path.unlink()
        log_path = workdir / "notes.log"
        if log_path.exists():
            log_path.unlink()
        returncode, candidate_output = _run_capture(tectonic_cmd, cwd=workdir)
        candidate_output = _with_tex_log(candidate_output, workdir)
        outputs.append(candidate_output)

        candidate_ok = returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size >= 1000
        if not candidate_ok:
            print(
                "[LaTeX Layout] rejected candidate because Tectonic no longer compiled; "
                "keeping previous valid PDF",
                flush=True,
            )
            tex_path.write_text(valid_tex, encoding="utf-8")
            pdf_path.write_bytes(valid_pdf)
            break

        candidate_score = overflow_score(candidate_output)
        # Require a meaningful improvement, not a line-number reshuffle that
        # merely reproduces the same overfull warning elsewhere.
        if candidate_score >= score - 1.0:
            print(
                f"[LaTeX Layout] rejected candidate: overflow score {score:.1f} -> "
                f"{candidate_score:.1f}; keeping previous valid PDF",
                flush=True,
            )
            tex_path.write_text(valid_tex, encoding="utf-8")
            pdf_path.write_bytes(valid_pdf)
            break

        valid_tex = candidate_tex
        valid_pdf = pdf_path.read_bytes()
        valid_output = candidate_output
        audits.append(proposal.audit.as_dict())
        print(
            f"[LaTeX Layout] accepted: overflow score {score:.1f} -> {candidate_score:.1f}",
            flush=True,
        )
        score = candidate_score
        if score <= 0:
            break

    tex_path.write_text(valid_tex, encoding="utf-8")
    pdf_path.write_bytes(valid_pdf)
    if audits:
        _write_layout_state(workdir, audits=audits, compiler_outputs=outputs)
    return valid_tex, valid_pdf, valid_output, audits, outputs


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
            original_tex = tex_path.read_text(encoding="utf-8")
            current_tex = original_tex
            repair_audits: list[dict] = []
            compiler_outputs: list[str] = []

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

            max_repairs = _REPAIR_ATTEMPTS if _REPAIR_ENABLED else 0
            for compile_index in range(max_repairs + 1):
                if pdf_path.exists():
                    pdf_path.unlink()
                log_path = workdir / "notes.log"
                if log_path.exists():
                    log_path.unlink()
                returncode, tectonic_output = _run_capture(tectonic_cmd, cwd=workdir)
                tectonic_output = _with_tex_log(tectonic_output, workdir)
                compiler_outputs.append(tectonic_output)

                if returncode == 0:
                    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
                        raise LatexPdfError(
                            "Tectonic reported success but did not produce a valid notes.pdf"
                        )

                    current_tex, final_pdf, final_output, layout_audits, _ = _apply_layout_repairs(
                        workdir=workdir,
                        tex_path=tex_path,
                        pdf_path=pdf_path,
                        tectonic_cmd=tectonic_cmd,
                        current_tex=current_tex,
                        successful_output=tectonic_output,
                    )

                    if repair_audits:
                        _write_repair_state(
                            workdir,
                            original_tex=original_tex,
                            current_tex=current_tex,
                            audits=repair_audits,
                            compiler_outputs=compiler_outputs,
                        )

                    if repair_audits or layout_audits:
                        audit_dir = _persist_successful_repair(workdir, debug_id)
                        print(
                            f"[LaTeX PDF] Accepted syntax_repairs={len(repair_audits)}, "
                            f"layout_repairs={len(layout_audits)}; audit={audit_dir}",
                            flush=True,
                        )

                    print(
                        f"[LaTeX PDF] Built native PDF: {len(markdown_text)} markdown chars -> "
                        f"{len(final_pdf)} bytes; pandoc_log={len(pandoc_output)}, "
                        f"tectonic_log={len(final_output)}, syntax_repairs={len(repair_audits)}, "
                        f"layout_repairs={len(layout_audits)}, overflow_score={overflow_score(final_output):.1f}",
                        flush=True,
                    )
                    return final_pdf

                if compile_index >= max_repairs:
                    tail = tectonic_output[-12000:]
                    raise LatexPdfError(
                        f"Tectonic failed with exit code {returncode} after "
                        f"{len(repair_audits)} accepted repair proposal(s)\n"
                        f"command: {' '.join(tectonic_cmd)}\n\n{tail}"
                    )

                try:
                    repair = repair_latex(current_tex, tectonic_output)
                except LatexRepairError as repair_exc:
                    tail = tectonic_output[-12000:]
                    raise LatexPdfError(
                        f"Tectonic failed with exit code {returncode}; constrained "
                        f"LaTeX repair was unavailable: {repair_exc}\n\n{tail}"
                    ) from repair_exc

                current_tex = repair.tex
                repair_audits.append(repair.audit.as_dict())
                tex_path.write_text(current_tex, encoding="utf-8")
                _write_repair_state(
                    workdir,
                    original_tex=original_tex,
                    current_tex=current_tex,
                    audits=repair_audits,
                    compiler_outputs=compiler_outputs,
                )

            raise LatexPdfError("unreachable LaTeX compilation state")
        except Exception as exc:
            debug_dir = _persist_debug(workdir, debug_id, exc)
            raise LatexPdfError(
                f"{exc}\nLaTeX debug files saved under {debug_dir}"
            ) from exc


def build_course_pdf(item: dict, *, math_transcript: str = "") -> bytes:
    """Build the existing notes PDF, appending the complete enhanced transcript."""
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
