"""Compatibility wrapper for the native Pandoc + Tectonic PDF pipeline.

New code lives in :mod:`src.pdf.latex_renderer`.  This module remains so older
imports continue to work without bringing back the retired ReportLab/PNG path.
"""

from src.pdf.latex_renderer import (  # noqa: F401
    LatexPdfError,
    build_course_pdf,
    pdf_filename,
    render_markdown_pdf,
)

__all__ = [
    "LatexPdfError",
    "build_course_pdf",
    "pdf_filename",
    "render_markdown_pdf",
]
