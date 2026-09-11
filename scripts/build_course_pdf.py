#!/usr/bin/env python3
"""Build a standalone FiCS course PDF through Pandoc + Tectonic.

This is primarily a local/CI debugging entry point.  Normal delivery calls the
same renderer through Emailer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pdf.latex_preprocessor import compose_course_markdown
from src.pdf.latex_renderer import render_markdown_pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Markdown course notes")
    parser.add_argument("output", type=Path, help="Destination PDF")
    parser.add_argument("--title", default="课程笔记")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--math-transcript", type=Path)
    parser.add_argument("--faithful-transcript", type=Path)
    args = parser.parse_args()

    summary = args.input.read_text(encoding="utf-8")
    math_transcript = ""
    if args.math_transcript:
        math_transcript = args.math_transcript.read_text(encoding="utf-8")
    faithful_transcript = ""
    if args.faithful_transcript:
        faithful_transcript = args.faithful_transcript.read_text(encoding="utf-8")

    md = compose_course_markdown(
        summary,
        faithful_transcript=faithful_transcript,
        math_transcript=math_transcript,
    )
    pdf = render_markdown_pdf(
        md,
        title=args.title,
        subtitle=args.subtitle,
        date=args.date,
        debug_id=args.input.stem,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf)
    print(f"Wrote {args.output} ({len(pdf)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
