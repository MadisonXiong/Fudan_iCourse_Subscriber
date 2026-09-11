"""PDF-first renderer for course notes with LaTeX formula support."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_DPI = int(os.environ.get("FICS_PDF_LATEX_DPI", "160"))
_CACHE: dict[tuple[str, bool], tuple[int, int, bytes] | None] = {}
_FORMULA_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)|\\\((.+?)\\\)",
    re.DOTALL,
)
_AI_NOTE_RE = re.compile(r'<div\s+data-ai-note="true"[^>]*>(.*?)</div>', re.I | re.DOTALL)
_VIS_OPEN_RE = re.compile(
    r'<span\s+data-visual-restored="true"[^>]*>\s*〔视觉补全〕?', re.I
)


def _font() -> None:
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def _render_formula(latex: str, block: bool) -> tuple[int, int, bytes] | None:
    key = (latex, block)
    if key in _CACHE:
        return _CACHE[key]
    prefix = rf"\dpi{{{_DPI}}}\bg{{white}}" + ("" if block else r"\inline")
    url = f"https://latex.codecogs.com/png.latex?{prefix}%20{quote(latex)}"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        with PILImage.open(BytesIO(r.content)) as img:
            w, h = img.size
            compact = img.convert("L")
            out = BytesIO()
            compact.save(out, format="PNG", optimize=True)
        _CACHE[key] = (int(w), int(h), out.getvalue())
    except Exception as exc:
        print(f"[PDF] LaTeX render failed; source kept: {exc}", flush=True)
        _CACHE[key] = None
    return _CACHE[key]


def _all_formulas(text: str) -> list[tuple[str, bool]]:
    found: list[tuple[str, bool]] = []
    seen = set()
    for m in _FORMULA_RE.finditer(text or ""):
        if m.group(1) is not None:
            item = (m.group(1).strip(), True)
        elif m.group(2) is not None:
            item = (m.group(2).strip(), True)
        elif m.group(3) is not None:
            item = (m.group(3).strip(), False)
        else:
            item = (m.group(4).strip(), False)
        if item[0] and item not in seen:
            seen.add(item)
            found.append(item)
    return found


def _prefetch(text: str) -> None:
    todo = [x for x in _all_formulas(text) if x not in _CACHE]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
        futures = [pool.submit(_render_formula, latex, block) for latex, block in todo]
        for future in as_completed(futures):
            future.result()


def _safe_name(text: str, fallback: str) -> str:
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", text).strip(" .-")
    return text or fallback


def pdf_filename(item: dict) -> str:
    return _safe_name(
        f"{item.get('course_title','课程')}-{item.get('sub_title','课堂')}-完整课程笔记.pdf",
        "完整课程笔记.pdf",
    )


class _Doc:
    def __init__(self, temp_dir: str):
        _font()
        self.temp_dir = temp_dir
        self.files: dict[tuple[str, bool], tuple[str, float, float] | None] = {}
        base = dict(
            fontName="STSong-Light",
            wordWrap="CJK",
            textColor=colors.HexColor("#1f2937"),
        )
        self.body = ParagraphStyle(
            "body", fontSize=10.5, leading=17, spaceAfter=4, **base
        )
        self.small = ParagraphStyle(
            "small",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            fontName="STSong-Light",
            wordWrap="CJK",
        )
        self.h1 = ParagraphStyle(
            "h1", fontSize=20, leading=27, spaceAfter=10, **base
        )
        self.h2 = ParagraphStyle(
            "h2",
            fontSize=15,
            leading=22,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1e3a8a"),
            fontName="STSong-Light",
            wordWrap="CJK",
        )
        self.h3 = ParagraphStyle(
            "h3", fontSize=12.5, leading=19, spaceBefore=7, spaceAfter=4, **base
        )
        self.quote = ParagraphStyle(
            "quote",
            parent=self.body,
            leftIndent=10,
            rightIndent=6,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=.5,
            borderPadding=5,
        )

    def _file(self, latex: str, block: bool) -> tuple[str, float, float] | None:
        key = (latex, block)
        if key in self.files:
            return self.files[key]
        rendered = _render_formula(latex, block)
        if not rendered:
            self.files[key] = None
            return None
        w, h, data = rendered
        name = hashlib.sha1((str(block) + latex).encode()).hexdigest()[:16] + ".png"
        path = os.path.join(self.temp_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        self.files[key] = (path, w * 72 / _DPI, h * 72 / _DPI)
        return self.files[key]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("**", "").replace("__", "")
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</?(?:strong|em)[^>]*>", "", text, flags=re.I)
        return text

    def inline(self, text: str) -> str:
        text = self._clean(text)
        marker = "__VISUAL_RESTORE_LABEL__"
        text = _VIS_OPEN_RE.sub(marker, text).replace("</span>", "")
        out, pos = [], 0
        for m in _FORMULA_RE.finditer(text):
            out.append(escape(text[pos:m.start()]).replace("\n", "<br/>"))
            latex = next(x for x in m.groups() if x is not None).strip()
            img = self._file(latex, False)
            if img:
                path, w, h = img
                target_h = min(max(h, 9), 18)
                target_w = min(w * target_h / max(h, 1), 330)
                out.append(
                    f'<img src="{escape(path)}" width="{target_w:.1f}" '
                    f'height="{target_h:.1f}" valign="middle"/>'
                )
            else:
                out.append(f'<font face="Courier" size="8">{escape(latex)}</font>')
            pos = m.end()
        out.append(escape(text[pos:]).replace("\n", "<br/>"))
        return "".join(out).replace(
            marker, '<font color="#7c3aed">〔视觉补全〕</font>'
        )

    def para(self, text: str, style=None):
        return Paragraph(self.inline(text), style or self.body)

    def formula(self, latex: str):
        img = self._file(latex.strip(), True)
        if not img:
            return Paragraph(
                f'<font face="Courier" size="8">{escape(latex.strip())}</font>',
                self.body,
            )
        path, w, h = img
        max_w = A4[0] - 40 * mm
        if w > max_w:
            s = max_w / w
            w, h = w * s, h * s
        if h > 90 * mm:
            s = 90 * mm / h
            w, h = w * s, h * s
        flow = Image(path, width=w, height=h)
        flow.hAlign = "CENTER"
        return flow

    def ai_note(self, raw: str):
        m = _AI_NOTE_RE.search(raw)
        inner = m.group(1) if m else raw
        inner = re.sub(
            r"<strong[^>]*>\s*AI\s*补充\s*</strong>\s*<br\s*/?>",
            "",
            inner,
            flags=re.I,
        )
        p = Paragraph(
            '<font color="#6d28d9">AI 补充</font><br/>' + self.inline(inner),
            self.body,
        )
        t = Table([[p]], colWidths=[A4[0] - 40 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f3ff")),
                    ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#c4b5fd")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return t

    def flows(self, md: str) -> list:
        lines = (md or "").replace("\r\n", "\n").split("\n")
        story, buf = [], []
        i = 0

        def flush():
            if buf:
                text = " ".join(x.strip() for x in buf if x.strip()).strip()
                buf.clear()
                if text:
                    story.append(self.para(text))

        while i < len(lines):
            s = lines[i].strip()
            if not s:
                flush()
                story.append(Spacer(1, 2))
                i += 1
                continue
            if s.startswith("<div") and 'data-ai-note="true"' in s:
                flush()
                block = [lines[i]]
                while "</div>" not in block[-1] and i + 1 < len(lines):
                    i += 1
                    block.append(lines[i])
                story.append(self.ai_note("\n".join(block)))
                i += 1
                continue
            if s.startswith("$$") or s.startswith(r"\["):
                flush()
                token = "$$" if s.startswith("$$") else r"\]"
                content = s[2:]
                if content.endswith(token) and content != token:
                    content = content[:-len(token)]
                else:
                    parts = [content]
                    i += 1
                    while i < len(lines):
                        cur = lines[i]
                        if cur.strip().endswith(token):
                            parts.append(cur[:cur.rfind(token)])
                            break
                        parts.append(cur)
                        i += 1
                    content = "\n".join(parts)
                story += [self.formula(content), Spacer(1, 4)]
                i += 1
                continue
            if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
                flush()
                story.append(
                    HRFlowable(
                        width="100%", thickness=.5, color=colors.HexColor("#cbd5e1")
                    )
                )
                i += 1
                continue
            heading = re.match(r"^(#{1,4})\s+(.+)$", s)
            if heading:
                flush()
                level = len(heading.group(1))
                style = self.h1 if level == 1 else self.h2 if level == 2 else self.h3
                story.append(self.para(heading.group(2), style))
                i += 1
                continue
            if s.startswith(">"):
                flush()
                story.append(self.para(s.lstrip("> "), self.quote))
                i += 1
                continue
            li = re.match(r"^([-*+]|\d+\.)\s+(.+)$", s)
            if li:
                flush()
                story.append(
                    self.para(
                        f"{li.group(1)} {li.group(2)}",
                        ParagraphStyle(
                            "li", parent=self.body, leftIndent=10, firstLineIndent=-7
                        ),
                    )
                )
                i += 1
                continue
            buf.append(lines[i])
            i += 1
        flush()
        return story


def build_course_pdf(item: dict, *, math_transcript: str = "") -> bytes:
    course = str(item.get("course_title") or "课程")
    sub = str(item.get("sub_title") or "课堂")
    date = str(item.get("date") or "")
    summary = str(item.get("summary") or "").strip()
    transcript = str(math_transcript or item.get("transcript_attachment") or "").strip()

    _prefetch(summary + "\n" + transcript)
    buf = BytesIO()
    with tempfile.TemporaryDirectory(prefix="fics-pdf-") as tmp:
        renderer = _Doc(tmp)
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"{course} - {sub}",
            author="iCourse Subscriber",
        )
        story = [
            Paragraph(course, renderer.h1),
            Paragraph(sub, renderer.h2),
            Paragraph(date, renderer.small),
            Spacer(1, 6),
            HRFlowable(
                width="100%", thickness=.6, color=colors.HexColor("#94a3b8")
            ),
            Paragraph("课程笔记", renderer.h2),
        ]
        story += renderer.flows(summary)
        if transcript:
            story += [
                Spacer(1, 8),
                HRFlowable(
                    width="100%", thickness=.6, color=colors.HexColor("#94a3b8")
                ),
                Paragraph("语音转写", renderer.h2),
            ] + renderer.flows(transcript)

        def footer(canvas, d):
            canvas.saveState()
            canvas.setFont("STSong-Light", 8)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawString(18 * mm, 9 * mm, course[:36])
            canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, str(d.page))
            canvas.restoreState()

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
