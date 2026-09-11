"""Compact PDF-first course email delivery.

Email bodies stay intentionally small.  Complete notes are compiled by Pandoc +
Tectonic into native vector PDFs, so mathematical expressions remain real LaTeX
until the final document is typeset; no formula PNG/CID pipeline is involved.

For Functional Analysis, the PDF also contains a separate math-enhanced timed
transcript.  It may restore ASR-lost formulas only from same-window PPT/board
evidence; the faithful AI-proofread transcript remains attached independently.
"""

from __future__ import annotations

import re
import smtplib
import time
from collections import OrderedDict
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from html import escape

from src.ai.blackboard_vision import course_requires_blackboard
from src.ai.math_transcript_enhancer import MathTranscriptEnhancer
from src.data.blackboard_store import get_blackboard
from src.data.database import Database
from src.data.math_transcript_store import (
    cache_matches,
    load_math_transcript,
    save_math_transcript,
    source_fingerprint,
)
from src.data.transcript_store import load_proofread
from src.pdf.latex_renderer import build_course_pdf, pdf_filename
from src.runtime import config


def _safe_filename(raw: str, fallback: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", raw).strip(" .-")
    return cleaned or fallback


def _attachment_filename(item: dict) -> str:
    raw = (
        f"{item.get('course_title','课程')}-"
        f"{item.get('sub_title','课堂')}-AI校订语音转写.md"
    )
    return _safe_filename(raw, "AI校订语音转写.md")


def _summary_filename(item: dict) -> str:
    raw = (
        f"{item.get('course_title','课程')}-"
        f"{item.get('sub_title','课堂')}-课程笔记.md"
    )
    return _safe_filename(raw, "课程笔记.md")


class Emailer:
    """Send small notification bodies plus PDF/traceability attachments."""

    def __init__(self):
        self.host = config.SMTP_HOST
        self.port = config.SMTP_PORT
        self.sender = config.SMTP_EMAIL
        self.password = config.SMTP_PASSWORD
        self.receiver = config.RECEIVER_EMAIL
        self._db: Database | None = None
        self._math_enhancer: MathTranscriptEnhancer | None = None

    def _database(self) -> Database:
        if self._db is None:
            self._db = Database()
        return self._db

    def _math_transcript(self, item: dict) -> str:
        """Return cached/generated evidence-constrained math transcript when applicable."""
        course_title = str(item.get("course_title") or "")
        sub_id = str(item.get("sub_id") or "")
        if not sub_id or not course_requires_blackboard(course_title):
            return ""

        db = self._database()
        proofread = load_proofread(db, sub_id)
        if not proofread:
            return ""
        proofread_markdown, proofread_segments, _ = proofread
        if not proofread_segments:
            return ""

        board_cached = get_blackboard(db, sub_id)
        raw_blackboard = board_cached[0] if board_cached else ""
        ppt_pages = db.get_done_ppt_pages(sub_id)
        fingerprint = source_fingerprint(
            course_title,
            proofread_markdown,
            proofread_segments,
            ppt_pages,
            raw_blackboard,
        )

        cached = load_math_transcript(db, sub_id)
        if cache_matches(cached, fingerprint):
            markdown = str(cached.get("markdown") or "")
            print(
                f"[Emailer] Reusing math-enhanced transcript for {sub_id} "
                f"({len(markdown)} chars).",
                flush=True,
            )
            return markdown

        try:
            if self._math_enhancer is None:
                self._math_enhancer = MathTranscriptEnhancer()
            print(
                f"[Emailer] Building evidence-constrained math transcript for {sub_id}...",
                flush=True,
            )
            result = self._math_enhancer.enhance(
                proofread_segments,
                ppt_pages,
                course_title=course_title,
                raw_blackboard=raw_blackboard,
            )
            save_math_transcript(
                db,
                sub_id,
                markdown=result.markdown,
                segments=result.segments,
                model=result.model_label,
                source_sha256=fingerprint,
            )
            print(
                f"[Emailer] Math-enhanced transcript ready: "
                f"{len(result.segments)} chunks, {len(result.markdown)} chars.",
                flush=True,
            )
            return result.markdown
        except Exception as exc:
            # Delivery must still succeed if the optional enhancement API is down.
            print(
                f"[Emailer] Math enhancement unavailable; PDF will contain the "
                f"stored course notes only: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""

    def _build_pdf(self, item: dict) -> tuple[bytes | None, str]:
        math_transcript = self._math_transcript(item)
        try:
            data = build_course_pdf(item, math_transcript=math_transcript)
            return data, pdf_filename(item)
        except Exception as exc:
            print(
                f"[Emailer] LaTeX PDF generation failed for {item.get('sub_id')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None, ""

    def send(self, items: list[dict]) -> bool:
        if not items:
            return True

        any_update = any(item.get("is_update") for item in items)
        courses: OrderedDict[str, list[dict]] = OrderedDict()
        for item in items:
            courses.setdefault(str(item["course_title"]), []).append(item)

        parts = [f"{ct} ({len(lecs)})" for ct, lecs in courses.items()]
        subject = f"[FiCS] {', '.join(parts)}"
        if any_update:
            subject += "（含 PPT 识别·更新）"

        plain_lines = [
            "完整课程内容见 LaTeX PDF 附件。",
            "PDF 由 Pandoc + Tectonic 原生排版，数学公式不再转换为 CID/PNG 图片。",
            "泛函分析 PDF 还会加入有视觉证据约束的数学增强转写。",
            "忠实 AI 校订语音转写仍作为独立附件保留，用于核对原视频。",
            "",
        ]
        html_parts = [
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;'
            'font-size:15px;line-height:1.7;color:#1f2937;max-width:720px;margin:auto;padding:20px;">',
            '<p>完整课程内容见 <strong>LaTeX PDF 附件</strong>。</p>',
            '<p>PDF 由 Pandoc + Tectonic 原生排版，数学公式不再转换为 CID/PNG 图片。'
            '泛函分析还会加入有视觉证据约束的数学增强转写；忠实 AI 校订语音转写'
            '仍作为独立附件保留，用于核对原视频。</p>',
        ]

        for course_title, lectures in courses.items():
            plain_lines.append(f"课程：{course_title}")
            html_parts.append(
                f'<h2 style="font-size:18px;color:#1e3a8a;margin:22px 0 8px;">'
                f"{escape(course_title)}</h2>"
            )
            for item in lectures:
                tag = "[更新] " if item.get("is_update") else ""
                plain_lines.append(
                    f"- {tag}{item.get('sub_title','')} ({item.get('date','')})：完整笔记见 PDF"
                )
                html_parts.append(
                    '<div style="margin:8px 0 14px;padding:10px 12px;background:#f8fafc;'
                    'border:1px solid #e2e8f0;border-radius:6px;">'
                    f"<strong>{escape(tag + str(item.get('sub_title') or '课堂'))}</strong> "
                    f"<span style=\"color:#64748b\">({escape(str(item.get('date') or ''))})</span><br>"
                    '<span style="color:#475569">完整笔记与原生 LaTeX 公式见 PDF 附件。</span>'
                    "</div>"
                )
        html_parts.append("</div>")
        plain = "\n".join(plain_lines)
        html = "".join(html_parts)

        # Outer multipart/mixed is intentional: normal attachments must not live
        # inside multipart/related, which some webmail clients hide.
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = formataddr(("iCourse Subscriber", self.sender))
        msg["To"] = self.receiver

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain, "plain", "utf-8"))
        alt.attach(MIMEText(html, "html", "utf-8"))
        msg.attach(alt)

        pdf_count = 0
        transcript_count = 0
        markdown_fallback_count = 0

        for item in items:
            pdf_data, filename = self._build_pdf(item)
            if pdf_data:
                part = MIMEApplication(pdf_data, _subtype="pdf")
                part.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=("utf-8", "", filename),
                )
                msg.attach(part)
                pdf_count += 1
            else:
                # Never lose the stored notes if Pandoc/Tectonic encounters an
                # unexpected LaTeX edge case.  Debug .tex/.log files are saved
                # separately by the renderer and uploaded by the workflow.
                summary = str(item.get("summary") or "").strip()
                if summary:
                    fallback = MIMEText(summary, "plain", "utf-8")
                    fallback.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=("utf-8", "", _summary_filename(item)),
                    )
                    msg.attach(fallback)
                    markdown_fallback_count += 1

            transcript_md = str(item.get("transcript_attachment") or "").strip()
            if transcript_md:
                transcript_name = (
                    item.get("transcript_filename") or _attachment_filename(item)
                )
                transcript_part = MIMEText(transcript_md, "plain", "utf-8")
                transcript_part.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=("utf-8", "", str(transcript_name)),
                )
                msg.attach(transcript_part)
                transcript_count += 1

        print(
            f"[Emailer] Attachments: latex_pdf={pdf_count}, "
            f"faithful_transcript={transcript_count}, "
            f"markdown_fallback={markdown_fallback_count}; CID images=0",
            flush=True,
        )

        for attempt in range(3):
            try:
                with smtplib.SMTP_SSL(self.host, self.port) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, self.receiver, msg.as_string())
                print(f"[Emailer] Sent: {subject}")
                return True
            except Exception as exc:
                if attempt >= 2:
                    print(
                        f"[Emailer] Failed after 3 attempts: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    return False
                wait = 2 ** attempt
                print(
                    f"[Emailer] Send failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {wait}s...",
                    flush=True,
                )
                time.sleep(wait)
        return False
