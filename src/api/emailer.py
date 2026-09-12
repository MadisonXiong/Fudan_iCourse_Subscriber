"""Compact PDF-first course email delivery.

Email bodies stay intentionally small.  Complete notes are compiled by Pandoc +
Tectonic into native vector PDFs, so mathematical expressions remain real LaTeX
until the final document is typeset; no formula PNG/CID pipeline is involved.

Every PDF appends the complete AI-proofread classroom transcript after the
existing notes. Functional Analysis additionally receives contextual ASR
correction and evidence-constrained formula restoration. The conservative
proofread transcript remains persisted in the database for audit, but email
delivery exposes one final PDF rather than implementation-format Markdown.
"""

from __future__ import annotations

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
from src.pdf.latex_renderer import LatexPdfError, build_course_pdf, pdf_filename
from src.runtime import config

class Emailer:
    """Send small notification bodies plus complete PDF attachments."""

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

    def _complete_transcript(self, item: dict) -> str:
        """Return the mandatory complete transcript appendix for this PDF."""
        course_title = str(item.get("course_title") or "")
        sub_id = str(item.get("sub_id") or "")
        stored_markdown = str(item.get("transcript_attachment") or "").strip()
        if not sub_id:
            return stored_markdown

        db = self._database()
        proofread = load_proofread(db, sub_id)
        if not proofread:
            return stored_markdown
        proofread_markdown, proofread_segments, _ = proofread
        proofread_markdown = str(proofread_markdown or stored_markdown).strip()
        if not course_requires_blackboard(course_title):
            return proofread_markdown
        if not proofread_segments:
            return proofread_markdown

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
            # The transcript appendix is mandatory. If enhancement is down,
            # preserve the complete AI-proofread text rather than omitting it.
            print(
                f"[Emailer] Math enhancement unavailable; preserving the complete "
                f"AI-proofread transcript appendix: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return proofread_markdown

    def _build_pdf(self, item: dict) -> tuple[bytes, str]:
        complete_transcript = self._complete_transcript(item)
        try:
            data = build_course_pdf(item, math_transcript=complete_transcript)
            return data, pdf_filename(item)
        except Exception as exc:
            print(
                f"[Emailer] LaTeX PDF generation failed for {item.get('sub_id')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise LatexPdfError(
                "refusing to send Markdown fallback: the required complete PDF "
                f"could not be built for {item.get('sub_id')}"
            ) from exc

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
            "PDF 在原课程笔记之后附有完整课堂语音转写；泛函分析还会校正 ASR 并补全有证据的公式。",
            "忠实 AI 校订底稿保存在系统中，邮件仅发送最终 PDF。",
            "",
        ]
        html_parts = [
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;'
            'font-size:15px;line-height:1.7;color:#1f2937;max-width:720px;margin:auto;padding:20px;">',
            '<p>完整课程内容见 <strong>LaTeX PDF 附件</strong>。</p>',
            '<p>PDF 由 Pandoc + Tectonic 原生排版，数学公式不再转换为 CID/PNG 图片。'
            'PDF 在原课程笔记之后附有完整课堂语音转写；泛函分析还会校正 ASR 并补全有证据的公式；'
            '忠实 AI 校订底稿保存在系统中，邮件仅发送最终 PDF。</p>',
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
                    '<span style="color:#475569">完整笔记与文末增强语音转写见 PDF 附件。</span>'
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

        # Build every PDF before opening SMTP. A compiler failure is a failed
        # delivery, never a green run that silently substitutes .md files.
        try:
            rendered = [self._build_pdf(item) for item in items]
        except LatexPdfError as exc:
            print(f"[Emailer] Delivery aborted: {exc}", flush=True)
            return False

        for pdf_data, filename in rendered:
            part = MIMEApplication(pdf_data, _subtype="pdf")
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=("utf-8", "", filename),
            )
            msg.attach(part)

        print(
            f"[Emailer] Attachments: latex_pdf={len(rendered)}, markdown=0; CID images=0",
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
