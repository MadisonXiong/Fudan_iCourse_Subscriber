import re
import smtplib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from html import escape
from email.utils import formataddr
from urllib.parse import quote

import markdown
import requests
from PIL import Image
from pygments.formatters import HtmlFormatter

from src.runtime import config


_MD_EXTENSIONS = ["tables", "fenced_code", "nl2br", "sane_lists", "codehilite"]
_MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "guess_lang": False,
        "linenums": False,
        "css_class": "highlight",
    }
}
_PYGMENTS_CSS = HtmlFormatter(style="friendly").get_style_defs(".highlight")

_EMAIL_CSS = """\
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
    margin-top: 32px;
}
h3 {
    color: #34495e;
    margin-top: 24px;
}
h3 small {
    color: #7f8c8d;
    font-weight: normal;
}
h4 { color: #555; margin-top: 18px; }
hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 28px 0;
}
strong { color: #c0392b; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th {
    background: #f5f6fa;
    font-weight: 600;
}
tr:nth-child(even) { background: #fafafa; }
pre {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
}
code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 13px;
}
p code {
    background: #f0f0f0;
    padding: 2px 5px;
    border-radius: 3px;
}
blockquote {
    border-left: 4px solid #3498db;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8f9fa;
    color: #555;
}
ul, ol { padding-left: 24px; }
li { margin-bottom: 4px; }
.video-location {
    display: inline-block;
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 7px;
    margin: 2px 0 8px;
    font-size: 13px;
}
"""

_MIN_INLINE_HEIGHT = 13
_IMAGE_CACHE: dict[str, tuple] = {}


def _fetch_latex_image(url: str, dpi: int = 300) -> tuple:
    if url in _IMAGE_CACHE:
        return _IMAGE_CACHE[url]
    try:
        scale_factor = dpi / 96.0
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        logical_width = max(1, int(img.width / scale_factor))
        logical_height = max(1, int(img.height / scale_factor))
        result = (logical_width, logical_height, response.content)
        _IMAGE_CACHE[url] = result
        return result
    except Exception as e:
        print(f"[LaTeX Render] Image fetch failed: {e}")
        return None, None, None


def _prefetch_latex_images(urls: list[str], dpi: int = 300) -> None:
    uncached = [u for u in urls if u not in _IMAGE_CACHE]
    if not uncached:
        return
    with ThreadPoolExecutor(max_workers=min(len(uncached), 8)) as pool:
        futures = {pool.submit(_fetch_latex_image, u, dpi): u for u in uncached}
        for future in as_completed(futures):
            future.result()


def _md_to_html(md_text: str, cid_images: dict | None = None) -> str:
    """Convert Markdown to HTML and render LaTeX formulas as CID images."""
    latex_map: dict[str, str] = {}
    counter = 0

    def _stash(match):
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = match.group(0)
        return key

    def _stash_block(match):
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = "$$" + match.group(1) + "$$"
        return key

    def _stash_inline(match):
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = "$" + match.group(1) + "$"
        return key

    text = re.sub(r"\$\$(.+?)\$\$", _stash, md_text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", _stash_block, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _stash, text)
    text = re.sub(r"\\\((.+?)\\\)", _stash_inline, text)

    html = markdown.markdown(
        text,
        extensions=_MD_EXTENSIONS,
        extension_configs=_MD_EXTENSION_CONFIGS,
    )

    latex_info: dict[str, tuple[str, str, bool]] = {}
    for key, original in latex_map.items():
        is_block = original.startswith("$$")
        latex_content = original[2:-2] if is_block else original[1:-1]
        prefix = r"\dpi{300}\bg{white}" if is_block else r"\dpi{300}\bg{white}\inline"
        url = f"https://latex.codecogs.com/png.latex?{prefix}%20{quote(latex_content)}"
        latex_info[key] = (url, latex_content, is_block)

    _prefetch_latex_images([info[0] for info in latex_info.values()])

    for key, (url, latex_content, is_block) in latex_info.items():
        w, h, img_data = _fetch_latex_image(url)
        if is_block:
            if w and h:
                src = _resolve_src(url, img_data, cid_images)
                img_tag = (
                    f'<div style="text-align:center;margin:16px 0">'
                    f'<img src="{src}" alt="{escape(latex_content)}" '
                    f'width="{w}" height="{h}" '
                    f'style="width:{w}px;height:{h}px;max-width:none;'
                    f'vertical-align:middle;border:none;display:inline-block;">'
                    f'</div>'
                )
            else:
                img_tag = (
                    f'<div style="text-align:center;margin:16px 0">'
                    f'<code>{escape(latex_content)}</code></div>'
                )
        else:
            if w and h:
                if h < _MIN_INLINE_HEIGHT:
                    scale = _MIN_INLINE_HEIGHT / h
                    w = max(1, int(w * scale))
                    h = _MIN_INLINE_HEIGHT
                src = _resolve_src(url, img_data, cid_images)
                img_tag = (
                    f'<img src="{src}" alt="{escape(latex_content)}" '
                    f'width="{w}" height="{h}" '
                    f'style="width:{w}px;height:{h}px;max-width:none;'
                    f'vertical-align:-3px;border:none;margin:0 2px;">'
                )
            else:
                img_tag = f'<code>{escape(latex_content)}</code>'
        html = html.replace(key, img_tag)

    # The summary renderer emits validated **视频定位：...** labels.  Give those
    # labels a restrained visual treatment after Markdown conversion.
    html = re.sub(
        r"<p><strong>视频定位：([^<]+)</strong></p>",
        r'<div class="video-location">视频定位：\1</div>',
        html,
    )
    return html


def _resolve_src(url: str, img_data: bytes | None,
                 cid_images: dict | None) -> str:
    if cid_images is not None and img_data:
        cid = f"latex-{uuid.uuid4().hex[:12]}"
        cid_images[cid] = img_data
        return f"cid:{cid}"
    return url


def _attachment_filename(item: dict) -> str:
    raw = f"{item.get('course_title','课程')}-{item.get('sub_title','课堂')}-AI校订语音转写.md"
    # Only remove filesystem-hostile/control characters; RFC2231 below handles
    # UTF-8 Chinese filenames correctly.
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", raw).strip(" .-") or "AI校订语音转写.md"


class Emailer:
    """Send course summary emails with LaTeX images and transcript attachments."""

    def __init__(self):
        self.host = config.SMTP_HOST
        self.port = config.SMTP_PORT
        self.sender = config.SMTP_EMAIL
        self.password = config.SMTP_PASSWORD
        self.receiver = config.RECEIVER_EMAIL

    def send(self, items: list[dict]) -> bool:
        if not items:
            return True

        any_update = any(item.get("is_update") for item in items)
        courses: OrderedDict[str, list[dict]] = OrderedDict()
        for item in items:
            courses.setdefault(item["course_title"], []).append(item)

        parts = [f"{ct} ({len(lecs)})" for ct, lecs in courses.items()]
        subject = f"[FiCS] {', '.join(parts)}"
        if any_update:
            subject += "（含 PPT 识别·更新）"

        plain_sections = []
        for course_title, lectures in courses.items():
            plain_sections.append(f"{'=' * 40}")
            plain_sections.append(f"课程：{course_title}")
            plain_sections.append(f"{'=' * 40}")
            for lec in lectures:
                tag = "[更新] " if lec.get("is_update") else ""
                plain_sections.append(
                    f"\n--- {tag}{lec['sub_title']} ({lec['date']}) ---\n"
                )
                plain_sections.append(lec["summary"])
                if lec.get("transcript_attachment"):
                    plain_sections.append("\n[附件：AI 校订语音转写（含视频时间轴）]")
        plain = "\n".join(plain_sections)

        cid_images: dict[str, bytes] = {}
        course_anchors = {
            course_title: f"course-{i}"
            for i, course_title in enumerate(courses)
        }
        toc_items = [
            f'<li><a href="#{anchor}" style="color:#3498db;text-decoration:none;">'
            f"{escape(course_title)}</a></li>"
            for course_title, anchor in course_anchors.items()
        ]
        toc_html = (
            '<nav style="background:#f8f9fa;border:1px solid #e0e0e0;'
            'border-radius:6px;padding:16px 20px;margin-bottom:28px;">'
            '<strong style="color:#2c3e50;font-size:16px;">目录</strong>'
            '<ol style="margin:8px 0 0;padding-left:20px;">'
            + "\n".join(toc_items)
            + "</ol></nav>"
        )
        update_badge = (
            '<span style="background:#ff9800;color:white;padding:2px 8px;'
            'border-radius:3px;font-size:12px;margin-right:8px;'
            'vertical-align:middle;">更新</span>'
        )

        body_parts = [toc_html]
        for course_title, lectures in courses.items():
            anchor = course_anchors[course_title]
            body_parts.append(f'<h2 id="{anchor}">{escape(course_title)}</h2>')
            for lec in lectures:
                badge = update_badge if lec.get("is_update") else ""
                body_parts.append(
                    f"<h3>{badge}{escape(lec['sub_title'])} "
                    f"<small>({escape(lec['date'])})</small></h3>"
                )
                body_parts.append(
                    _md_to_html(lec["summary"], cid_images=cid_images)
                )
                if lec.get("transcript_attachment"):
                    body_parts.append(
                        '<p style="color:#64748b;font-size:13px;">'
                        '附件包含 AI 校订后的语音转写及真实视频时间轴，可用于从总结回查原视频。'
                        '</p>'
                    )
                body_parts.append("<hr>")

        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{_EMAIL_CSS}\n{_PYGMENTS_CSS}</style>"
            "</head><body>" + "\n".join(body_parts) + "</body></html>"
        )

        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = formataddr(("iCourse Subscriber", self.sender))
        msg["To"] = self.receiver

        msg_alt = MIMEMultipart("alternative")
        msg_alt.attach(MIMEText(plain, "plain", "utf-8"))
        msg_alt.attach(MIMEText(html, "html", "utf-8"))
        msg.attach(msg_alt)

        for cid, png_data in cid_images.items():
            img_part = MIMEImage(png_data, "png")
            img_part.add_header("Content-ID", f"<{cid}>")
            img_part.add_header(
                "Content-Disposition", "inline", filename=f"{cid}.png"
            )
            msg.attach(img_part)

        attachment_count = 0
        for item in items:
            transcript_md = str(item.get("transcript_attachment") or "").strip()
            if not transcript_md:
                continue
            filename = item.get("transcript_filename") or _attachment_filename(item)
            part = MIMEText(transcript_md, "markdown", "utf-8")
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=("utf-8", "", str(filename)),
            )
            msg.attach(part)
            attachment_count += 1

        if cid_images:
            print(f"[Emailer] Embedded {len(cid_images)} LaTeX images as CID")
        if attachment_count:
            print(
                f"[Emailer] Attached {attachment_count} AI-proofread timed transcript(s)"
            )

        for attempt in range(3):
            try:
                with smtplib.SMTP_SSL(self.host, self.port) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, self.receiver, msg.as_string())
                print(f"[Emailer] Sent: {subject}")
                return True
            except Exception as e:
                print(f"[Emailer] Attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        print("[Emailer] All send attempts failed.")
        return False
