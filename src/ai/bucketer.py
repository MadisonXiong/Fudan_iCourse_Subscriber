"""Assemble LLM input by aligning timed transcript segments and PPT OCR.

Legacy ``assemble`` remains available.  New ``assemble_traceable`` builds
shorter evidence windows with stable IDs (T00, T01, ...).  Summaries cite those
IDs and the summarizer deterministically renders them as validated video ranges,
so the model never invents timestamps.
"""

from __future__ import annotations

from collections import defaultdict

from src.ai.ppt_dedup import clean_ppt_text, dedup_text_subset

BUCKET_SIZE_SEC = 600
TRACE_BUCKET_SIZE_SEC = 180


def _format_timestamp(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def assemble_bucketed(
    transcript_segments: list[dict], ppt_pages: list[dict],
) -> str:
    asr_by_bucket = defaultdict(list)
    for seg in transcript_segments or []:
        bucket = int(seg.get("start_ms", 0)) // 1000 // BUCKET_SIZE_SEC
        text = (seg.get("text") or "").strip()
        if text:
            asr_by_bucket[bucket].append(text)

    ppt_by_bucket = defaultdict(list)
    for page in ppt_pages or []:
        bucket = int(page.get("created_sec", 0)) // BUCKET_SIZE_SEC
        ppt_by_bucket[bucket].append(page)

    all_buckets = sorted(set(asr_by_bucket.keys()) | set(ppt_by_bucket.keys()))
    if not all_buckets:
        return ""

    out = []
    for b in all_buckets:
        start = b * BUCKET_SIZE_SEC
        end = (b + 1) * BUCKET_SIZE_SEC
        out.append(
            f"\n=== 时间段 {_format_timestamp(start)} – "
            f"{_format_timestamp(end)} ===\n"
        )

        asr_text = " ".join(asr_by_bucket.get(b, []))
        if asr_text:
            out.append("【音频转录】")
            out.append(asr_text)
            out.append("")

        pages = ppt_by_bucket.get(b, [])
        if pages:
            out.append("【PPT 文字识别】")
            for p in pages:
                ts = _format_timestamp(int(p["created_sec"]))
                page_num = p.get("page_num", "")
                tag = f"[页 {page_num} @ {ts}]" if page_num else f"[@ {ts}]"
                text = clean_ppt_text(p.get("text") or "").strip()
                if text:
                    out.append(f"{tag}\n{text}")
            out.append("")

    return "\n".join(out).strip()


def assemble_flat(transcript: str, ppt_pages: list[dict]) -> str:
    parts = []
    if transcript and transcript.strip():
        parts.append("【音频转录（无时间轴）】")
        parts.append(transcript.strip())
        parts.append("")
    if ppt_pages:
        parts.append("【PPT 文字识别（按出现顺序）】")
        sorted_pages = sorted(
            ppt_pages, key=lambda p: int(p.get("created_sec", 0))
        )
        for p in sorted_pages:
            ts = _format_timestamp(int(p.get("created_sec", 0)))
            page_num = p.get("page_num", "")
            tag = f"[页 {page_num} @ {ts}]" if page_num else f"[@ {ts}]"
            text = (p.get("text") or "").strip()
            if text:
                parts.append(f"{tag}\n{text}")
        parts.append("")
    return "\n".join(parts).strip()


def assemble_traceable(
    transcript: str,
    transcript_segments: list[dict] | None,
    ppt_pages: list[dict] | None,
    *,
    bucket_size_sec: int = TRACE_BUCKET_SIZE_SEC,
) -> tuple[str, str, dict[str, tuple[int, int]]]:
    """Build a source-tagged prompt and validated time-window map.

    Returns ``(prompt, mode, windows)``.  ``windows`` maps source tags such as
    ``T03`` to integer ``(start_sec, end_sec)``.  The summarizer is instructed
    to cite only these IDs; it later converts them into human-readable video
    ranges after validating every tag against this mapping.
    """
    pages = dedup_text_subset(ppt_pages or [])
    if not transcript_segments:
        # This path should only occur for a legacy cached transcript before its
        # timestamp segments have been rebuilt. Keep a safe fallback rather
        # than fabricate time provenance.
        return assemble_flat(transcript or "", pages), "flat", {}

    asr_by_bucket: dict[int, list[str]] = defaultdict(list)
    max_sec = 0
    for seg in transcript_segments:
        start_sec = int(seg.get("start_ms", 0)) // 1000
        end_sec = int(seg.get("end_ms", seg.get("start_ms", 0))) // 1000
        max_sec = max(max_sec, end_sec)
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        bucket = start_sec // bucket_size_sec
        asr_by_bucket[bucket].append(text)

    ppt_by_bucket: dict[int, list[dict]] = defaultdict(list)
    for page in pages:
        sec = int(page.get("created_sec", 0))
        max_sec = max(max_sec, sec)
        ppt_by_bucket[sec // bucket_size_sec].append(page)

    all_buckets = sorted(set(asr_by_bucket) | set(ppt_by_bucket))
    if not all_buckets:
        return "", "traceable", {}

    out: list[str] = [
        "【来源标记说明】每个 Txx 都对应一个真实视频时段。总结时只能引用这里出现的 Txx，禁止自行编造时间。",
        "",
    ]
    windows: dict[str, tuple[int, int]] = {}

    for ordinal, bucket in enumerate(all_buckets):
        tag = f"T{ordinal:02d}"
        start = bucket * bucket_size_sec
        end = min(max_sec, (bucket + 1) * bucket_size_sec)
        if end <= start:
            end = (bucket + 1) * bucket_size_sec
        windows[tag] = (start, end)
        out.append(
            f"=== [{tag}] 视频 {_format_timestamp(start)}–{_format_timestamp(end)} ==="
        )

        audio = " ".join(asr_by_bucket.get(bucket, [])).strip()
        if audio:
            out.extend(["【AI 校订语音转写】", audio, ""])

        bucket_pages = ppt_by_bucket.get(bucket, [])
        if bucket_pages:
            out.append("【PPT 文字识别】")
            for page in bucket_pages:
                sec = int(page.get("created_sec", 0))
                page_num = page.get("page_num", "")
                label = (
                    f"[页 {page_num} @ {_format_timestamp(sec)}]"
                    if page_num else f"[@ {_format_timestamp(sec)}]"
                )
                text = clean_ppt_text(page.get("text") or "").strip()
                if text:
                    out.append(f"{label}\n{text}")
            out.append("")

    return "\n".join(out).strip(), "traceable", windows


def assemble(
    transcript: str,
    transcript_segments: list[dict] | None,
    ppt_pages: list[dict] | None,
) -> tuple[str, str]:
    pages = dedup_text_subset(ppt_pages or [])
    if transcript_segments:
        return assemble_bucketed(transcript_segments, pages), "bucketed"
    return assemble_flat(transcript or "", pages), "flat"
