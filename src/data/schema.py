"""Single source of truth for the SQLite schema.

This module is imported by every Python component that creates or migrates
the database (Database, sharder, merge_db) so the column list lives in
exactly one place.

IMPORTANT: migration columns are appended at the end of ``lectures`` rather
than inserted into the middle of the CREATE TABLE definition. SQLite ALTER
TABLE also appends columns. Keeping fresh and migrated databases in the same
physical column order prevents old/new shards from silently shifting values
when a legacy tool ever relies on positional ``SELECT *`` semantics.

The expensive raw blackboard transcription is additionally mirrored into the
``meta`` table by ``blackboard_store``.  Meta lives in its own shard and is
copied by key, so the raw cache no longer depends on the evolving physical
layout of the ``lectures`` table.

frontend/js/schema.js is a **manual mirror** of these constants. When you
change SCHEMA_SQL, LECTURES_MIGRATION_COLUMNS, or PPT_PAGES_MIGRATION_COLUMNS
here, update that file too — there is no automated sync. Both run in different
processes (Python on the CI runner, JS in the browser) and have to agree on
what tables and columns exist.
"""

from __future__ import annotations


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    title TEXT,
    teacher TEXT
);
CREATE TABLE IF NOT EXISTS lectures (
    sub_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    sub_title TEXT, date TEXT,
    transcript TEXT, summary TEXT,
    processed_at TEXT, emailed_at TEXT,
    error_msg TEXT, error_count INTEGER DEFAULT 0,
    error_stage TEXT, summary_model TEXT,
    blackboard_latex TEXT, blackboard_model TEXT,
    blackboard_at TEXT,
    transcript_segments_json TEXT,
    proofread_transcript TEXT, proofread_segments_json TEXT,
    proofread_model TEXT, proofread_at TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
CREATE TABLE IF NOT EXISTS ppt_pages (
    sub_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    created_sec INTEGER NOT NULL,
    pptimgurl TEXT,
    text TEXT,
    ocr_status TEXT NOT NULL DEFAULT 'pending',
    ocr_at TEXT,
    dhash TEXT,
    PRIMARY KEY (sub_id, page_num),
    FOREIGN KEY (sub_id) REFERENCES lectures(sub_id)
);
CREATE INDEX IF NOT EXISTS idx_ppt_pages_sub_status
    ON ppt_pages(sub_id, ocr_status);
CREATE TABLE IF NOT EXISTS all_courses (
    course_id TEXT NOT NULL,
    term TEXT NOT NULL,
    title TEXT,
    teacher TEXT,
    dept TEXT,
    last_seen_at TEXT,
    PRIMARY KEY (course_id, term)
);
CREATE INDEX IF NOT EXISTS idx_all_courses_term
    ON all_courses(term);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

LECTURES_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("error_msg", "TEXT"),
    ("error_count", "INTEGER DEFAULT 0"),
    ("error_stage", "TEXT"),
    ("summary_model", "TEXT"),
    ("blackboard_latex", "TEXT"),
    ("blackboard_model", "TEXT"),
    ("blackboard_at", "TEXT"),
    ("transcript_segments_json", "TEXT"),
    ("proofread_transcript", "TEXT"),
    ("proofread_segments_json", "TEXT"),
    ("proofread_model", "TEXT"),
    ("proofread_at", "TEXT"),
]

PPT_PAGES_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("dhash", "TEXT"),
]
