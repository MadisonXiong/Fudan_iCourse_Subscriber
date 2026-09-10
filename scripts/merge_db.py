#!/usr/bin/env python3
"""Merge local DB into remote DB (additive-only).

Used at deploy time to safely combine results from concurrent workflow runs.
For each lecture row, fields only progress forward (null -> non-null).
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import (
    LECTURES_MIGRATION_COLUMNS,
    PPT_PAGES_MIGRATION_COLUMNS,
    SCHEMA_SQL,
)


def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    existing_lectures = {r[1] for r in conn.execute("PRAGMA table_info(lectures)")}
    for col, typedef in LECTURES_MIGRATION_COLUMNS:
        if col not in existing_lectures:
            conn.execute(f"ALTER TABLE lectures ADD COLUMN {col} {typedef}")

    existing_ppt = {r[1] for r in conn.execute("PRAGMA table_info(ppt_pages)")}
    for col, typedef in PPT_PAGES_MIGRATION_COLUMNS:
        if col not in existing_ppt:
            conn.execute(f"ALTER TABLE ppt_pages ADD COLUMN {col} {typedef}")


def merge(local_path: str, remote_path: str):
    conn = sqlite3.connect(remote_path)
    _ensure_schema(conn)
    conn.execute("ATTACH DATABASE ? AS local", (local_path,))
    _ensure_schema(conn)

    local_columns = {r[1] for r in conn.execute("PRAGMA local.table_info(lectures)")}
    for col, typedef in LECTURES_MIGRATION_COLUMNS:
        if col not in local_columns:
            conn.execute(f"ALTER TABLE local.lectures ADD COLUMN {col} {typedef}")

    try:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO main.courses (course_id, title, teacher)
                SELECT course_id, title, teacher FROM local.courses
            """)

            conn.execute("""
                INSERT OR IGNORE INTO main.lectures
                    (sub_id, course_id, sub_title, date,
                     transcript, transcript_segments_json,
                     proofread_transcript, proofread_segments_json,
                     proofread_model, proofread_at,
                     summary, processed_at, emailed_at,
                     error_msg, error_count, error_stage,
                     summary_model, blackboard_latex, blackboard_model, blackboard_at)
                SELECT sub_id, course_id, sub_title, date,
                       transcript, transcript_segments_json,
                       proofread_transcript, proofread_segments_json,
                       proofread_model, proofread_at,
                       summary, processed_at, emailed_at,
                       error_msg, error_count, error_stage,
                       summary_model, blackboard_latex, blackboard_model, blackboard_at
                FROM local.lectures
            """)

            conn.execute("""
                UPDATE main.lectures SET
                    transcript = COALESCE(l.transcript, main.lectures.transcript),
                    transcript_segments_json = COALESCE(
                        l.transcript_segments_json,
                        main.lectures.transcript_segments_json
                    ),
                    proofread_transcript = COALESCE(
                        l.proofread_transcript,
                        main.lectures.proofread_transcript
                    ),
                    proofread_segments_json = COALESCE(
                        l.proofread_segments_json,
                        main.lectures.proofread_segments_json
                    ),
                    proofread_model = COALESCE(
                        l.proofread_model,
                        main.lectures.proofread_model
                    ),
                    proofread_at = COALESCE(
                        l.proofread_at,
                        main.lectures.proofread_at
                    ),
                    summary = COALESCE(l.summary, main.lectures.summary),
                    summary_model = COALESCE(
                        l.summary_model,
                        main.lectures.summary_model
                    ),
                    blackboard_latex = COALESCE(
                        l.blackboard_latex,
                        main.lectures.blackboard_latex
                    ),
                    blackboard_model = COALESCE(
                        l.blackboard_model,
                        main.lectures.blackboard_model
                    ),
                    blackboard_at = COALESCE(
                        l.blackboard_at,
                        main.lectures.blackboard_at
                    ),
                    processed_at = COALESCE(
                        l.processed_at,
                        main.lectures.processed_at
                    ),
                    emailed_at = COALESCE(l.emailed_at, main.lectures.emailed_at),
                    error_msg = CASE
                        WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                        THEN NULL
                        ELSE COALESCE(l.error_msg, main.lectures.error_msg)
                    END,
                    error_count = CASE
                        WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                        THEN 0
                        ELSE MAX(
                            COALESCE(l.error_count, 0),
                            COALESCE(main.lectures.error_count, 0)
                        )
                    END,
                    error_stage = CASE
                        WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                        THEN NULL
                        ELSE COALESCE(l.error_stage, main.lectures.error_stage)
                    END
                FROM local.lectures l
                WHERE main.lectures.sub_id = l.sub_id
            """)

            conn.execute("""
                INSERT OR IGNORE INTO main.ppt_pages
                    (sub_id, page_num, created_sec, pptimgurl, text, ocr_status, ocr_at, dhash)
                SELECT sub_id, page_num, created_sec, pptimgurl, text, ocr_status, ocr_at, dhash
                FROM local.ppt_pages
            """)

            # Blackboard metadata is deliberately stored in meta because that
            # table is copied by key, not by the physical lectures column order.
            has_local_meta = conn.execute(
                "SELECT 1 FROM local.sqlite_master "
                "WHERE type='table' AND name='meta'"
            ).fetchone()
            if has_local_meta:
                conn.execute("""
                    INSERT OR REPLACE INTO main.meta (key, value)
                    SELECT key, value
                    FROM local.meta
                    WHERE key LIKE 'blackboard_cache_version:%'
                       OR key LIKE 'blackboard_cache_blob:%'
                       OR key LIKE 'blackboard_checkpoint:%'
                       OR key LIKE 'blackboard_editor_checkpoint:%'
                """)

            has_all_courses = conn.execute(
                "SELECT 1 FROM local.sqlite_master "
                "WHERE type='table' AND name='all_courses'"
            ).fetchone()
            if has_all_courses:
                conn.execute("""
                    INSERT INTO main.all_courses
                        (course_id, term, title, teacher, dept, last_seen_at)
                    SELECT course_id, term, title, teacher, dept, last_seen_at
                    FROM local.all_courses
                    WHERE true
                    ON CONFLICT(course_id, term) DO UPDATE SET
                        title        = excluded.title,
                        teacher      = excluded.teacher,
                        dept         = excluded.dept,
                        last_seen_at = excluded.last_seen_at
                    WHERE excluded.last_seen_at > all_courses.last_seen_at
                """)

    finally:
        course_ids_env = os.environ.get("COURSE_IDS", "")
        if course_ids_env:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) "
                "VALUES ('course_ids', ?)", (course_ids_env,),
            )
            conn.commit()

        try:
            conn.execute("DETACH DATABASE local")
        except sqlite3.Error:
            pass
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} LOCAL_DB REMOTE_DB")
        print("Merges LOCAL_DB into REMOTE_DB (additive-only).")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
    print("Merge complete.")
