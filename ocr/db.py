"""PostgreSQL persistence for OCR jobs. Configure with DATABASE_URL."""
import os
from contextlib import contextmanager

import psycopg

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_jobs (
    id        uuid PRIMARY KEY,
    status    text NOT NULL,
    format    text NOT NULL,
    filename  text,
    result    text,
    error     text,
    created   timestamptz NOT NULL DEFAULT now(),
    finished  timestamptz
);
"""


def _dsn() -> str:
    return os.environ.get("DATABASE_URL", "postgresql:///khmer_ocr")


@contextmanager
def conn():
    with psycopg.connect(_dsn()) as c:
        yield c


def init_db() -> None:
    with conn() as c:
        c.execute(_SCHEMA)


def create_job(job_id: str, format: str, filename: str | None) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO ocr_jobs (id, status, format, filename)"
            " VALUES (%s, 'queued', %s, %s)",
            (job_id, format, filename),
        )


def mark_processing(job_id: str) -> None:
    with conn() as c:
        c.execute(
            "UPDATE ocr_jobs SET status = 'processing' WHERE id = %s",
            (job_id,),
        )


def finish_job(job_id: str, result: str | None, error: str | None) -> None:
    status = "done" if error is None else "error"
    with conn() as c:
        c.execute(
            "UPDATE ocr_jobs SET status = %s, result = %s, error = %s,"
            " finished = now() WHERE id = %s",
            (status, result, error, job_id),
        )


def get_job(job_id: str) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT id, status, format, filename, result, error, created"
            " FROM ocr_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    keys = ("id", "status", "format", "filename", "result", "error", "created")
    return dict(zip(keys, row))


def queue_position(job_id: str) -> int:
    """1-based position among queued jobs (0 if not queued)."""
    with conn() as c:
        row = c.execute(
            "SELECT count(*) FROM ocr_jobs q, ocr_jobs me"
            " WHERE me.id = %s AND me.status = 'queued'"
            " AND q.status = 'queued' AND q.created <= me.created",
            (job_id,),
        ).fetchone()
    return int(row[0]) if row else 0
