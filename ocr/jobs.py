"""OCR job queue: results persist in PostgreSQL, work runs in-process.

Single worker thread: Tesseract and Ollama are CPU-bound, so running
jobs one at a time keeps the box responsive. Job state and results live
in PostgreSQL (see db.py), so finished results survive restarts; a job
still queued or processing when the process dies stays in that state —
resubmit it.
"""
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import db

_executor = ThreadPoolExecutor(max_workers=1)


def submit(work: Callable[[], str], format: str, filename: str | None) -> str:
    """Queue `work`; returns the job's UUID immediately."""
    job_id = str(uuid.uuid4())
    db.create_job(job_id, format, filename)

    def _run() -> None:
        db.mark_processing(job_id)
        try:
            db.finish_job(job_id, work(), None)
        except Exception as exc:  # surfaced to the client via GET /jobs/{id}
            db.finish_job(job_id, None, str(exc))

    _executor.submit(_run)
    return job_id
