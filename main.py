"""Khmer OCR API — image/PDF to text or MS Word. 100% local, no paid services.

Run: uvicorn main:app --reload
"""
import asyncio
import io
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

load_dotenv()
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from ocr import db, jobs
from ocr.correct import correct_text
from ocr.engine import DEFAULT_LANG, ocr_image_conf
from ocr.export import to_docx_bytes, to_txt_bytes
from ocr.pdf_handler import pdf_to_text

app = FastAPI(title="Khmer OCR", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    db.init_db()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr/image")
async def ocr_image_endpoint(
    file: UploadFile = File(...),
    format: str = Query("txt", pattern="^(txt|docx|json)$"),
    lang: str = Query(DEFAULT_LANG),
):
    data = await _read_upload(file)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "File is not a valid image (use PNG/JPG/TIFF/WebP)")

    def _ocr() -> str:
        text, conf = ocr_image_conf(image, lang)
        return correct_text(text, conf)

    text = await asyncio.to_thread(_ocr)
    return _respond(text, format, file.filename)


@app.post("/ocr/pdf", status_code=202)
async def ocr_pdf_endpoint(
    file: UploadFile = File(...),
    format: str = Query("txt", pattern="^(txt|docx|json)$"),
    lang: str = Query(DEFAULT_LANG),
):
    """Enqueue a PDF OCR job. Big scans take minutes — poll /jobs/{id}."""
    data = await _read_upload(file)
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")

    job_id = jobs.submit(lambda: pdf_to_text(data, lang), format, file.filename)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/jobs/{job_id}",
        "result_url": f"/jobs/{job_id}/result",
    }


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = await asyncio.to_thread(_get_job, job_id)
    body = {"job_id": job["id"], "status": job["status"]}
    if job["status"] == "queued":
        body["queue_position"] = db.queue_position(job_id)
    if job["status"] == "error":
        body["error"] = job["error"]
    if job["status"] == "done":
        body["chars"] = len(job["result"] or "")
        body["result_url"] = f"/jobs/{job_id}/result"
    return body


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = await asyncio.to_thread(_get_job, job_id)
    if job["status"] == "error":
        raise HTTPException(422, f"Could not process PDF: {job['error']}")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not finished (status: {job['status']})")
    return _respond(job["result"] or "", job["format"], job["filename"])


def _get_job(job_id: str) -> dict:
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(404, "Unknown job id")
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return job


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File larger than 50MB")
    return data


def _respond(text: str, format: str, filename: str | None):
    stem = Path(filename or "output").stem or "output"
    if format == "json":
        return {"text": text, "chars": len(text)}
    if format == "docx":
        return Response(
            to_docx_bytes(text),
            media_type=DOCX_MIME,
            headers={"Content-Disposition": f'attachment; filename="{stem}.docx"'},
        )
    return Response(
        to_txt_bytes(text),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.txt"'},
    )
