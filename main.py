"""Khmer OCR API — image/PDF to text or MS Word. 100% local, no paid services.

Run: uvicorn main:app --reload
"""
import asyncio
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from ocr.correct import correct_text
from ocr.engine import DEFAULT_LANG, ocr_image_conf
from ocr.export import to_docx_bytes, to_txt_bytes
from ocr.pdf_handler import pdf_to_text

app = FastAPI(title="Khmer OCR", version="1.0.0")

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


@app.post("/ocr/pdf")
async def ocr_pdf_endpoint(
    file: UploadFile = File(...),
    format: str = Query("txt", pattern="^(txt|docx|json)$"),
    lang: str = Query(DEFAULT_LANG),
):
    data = await _read_upload(file)
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")

    try:
        text = await asyncio.to_thread(pdf_to_text, data, lang)
    except Exception as exc:
        raise HTTPException(422, f"Could not process PDF: {exc}")
    return _respond(text, format, file.filename)


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
