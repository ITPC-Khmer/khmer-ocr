"""PDF → text. Digital PDFs are read directly; scanned PDFs go through OCR."""
import io
import unicodedata

import pymupdf
from PIL import Image

from .correct import correct_text
from .engine import DEFAULT_LANG, ocr_image_conf, postprocess

OCR_DPI = 300
RETRY_DPI = 400  # small screenshot text often needs the extra resolution
RETRY_PAGE_CONFIDENCE = 75.0
# If a page yields fewer characters than this, treat it as scanned
MIN_TEXT_CHARS_PER_PAGE = 20


def pdf_to_text(pdf_bytes: bytes, lang: str = DEFAULT_LANG) -> str:
    """Extract text from a PDF, using the embedded text layer when present."""
    embedded = _extract_text_layer(pdf_bytes)
    if embedded is not None:
        return embedded
    return _ocr_pdf(pdf_bytes, lang)


def _extract_text_layer(pdf_bytes: bytes) -> str | None:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text().strip() for page in doc]
    total = sum(len(p) for p in pages)
    if total < MIN_TEXT_CHARS_PER_PAGE * max(len(pages), 1):
        return None
    text = "\n\n".join(pages)
    if _looks_garbled(text):
        return None
    return unicodedata.normalize("NFC", text).strip()


# Legacy Khmer fonts (Limon-style) map subglyphs onto Latin Extended-A/B and
# combining-diacritic codepoints; their presence next to Khmer script means the
# text layer is unusable and the page must be OCR'd instead.
_GARBLE_RANGES = ((0x0100, 0x024F), (0x0250, 0x02FF))


def _looks_garbled(text: str) -> bool:
    khmer = sum(1 for c in text if 0x1780 <= ord(c) <= 0x17FF)
    if not khmer:
        return False
    garbled = sum(
        1 for c in text if any(lo <= ord(c) <= hi for lo, hi in _GARBLE_RANGES)
    )
    return garbled / (khmer + garbled) > 0.02


def _ocr_pdf(pdf_bytes: bytes, lang: str) -> str:
    pages: list[str] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text, conf = ocr_image_conf(_render(page, OCR_DPI), lang)
            if conf < RETRY_PAGE_CONFIDENCE:
                # Rendering resolution changes which glyphs survive
                # binarization; keep whichever pass Tesseract trusts more.
                text2, conf2 = ocr_image_conf(_render(page, RETRY_DPI), lang)
                if conf2 > conf:
                    text, conf = text2, conf2
            pages.append(correct_text(text, conf))
    return postprocess("\n\n".join(pages))


def _render(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))
