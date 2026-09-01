"""Tesseract OCR wrapper for Khmer.

Two-pass strategy: clean digital images OCR best untouched, while dirty
scans need aggressive preprocessing — so run on the near-raw image first
and retry with full cleanup only when confidence is low.
"""
import unicodedata
from pathlib import Path

import pytesseract
from PIL import Image

from .preprocess import preprocess, upscale_if_small

DEFAULT_LANG = "khm+eng"  # Khmer documents often contain Latin numbers/words
RETRY_CONFIDENCE = 70.0

# Project-local tessdata_best models take priority over system ones
_TESSDATA_DIR = Path(__file__).resolve().parent.parent / "tessdata"
_BASE_CONFIG = (
    f'--tessdata-dir "{_TESSDATA_DIR}"' if _TESSDATA_DIR.is_dir() else ""
)


def ocr_image(pil_image: Image.Image, lang: str = DEFAULT_LANG) -> str:
    text, _ = ocr_image_conf(pil_image, lang)
    return text


def ocr_image_conf(
    pil_image: Image.Image, lang: str = DEFAULT_LANG
) -> tuple[str, float]:
    """OCR an image and return (text, mean word confidence 0-100)."""
    gray = upscale_if_small(pil_image.convert("L"))
    text, conf = _run(gray, lang)

    if conf < RETRY_CONFIDENCE:
        # psm 6 (uniform block) handles single lines / sparse pages better
        # than the default auto segmentation
        for candidate in (gray, preprocess(pil_image)):
            for psm in ("--psm 6", ""):
                if candidate is gray and not psm:
                    continue  # already ran
                text2, conf2 = _run(candidate, lang, config=psm)
                if conf2 > conf:
                    text, conf = text2, conf2
            if conf >= RETRY_CONFIDENCE:
                break

    return postprocess(text), conf


def _run(image: Image.Image, lang: str, config: str = "") -> tuple[str, float]:
    config = f"{_BASE_CONFIG} {config}".strip()
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    line_key = None
    lines: list[str] = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key != line_key:
            if words:
                lines.append(" ".join(words))
            words = []
            line_key = key
        words.append(word)
        confs.append(conf)
    if words:
        lines.append(" ".join(words))
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "\n".join(lines), mean_conf


def postprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")  # zero-width space / BOM
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()
