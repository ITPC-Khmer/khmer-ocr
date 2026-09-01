"""Export OCR text as .txt or .docx bytes."""
import io

from docx import Document
from docx.oxml.ns import qn

KHMER_FONT = "Noto Sans Khmer"  # sudo apt install fonts-noto


def to_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def to_docx_bytes(text: str) -> bytes:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = KHMER_FONT
    # Word needs the complex-script font set separately for Khmer
    style.element.rPr.rFonts.set(qn("w:cs"), KHMER_FONT)

    for para in text.split("\n\n"):
        doc.add_paragraph(para)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
