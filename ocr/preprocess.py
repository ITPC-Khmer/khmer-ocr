"""Image cleanup before OCR. Khmer diacritics need high resolution and clean binarization."""
import cv2
import numpy as np
from PIL import Image

MIN_HEIGHT = 700  # upscale small images so stacked consonants stay legible
MAX_SCALE = 2.5   # beyond this, interpolation artifacts hurt more than size helps


def upscale_if_small(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if h >= MIN_HEIGHT:
        return pil_image
    scale = min(MIN_HEIGHT / h, MAX_SCALE)
    return pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def preprocess(pil_image: Image.Image) -> Image.Image:
    img = np.array(upscale_if_small(pil_image.convert("L")))

    # Denoise, then binarize with Otsu
    img = cv2.fastNlMeansDenoising(img, h=10)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = _deskew(img)
    return Image.fromarray(img)


def _deskew(img: np.ndarray) -> np.ndarray:
    # Estimate skew from the minimum-area rectangle around dark pixels
    inverted = 255 - img
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > 15:  # ignore noise and false detections
        return img
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
