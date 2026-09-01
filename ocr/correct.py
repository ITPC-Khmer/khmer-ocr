"""Optional post-OCR correction via a local Ollama model.

Entirely best-effort: if Ollama is not installed, not running, or errors
out, the raw OCR text is returned unchanged. Configure with env vars
OLLAMA_URL and OLLAMA_MODEL.
"""
import json
import os
import urllib.error
import urllib.request

def _url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _model() -> str:
    return os.environ.get("OLLAMA_MODEL", "gemma2:9b")


# Only correct pages Tesseract itself is unsure about
CORRECT_BELOW_CONFIDENCE = float(os.environ.get("CORRECT_BELOW_CONFIDENCE", "85"))
TIMEOUT_SECONDS = 120

_PROMPT = (
    "The following is OCR output from a Khmer document (it may also contain "
    "English words and numbers). Fix obvious OCR errors: wrong or missing "
    "Khmer characters, broken syllables, stray symbols. Keep the line "
    "structure, keep correct text unchanged, and do NOT translate, summarize "
    "or add anything. Output only the corrected text.\n\n{text}"
)


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{_url()}/api/tags", timeout=2) as resp:
            models = json.load(resp).get("models", [])
        return any(m.get("name", "").startswith(_model()) for m in models)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def correct_text(text: str, confidence: float = 0.0) -> str:
    """Return LLM-corrected text, or the input unchanged on any failure."""
    # Off by default: LLM correction is too slow on small servers.
    # Set OLLAMA_CORRECT=1 to re-enable.
    if os.environ.get("OLLAMA_CORRECT", "0") != "1":
        return text
    if not text.strip() or confidence >= CORRECT_BELOW_CONFIDENCE:
        return text
    if not ollama_available():
        return text
    payload = json.dumps(
        {
            "model": _model(),
            "prompt": _PROMPT.format(text=text),
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        f"{_url()}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            corrected = json.load(resp).get("response", "").strip()
    except (urllib.error.URLError, OSError, ValueError):
        return text
    return corrected if _plausible(text, corrected) else text


def _plausible(original: str, corrected: str) -> bool:
    # Guard against the model rambling, refusing, or dropping the page
    if not corrected:
        return False
    ratio = len(corrected) / max(len(original), 1)
    return 0.5 <= ratio <= 1.5
