"""Load lecture material from files into plain text."""
import re
from pathlib import Path

_TIMESTAMP = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?([.,]\d{3})?\s*-->.*$")
_CUE_NUMBER = re.compile(r"^\d+$")
_TAGS = re.compile(r"<[^>]+>")

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIN_CHARS_PER_PAGE = 25  # below this, a PDF is treated as scanned/handwritten


def clean_transcript(raw: str) -> str:
    """Strip WEBVTT/SRT headers, cue numbers, timestamps, and inline tags."""
    kept = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s == "WEBVTT" or _TIMESTAMP.match(s) or _CUE_NUMBER.match(s):
            continue
        kept.append(_TAGS.sub("", s))
    return " ".join(kept)


def _ocr(path: Path) -> str:
    from studyguide import ocr  # lazy: Anthropic SDK only needed for this path

    return ocr.transcribe(path)


def load_text(path: str | Path, ocr_fallback: bool = True) -> str:
    """Read .txt/.md, .vtt/.srt, .pdf, or images into a string.

    PDFs with no real text layer (scans, handwriting) fall back to Claude vision transcription
    when ocr_fallback is True.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix in {".vtt", ".srt"}:
        return clean_transcript(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        from pypdf import PdfReader  # lazy so other formats work without it

        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        if ocr_fallback and len(text.strip()) < MIN_CHARS_PER_PAGE * max(len(reader.pages), 1):
            return _ocr(path)
        return text
    if suffix in IMAGE_TYPES:
        return _ocr(path)
    raise ValueError(f"Unsupported file type: {suffix}")
