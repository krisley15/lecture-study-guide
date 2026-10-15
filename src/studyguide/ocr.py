"""Transcribe scanned or handwritten documents with a vision-capable Claude model.

Credentials come from the ANTHROPIC_API_KEY environment variable (never hard-code it).
Set STUDYGUIDE_MODEL to override the default model.
Results are cached on disk by file hash so each document is only billed once.
"""
import base64
import hashlib
import os
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8000

PROMPT = (
    "Transcribe all handwritten and printed text in this document exactly as written. "
    "Keep the reading order and line breaks. Write code, math, and symbols as faithfully "
    "as you can. Describe any diagram in square brackets, like [diagram: ...]. Mark words "
    "you cannot read as [illegible]. Output only the transcription, with no commentary."
)

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class OCRConfigError(RuntimeError):
    """Raised when the API key is missing."""


def _make_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise OCRConfigError(
            "Set ANTHROPIC_API_KEY to transcribe scanned or handwritten files."
        )
    import anthropic  # lazy: only needed on this path

    return anthropic.Anthropic()


def transcribe(path: str | Path, client=None, cache_dir: str | Path = ".cache/ocr") -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError(f"Can't transcribe file type: {suffix}")

    model = os.environ.get("STUDYGUIDE_MODEL", DEFAULT_MODEL)
    data = path.read_bytes()
    key = hashlib.sha256(data + model.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    client = client or _make_client()
    block_type = "document" if suffix == ".pdf" else "image"
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": block_type,
                        "source": {
                            "type": "base64",
                            "media_type": MEDIA_TYPES[suffix],
                            "data": base64.standard_b64encode(data).decode(),
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            "Transcription was cut off. Try a file with fewer pages."
        )
    text = "".join(b.text for b in message.content if b.type == "text").strip()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text
