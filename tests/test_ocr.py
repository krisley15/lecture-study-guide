from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from studyguide import ingest, ocr


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = 0
        self.last_kwargs = None
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text="  Quicksort picks a pivot.  ")],
        )


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


def test_pdf_is_sent_as_document_block(tmp_path):
    f = tmp_path / "notes.pdf"
    f.write_bytes(b"%PDF-fake")
    client = FakeClient()
    text = ocr.transcribe(f, client=client, cache_dir=tmp_path / "cache")
    assert text == "Quicksort picks a pivot."
    block = client.messages.last_kwargs["messages"][0]["content"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_image_is_sent_as_image_block(tmp_path):
    f = tmp_path / "page.png"
    f.write_bytes(b"fake png")
    client = FakeClient()
    ocr.transcribe(f, client=client, cache_dir=tmp_path / "cache")
    block = client.messages.last_kwargs["messages"][0]["content"][0]
    assert block["type"] == "image"


def test_transcribe_caches_by_file_hash(tmp_path):
    f = tmp_path / "page.png"
    f.write_bytes(b"fake png")
    client = FakeClient()
    ocr.transcribe(f, client=client, cache_dir=tmp_path / "cache")
    ocr.transcribe(f, client=client, cache_dir=tmp_path / "cache")
    assert client.messages.calls == 1


def test_truncated_output_is_not_cached(tmp_path):
    f = tmp_path / "page.png"
    f.write_bytes(b"fake png")
    with pytest.raises(RuntimeError):
        ocr.transcribe(f, client=FakeClient(stop_reason="max_tokens"), cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / "page.png"
    f.write_bytes(b"x")
    with pytest.raises(ocr.OCRConfigError):
        ocr.transcribe(f, cache_dir=tmp_path / "cache")


def _blank_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_textless_pdf_falls_back_to_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "handwritten.pdf"
    _blank_pdf(pdf)
    monkeypatch.setattr(ingest, "_ocr", lambda p: "transcribed text")
    assert ingest.load_text(pdf) == "transcribed text"


def test_fallback_can_be_disabled(tmp_path):
    pdf = tmp_path / "handwritten.pdf"
    _blank_pdf(pdf)
    assert ingest.load_text(pdf, ocr_fallback=False).strip() == ""
