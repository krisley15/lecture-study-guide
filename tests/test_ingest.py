import pytest

from studyguide.ingest import clean_transcript, load_text


def test_vtt_is_cleaned():
    raw = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nHello <b>class</b>.\n\n2\n00:00:03.500 --> 00:00:05.000\nToday: graphs.\n"
    assert clean_transcript(raw) == "Hello class. Today: graphs."


def test_load_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Some notes.")
    assert load_text(f) == "Some notes."


def test_unsupported_type(tmp_path):
    f = tmp_path / "slides.pptx"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_text(f)
