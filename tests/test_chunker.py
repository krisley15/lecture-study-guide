import pytest

from studyguide.chunker import chunk_text


def sentences(n, words_each=10):
    return " ".join(("word " * (words_each - 1) + "end.") for _ in range(n))


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_max_words():
    chunks = chunk_text(sentences(50), max_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert all(c.word_count <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_words=40, overlap_words=10)
    assert chunks[0].text.split(". ")[-1] in chunks[1].text


def test_long_sentence_is_split():
    chunks = chunk_text("word " * 500, max_words=100, overlap_words=0)
    assert all(c.word_count <= 100 for c in chunks)


def test_indices_are_sequential():
    chunks = chunk_text(sentences(50), max_words=100, overlap_words=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("hi.", max_words=10, overlap_words=10)
