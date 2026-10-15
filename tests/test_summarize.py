import pytest

from studyguide.chunker import Chunk
from studyguide.llm import FakeClient
from studyguide.summarize import CHUNK_PROMPT, build_study_guide


def make_chunks(n):
    return [Chunk(i, f"lecture text number {i}", 4) for i in range(n)]


def test_small_lecture_uses_one_call_per_chunk_plus_combine():
    fake = FakeClient()
    guide = build_study_guide(make_chunks(3), fake)
    assert len(fake.prompts) == 4
    assert guide == "[fake response 4]"


def test_each_chunk_text_appears_in_its_prompt():
    fake = FakeClient()
    build_study_guide(make_chunks(2), fake)
    assert "lecture text number 0" in fake.prompts[0]
    assert "lecture text number 1" in fake.prompts[1]


def test_combine_prompt_contains_chunk_notes():
    fake = FakeClient(["NOTES-A", "NOTES-B", "final guide"])
    build_study_guide(make_chunks(2), fake)
    assert "NOTES-A" in fake.prompts[-1] and "NOTES-B" in fake.prompts[-1]


def test_long_notes_are_condensed_before_combining():
    fake = FakeClient()  # each response is 3 words
    build_study_guide(make_chunks(6), fake, max_combine_words=10)
    # 6 chunk summaries + 2 condense calls (groups of 3) + 1 final combine
    assert len(fake.prompts) == 9


def test_empty_input_raises():
    with pytest.raises(ValueError):
        build_study_guide([], FakeClient())


def test_chunk_prompt_tells_model_not_to_add_outside_facts():
    assert "ONLY" in CHUNK_PROMPT
