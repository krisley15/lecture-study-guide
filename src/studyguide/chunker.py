"""Split long text into overlapping, sentence-aligned chunks."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    word_count: int


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def chunk_text(text: str, max_words: int = 400, overlap_words: int = 40) -> list[Chunk]:
    """Group sentences into chunks of at most max_words.

    Consecutive chunks share up to overlap_words of trailing sentences so
    context isn't lost at boundaries. Sentences longer than max_words are
    split by words.
    """
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError("need max_words > 0 and 0 <= overlap_words < max_words")

    units: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for i in range(0, len(words), max_words):
            units.append(" ".join(words[i : i + max_words]))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                pn = len(prev.split())
                if carried + pn > overlap_words:
                    break
                carry.insert(0, prev)
                carried += pn
            current, count = carry, carried
            if count + n > max_words:  # overlap doesn't fit alongside new unit
                current, count = [], 0
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))

    return [Chunk(i, t, len(t.split())) for i, t in enumerate(chunks)]
