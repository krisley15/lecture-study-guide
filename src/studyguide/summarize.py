"""Map-reduce study guide generation: summarize each chunk, then merge."""
from collections.abc import Callable, Sequence

from studyguide.chunker import Chunk
from studyguide.llm import LLMClient

CHUNK_PROMPT = """You are helping a student study. Summarize the lecture excerpt below.
Use ONLY information in the excerpt. Do not add outside facts.
Write 3 to 6 short bullet points of the main ideas, then one line starting "Key terms:" \
listing the important terms.

Excerpt:
\"\"\"
{text}
\"\"\"
"""

CONDENSE_PROMPT = """Below are study notes from consecutive parts of one lecture. \
Merge them into one shorter set of bullet points. Remove duplicates, keep every important \
idea and key term, and add nothing new.

Notes:
{notes}
"""

COMBINE_PROMPT = """Below are notes from one lecture. Turn them into a study guide in Markdown \
with exactly these sections:

## Overview
2 to 3 sentences.

## Key Ideas
Bullet points, with duplicates merged.

## Key Terms
One line per term: "term: short definition".

## Practice Questions
5 questions that test understanding, each followed by "Answer: ...".

Use ONLY information in the notes. If the notes don't support a section, keep it short \
rather than inventing content.

Notes:
{notes}
"""


def summarize_chunks(
    chunks: Sequence[Chunk],
    client: LLMClient,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    notes = []
    for i, chunk in enumerate(chunks, start=1):
        if progress:
            progress(i, len(chunks))
        notes.append(client.generate(CHUNK_PROMPT.format(text=chunk.text)))
    return notes


def _word_total(notes: list[str]) -> int:
    return sum(len(n.split()) for n in notes)


def _group(notes: list[str], limit: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for note in notes:
        words = len(note.split())
        if current and count + words > limit:
            groups.append(current)
            current, count = [], 0
        current.append(note)
        count += words
    if current:
        groups.append(current)
    return groups


def build_study_guide(
    chunks: Sequence[Chunk],
    client: LLMClient,
    max_combine_words: int = 1500,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Summarize each chunk, condense if the notes are too long, then write the guide."""
    if not chunks:
        raise ValueError("No chunks to summarize.")
    notes = summarize_chunks(chunks, client, progress)

    # Keep the final prompt inside the model's context window.
    while len(notes) > 1 and _word_total(notes) > max_combine_words:
        groups = _group(notes, max_combine_words)
        if len(groups) == len(notes):  # nothing to merge, avoid looping forever
            break
        notes = [
            client.generate(CONDENSE_PROMPT.format(notes="\n\n".join(g))) if len(g) > 1 else g[0]
            for g in groups
        ]

    return client.generate(COMBINE_PROMPT.format(notes="\n\n".join(notes)))
