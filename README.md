# Lecture Study Guide Generator

A command-line tool that turns lecture notes or transcripts into a structured Markdown study guide (overview, key ideas, key terms, and practice questions with answers). It runs entirely on your own machine using a local language model through [Ollama](https://ollama.com), so no account, API key, or cost is required and your notes never leave your computer.

## Features

- **Multiple input formats:** `.txt`, `.md`, `.vtt` / `.srt` caption files, and PDFs.
- **Sentence-aware chunking** with overlap, so long lectures fit a small model's context window without cutting ideas in half.
- **Map-reduce summarization:** each chunk is summarized separately, long note sets are condensed in groups, and a final pass writes the guide.
- **Model-agnostic design:** the pipeline depends on a small `LLMClient` interface. Included backends are a fake client (for tests) and Ollama. Adding a hosted model means writing one class.
- **Disk caching** of model responses, so re-running the same lecture is nearly instant.
- **Optional handwriting transcription:** PDFs with no text layer (scans, handwritten notes exported from a tablet) can be transcribed with a vision-capable Claude model if you provide an `ANTHROPIC_API_KEY`. Without a key, the tool tells you to convert the notes to text first.
- **Tested:** 26 unit tests, none of which require a model or network access.

## How it works

```
file ──> ingest ──> chunk ──> summarize each chunk ──> (condense if long) ──> combine ──> study_guide.md
          │                          │
   PDF text layer?              LLMClient (Ollama)
   no → optional OCR            responses cached on disk
```

1. `ingest.py` reads the file. Caption files are stripped of timestamps and cue numbers. PDFs with almost no extractable text fall back to optional transcription.
2. `chunker.py` splits text into sentence-aligned chunks of at most 400 words with 40 words of overlap.
3. `summarize.py` asks the model for bullet points and key terms per chunk, condenses the notes if they would overflow the context window, and generates the final guide.
4. `llm.py` provides the model interface, the Ollama backend, and the caching wrapper.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/krisley15/lecture-study-guide.git
cd lecture-study-guide

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1:8b
```

Make sure Ollama is running (the desktop app starts it automatically, or run `ollama serve`).

## Usage

Put a lecture file in `data/` (this folder is git-ignored so your course material stays private), then:

```bash
# See how a file would be chunked
PYTHONPATH=src python -m studyguide.cli data/lecture.txt

# Generate a study guide
PYTHONPATH=src python -m studyguide.cli data/lecture.txt --guide -o data/lecture_guide.md

# Use a different local model
PYTHONPATH=src python -m studyguide.cli data/lecture.txt --guide --model qwen2.5:3b
```

Responses are cached in `.cache/llm/`. Delete that folder to force a fresh run.

### Handwritten or scanned notes

If a PDF has no text layer, the tool can transcribe it with a Claude vision model:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
PYTHONPATH=src python -m studyguide.cli data/handwritten_notes.pdf --guide -o data/guide.md
```

This sends the document to Anthropic's API and is billed by usage. Transcriptions are cached by file hash. If you would rather not use an API, convert handwriting to text in your note-taking app and use the `.txt` file instead.

## Example output

Generated from a set of quicksort lecture notes with `llama3.1:8b` (excerpt, unedited model output):

```markdown
## Overview
Quicksort is a fast sorting algorithm that uses the divide and conquer approach to sort
arrays. It has a time complexity of O(n log n) on average, but can be O(n^2) in the worst
case. Quicksort is an in-place sort, meaning it does not require additional memory.

## Key Terms
* Pivot: an element used to divide the array into smaller subproblems.
* In-place sort: a sort that does not require additional memory.
* Stable sort: a sort that preserves the order of equal elements.

## Practice Questions
4. What is the worst-case time complexity of Quicksort?
Answer: O(n^2)

5. Is Quicksort a stable sort?
Answer: No
```

## Limitations

- **Small local models make mistakes.** The sample above is a good example: it says quicksort needs no additional memory, but strictly it uses O(log n) stack space on average. Always check generated guides against your source material, especially the practice-question answers.
- **Speed:** each chunk is a separate model call, so long lectures take minutes on a laptop.
- **Handwriting** depends on the optional transcription step and can misread code, symbols, and diagrams.
- Sentence splitting is regex-based and can mis-split text with abbreviations or code.

## Project structure

```
src/studyguide/
  chunker.py     sentence-aligned chunking with overlap
  ingest.py      file loading, transcript cleaning, OCR fallback
  llm.py         LLMClient interface, FakeClient, OllamaClient, CachedClient
  summarize.py   map-reduce guide generation and prompts
  ocr.py         optional handwriting transcription (Claude vision)
  cli.py         command-line interface
tests/           unit tests (no model or network needed)
```

## Running the tests

```bash
pytest
```

## Roadmap

- [ ] Structured JSON output (key terms and questions) validated with Pydantic, with retries on malformed output
- [ ] Evaluation set measuring how many expected key points each guide covers, to compare prompts and chunk sizes
- [ ] Streamlit upload interface
- [ ] Docker image and cloud deployment

## Privacy

Lecture material stays local by default. The only feature that sends data off your machine is the optional handwriting transcription, and only when you set an API key. Only use course material you have the right to use, and keep it out of public repositories.

## License

Choose a license (for example MIT) and add a `LICENSE` file.
