"""Usage:
  python -m studyguide.cli FILE                     show how FILE is chunked
  python -m studyguide.cli FILE --guide -o OUT.md   generate a study guide with a local model
FILE can be .txt, .md, .vtt, .srt, .pdf, or an image.
"""
import argparse
import sys
from pathlib import Path

from studyguide.chunker import chunk_text
from studyguide.ingest import load_text
from studyguide.llm import CachedClient, LLMError, OllamaClient
from studyguide.ocr import OCRConfigError
from studyguide.summarize import build_study_guide


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="studyguide", description="Lecture to study guide.")
    parser.add_argument("path")
    parser.add_argument("--guide", action="store_true", help="generate a study guide")
    parser.add_argument("--model", default=None, help="Ollama model (default llama3.1:8b)")
    parser.add_argument("-o", "--output", help="write the guide to this file")
    args = parser.parse_args(argv)

    try:
        text = load_text(args.path)
    except OCRConfigError as e:
        sys.exit(f"{e}\nOr convert your handwriting to a .txt file and use that instead.")
    if not text.strip():
        sys.exit("No text found. If this is a PDF, it may be scanned or image-based.")

    chunks = chunk_text(text)
    if not args.guide:
        print(f"{len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.index}] {c.word_count} words: {c.text[:60]}...")
        return

    ollama_client = OllamaClient(args.model)
    client = CachedClient(ollama_client, namespace=ollama_client.model)
    try:
        guide = build_study_guide(
            chunks,
            client,
            progress=lambda i, n: print(f"Summarizing chunk {i}/{n}...", file=sys.stderr),
        )
    except LLMError as e:
        sys.exit(str(e))

    if args.output:
        Path(args.output).write_text(guide, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(guide)


if __name__ == "__main__":
    main()
