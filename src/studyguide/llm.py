"""A tiny model-agnostic interface, plus a fake (for tests), Ollama, and a cache wrapper."""
import hashlib
import os
from pathlib import Path
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the model can't be reached or returns an error."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class FakeClient:
    """Records prompts and returns scripted (or numbered) responses. For tests."""

    def __init__(self, responses: list[str] | None = None):
        self.prompts: list[str] = []
        self._responses = list(responses) if responses else None

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return f"[fake response {len(self.prompts)}]"


class OllamaClient:
    """Talks to a local Ollama server. Override the model with STUDYGUIDE_OLLAMA_MODEL."""

    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.2, num_ctx: int = 4096):
        self.model = model or os.environ.get("STUDYGUIDE_OLLAMA_MODEL", "llama3.1:8b")
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx

    def generate(self, prompt: str) -> str:
        import ollama  # lazy so tests don't need the server or the package

        try:
            response = ollama.Client(host=self.host).chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
            )
        except ConnectionError as e:
            raise LLMError("Can't reach Ollama. Is it running? Try `ollama serve`.") from e
        except ollama.ResponseError as e:
            raise LLMError(
                f"Ollama error: {e}. If the model is missing, run `ollama pull {self.model}`."
            ) from e
        return response["message"]["content"].strip()


class CachedClient:
    """Wraps any client; identical prompts are answered from disk instead of the model."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path = ".cache/llm", namespace: str = ""):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace

    def generate(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.namespace}\x00{prompt}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        text = self.inner.generate(prompt)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
        return text
