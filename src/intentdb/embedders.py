"""Pluggable text embedders.

Every embedder turns text into a unit-norm float32 vector of a fixed
dimension. The database persists the embedder *spec string* (for example
``"hashing:dim=512"``) so a store is always reopened with the embedder it
was built with.

Built-in embedders:

- ``hashing``, deterministic feature-hashing embedder with zero external
  dependencies. No model download, no network, fully reproducible. Quality
  is lexical (token/character-n-gram overlap), which is plenty for tests,
  demos, and keyword-ish corpora.
- ``ollama``, calls a local Ollama server (``/api/embeddings``) using only
  the standard library. Use any embedding model you have pulled, e.g.
  ``ollama:model=nomic-embed-text``.
- ``sbert``, sentence-transformers models, e.g.
  ``sbert:model=all-MiniLM-L6-v2`` (requires the optional dependency).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    """Interface all embedders implement.

    Beyond plain :meth:`embed`, embedders expose an *asymmetric*,
    *instruction-aware* API in the style of instruction-finetuned models
    (INSTRUCTOR, nomic-embed, e5): queries and documents may be embedded
    differently, and a query can be conditioned on an intent instruction so
    its vectorization changes with the retrieval intent. Embedders that
    cannot make use of instructions (e.g. the lexical hashing embedder)
    simply ignore them, keeping ``supports_instructions = False``.
    """

    #: embedding dimensionality
    dim: int

    #: whether intent instructions change this embedder's query vectors
    supports_instructions: bool = False

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a unit-norm float32 vector of shape ``(dim,)``."""

    def embed_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        """Embed a search query, optionally conditioned on an intent
        instruction. The default ignores the instruction."""
        return self.embed(text)

    def embed_document(self, text: str) -> np.ndarray:
        """Embed a document for storage."""
        return self.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix of shape ``(len(texts), dim)``."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])

    def embed_document_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_document(t) for t in texts])

    @property
    @abstractmethod
    def spec(self) -> str:
        """Spec string that :func:`get_embedder` can rebuild this from."""


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32)


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder (no model, no dependencies).

    Tokens (lowercased words plus character trigrams for fuzziness) are
    hashed into ``dim`` signed buckets with sublinear term-frequency
    weighting, then L2-normalized. The same text always produces the same
    vector, across processes and machines.
    """

    def __init__(self, dim: int = 512, char_ngrams: bool = True):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.char_ngrams = char_ngrams

    @property
    def spec(self) -> str:
        return f"hashing:dim={self.dim},char_ngrams={int(self.char_ngrams)}"

    def _features(self, text: str) -> dict[str, float]:
        words = _TOKEN_RE.findall(text.lower())
        feats: dict[str, float] = {}
        for w in words:
            feats[w] = feats.get(w, 0.0) + 1.0
            if self.char_ngrams and len(w) > 3:
                for i in range(len(w) - 2):
                    g = "#" + w[i : i + 3]
                    feats[g] = feats.get(g, 0.0) + 0.25
        return feats

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for feat, count in self._features(text).items():
            h = int.from_bytes(
                hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 62) & 1 else -1.0
            v[idx] += sign * (1.0 + math.log(count)) if count >= 1 else sign * count
        return _normalize(v)


#: Task-prefix presets for asymmetric-retrieval embedding models.
PREFIX_MODES: dict[str, tuple[str, str]] = {
    "none": ("", ""),
    "nomic": ("search_query: ", "search_document: "),
    "e5": ("query: ", "passage: "),
}


class OllamaEmbedder(Embedder):
    """Embeds via a local Ollama server using only the standard library.

    ``prefix_mode`` selects the task prefixes the model was trained with
    (``nomic`` → ``search_query:`` / ``search_document:``, ``e5`` →
    ``query:`` / ``passage:``). Intent instructions are injected into the
    query text, INSTRUCTOR-style, so the query vector shifts toward the
    active intent.
    """

    supports_instructions = True

    #: how many texts to send per /api/embed request
    batch_size = 32

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        dim: int | None = None,
        prefix_mode: str = "nomic",
    ):
        if prefix_mode not in PREFIX_MODES:
            raise ValueError(
                f"unknown prefix_mode {prefix_mode!r} (known: {sorted(PREFIX_MODES)})"
            )
        self.model = model
        self.host = host.rstrip("/")
        self.prefix_mode = prefix_mode
        self.query_prefix, self.doc_prefix = PREFIX_MODES[prefix_mode]
        # Probe the dimension once if not given.
        self.dim = dim if dim is not None else self._embed_many(["dimension probe"]).shape[1]

    @property
    def spec(self) -> str:
        return (
            f"ollama:model={self.model},host={self.host},dim={self.dim},"
            f"prefix_mode={self.prefix_mode}"
        )

    def _post(self, path: str, body: dict) -> dict:
        """POST JSON to the Ollama server, retrying transient 5xx/connection errors."""
        payload = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    f"{self.host}{path}",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                last = e
                if 500 <= e.code < 600 and attempt < 3:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise
            except urllib.error.URLError as e:
                last = e
                if attempt < 3:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise
        raise last  # pragma: no cover

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the batched /api/embed endpoint; returns a unit-norm matrix.

        Batching keeps the number of HTTP requests small (one per ``batch_size``
        texts) instead of one per text, which is both faster and far gentler on
        the Ollama server than a long burst of single-prompt calls.
        """
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            data = self._post("/api/embed", {"model": self.model, "input": chunk})
            vectors.extend(data["embeddings"])
        mat = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    def embed_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        if instruction:
            text = f"{instruction.strip()}: {text}"
        return self._embed_many([self.query_prefix + text])[0]

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed_many([self.doc_prefix + text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed_many(texts)

    def embed_document_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed_many([self.doc_prefix + t for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Embeds with a sentence-transformers model (optional dependency).

    Intent instructions are prepended to the query text, INSTRUCTOR-style.
    """

    supports_instructions = True

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers is required for the 'sbert' embedder: "
                "pip install intentdb[sbert]"
            ) from e
        self.model_name = model
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def spec(self) -> str:
        return f"sbert:model={self.model_name}"

    def embed(self, text: str) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(text), dtype=np.float64))

    def embed_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        if instruction:
            text = f"{instruction.strip()}: {text}"
        return self.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.asarray(self._model.encode(texts), dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


def _parse_kwargs(arg_str: str) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if arg_str:
        for part in arg_str.split(","):
            if "=" not in part:
                raise ValueError(f"bad embedder argument {part!r} (want key=value)")
            k, v = part.split("=", 1)
            kwargs[k.strip()] = v.strip()
    return kwargs


def get_embedder(spec: str) -> Embedder:
    """Build an embedder from a spec string like ``"hashing:dim=512"``.

    Format: ``name`` or ``name:key=value,key=value``.
    """
    name, _, arg_str = spec.partition(":")
    kwargs = _parse_kwargs(arg_str)
    name = name.strip().lower()
    if name == "hashing":
        return HashingEmbedder(
            dim=int(kwargs.get("dim", 512)),
            char_ngrams=bool(int(kwargs.get("char_ngrams", 1))),
        )
    if name == "ollama":
        return OllamaEmbedder(
            model=kwargs.get("model", "nomic-embed-text"),
            host=kwargs.get("host", "http://localhost:11434"),
            dim=int(kwargs["dim"]) if "dim" in kwargs else None,
            prefix_mode=kwargs.get("prefix_mode", "nomic"),
        )
    if name == "sbert":
        return SentenceTransformerEmbedder(model=kwargs.get("model", "all-MiniLM-L6-v2"))
    raise ValueError(f"unknown embedder {name!r} (known: hashing, ollama, sbert)")
