"""Pluggable cross-encoder rerankers (optional second retrieval stage).

A reranker scores ``(query, document)`` text pairs jointly with a
cross-encoder, which reads both texts at once instead of comparing
pre-computed vectors. Reranking the top candidates is the
best-documented quality jump over bi-encoder retrieval, and — unlike
small bi-encoders, which largely ignore instructions — a cross-encoder
actually attends to intent text injected into the pair, so
:meth:`intentdb.db.IntentDB.query` prefixes the query with the active
intent's instruction before reranking.

Built-in rerankers (both optional dependencies):

- ``flashrank`` — ONNX cross-encoders on CPU via the ``flashrank``
  package; the default model (ms-marco-TinyBERT-L-2-v2) is ~4 MB.
  ``pip install intentdb[rerank]``
- ``crossencoder`` — sentence-transformers ``CrossEncoder`` models,
  e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
  ``pip install intentdb[sbert]``
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Spec used by ``query(rerank=True)``.
DEFAULT_RERANKER_SPEC = "flashrank"


class Reranker(ABC):
    """Interface all rerankers implement."""

    @abstractmethod
    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        """Relevance of each text to the query, higher is better.

        Returns a float array of shape ``(len(texts),)``. The scale is
        model-defined (logits, probabilities, ...); only the ordering is
        meaningful across rerankers.
        """

    @property
    @abstractmethod
    def spec(self) -> str:
        """Spec string that :func:`get_reranker` can rebuild this from."""


class FlashRankReranker(Reranker):
    """Reranks with a flashrank ONNX cross-encoder (CPU, no torch).

    The default model is a ~4 MB TinyBERT cross-encoder downloaded on
    first use; pass ``model`` to use any other flashrank-supported model.
    """

    DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"

    def __init__(self, model: str = DEFAULT_MODEL):
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "flashrank is required for the 'flashrank' reranker: "
                "pip install intentdb[rerank]"
            ) from e
        self.model = model
        self._ranker = Ranker(model_name=model)
        self._request_cls = RerankRequest

    @property
    def spec(self) -> str:
        return f"flashrank:model={self.model}"

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float64)
        passages = [{"id": i, "text": t} for i, t in enumerate(texts)]
        ranked = self._ranker.rerank(self._request_cls(query=query, passages=passages))
        # flashrank returns the passages sorted by score; only pairwise
        # cross-encoder models attach a per-passage "score"
        by_id = {p.get("id"): p.get("score") for p in ranked}
        if any(by_id.get(i) is None for i in range(len(texts))):
            raise ValueError(
                f"flashrank model {self.model!r} did not return per-passage "
                "scores; use a pairwise cross-encoder model such as "
                f"{self.DEFAULT_MODEL}"
            )
        return np.array([float(by_id[i]) for i in range(len(texts))], dtype=np.float64)


class CrossEncoderReranker(Reranker):
    """Reranks with a sentence-transformers ``CrossEncoder`` model."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers is required for the 'crossencoder' "
                "reranker: pip install intentdb[sbert]"
            ) from e
        self.model = model
        self._model = CrossEncoder(model)

    @property
    def spec(self) -> str:
        return f"crossencoder:model={self.model}"

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(
            self._model.predict([(query, t) for t in texts]), dtype=np.float64
        )


def get_reranker(spec: str) -> Reranker:
    """Build a reranker from a spec string like ``"flashrank:model=..."``.

    Format: ``name`` or ``name:key=value,key=value`` (same convention as
    embedder specs).
    """
    from .embedders import _parse_kwargs

    name, _, arg_str = spec.partition(":")
    kwargs = _parse_kwargs(arg_str)
    name = name.strip().lower()
    if name == "flashrank":
        return FlashRankReranker(
            model=kwargs.get("model", FlashRankReranker.DEFAULT_MODEL)
        )
    if name == "crossencoder":
        return CrossEncoderReranker(
            model=kwargs.get("model", CrossEncoderReranker.DEFAULT_MODEL)
        )
    raise ValueError(f"unknown reranker {name!r} (known: flashrank, crossencoder)")
