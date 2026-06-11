"""Lexical (sparse) retrieval: a small in-memory BM25 index.

Dense embeddings miss exact terms (identifiers, error codes, names);
sparse retrieval misses paraphrase. Production RAG fuses both. IntentDB
keeps a BM25 index over the corpus, rebuilt from the stored texts on open
and maintained incrementally on add/delete, and fuses its ranking with the
dense intent-aware ranking via Reciprocal Rank Fusion when ``hybrid=True``.

Postings are keyed by stable ``doc_key`` (not row position), so deletes
never invalidate the index.
"""

from __future__ import annotations

import math
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Incremental BM25 (Okapi) index keyed by document key."""

    def __init__(self) -> None:
        # term -> {doc_key: term frequency}
        self.postings: dict[str, dict[str, int]] = {}
        # doc_key -> {term: tf} (needed to undo a document's postings)
        self.doc_terms: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.doc_terms)

    @property
    def avg_doc_len(self) -> float:
        if not self.doc_len:
            return 0.0
        return sum(self.doc_len.values()) / len(self.doc_len)

    def add(self, doc_key: str, text: str) -> None:
        """Index (or re-index) a document."""
        if doc_key in self.doc_terms:
            self.remove(doc_key)
        tokens = tokenize(text)
        tfs: dict[str, int] = {}
        for t in tokens:
            tfs[t] = tfs.get(t, 0) + 1
        for term, tf in tfs.items():
            self.postings.setdefault(term, {})[doc_key] = tf
        self.doc_terms[doc_key] = tfs
        self.doc_len[doc_key] = len(tokens)

    def remove(self, doc_key: str) -> None:
        tfs = self.doc_terms.pop(doc_key, None)
        if tfs is None:
            return
        for term in tfs:
            bucket = self.postings.get(term)
            if bucket is not None:
                bucket.pop(doc_key, None)
                if not bucket:
                    del self.postings[term]
        del self.doc_len[doc_key]

    def scores(self, query: str, key_to_pos: dict[str, int], n: int) -> np.ndarray:
        """BM25 scores for ``query`` as an array aligned with matrix rows.

        ``key_to_pos`` maps doc_key to row position; ``n`` is the row count.
        """
        out = np.zeros(n, dtype=np.float64)
        total = len(self.doc_terms)
        if total == 0:
            return out
        avgdl = max(self.avg_doc_len, 1e-9)
        for term in set(tokenize(query)):
            bucket = self.postings.get(term)
            if not bucket:
                continue
            df = len(bucket)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for doc_key, tf in bucket.items():
                pos = key_to_pos.get(doc_key)
                if pos is None:
                    continue
                dl = self.doc_len[doc_key]
                out[pos] += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
        return out


def rrf_fuse(rankings: list[np.ndarray], n: int, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of result orderings.

    Each ranking is an array of row positions, best first. Returns a fused
    score per row: ``sum over rankings of 1 / (k + rank)``.
    """
    fused = np.zeros(n, dtype=np.float64)
    for order in rankings:
        for rank, pos in enumerate(order):
            fused[pos] += 1.0 / (k + rank + 1)
    return fused
