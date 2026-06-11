"""Intent mining: discover candidate intents from the query log.

Every query the database answers is logged (text + which intent, if any,
was active). Queries that arrive without a confident intent are the
interesting ones — recurring themes among them are intents the user has
but never declared. ``mine_intents`` clusters their embeddings with
spherical k-means and returns, per cluster, representative queries ready
to be used as ``exemplars`` for :meth:`IntentDB.register_intent` (an LLM
or a human picks the name).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntentSuggestion:
    """A mined cluster of queries that could become a registered intent."""

    size: int
    coherence: float  # mean cosine of members to their centroid, 0..1
    exemplars: list[str]  # representative queries, closest to centroid first

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "coherence": round(self.coherence, 4),
            "exemplars": self.exemplars,
        }


def _spherical_kmeans(
    matrix: np.ndarray, k: int, iterations: int = 25, seed: int = 0
) -> np.ndarray:
    """Cluster unit-norm rows by cosine; returns a label per row."""
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=np.int64)
    for it in range(iterations):
        sims = matrix @ centroids.T  # (n, k)
        new_labels = sims.argmax(axis=1)
        if it > 0 and (new_labels == labels).all():
            break
        labels = new_labels
        for j in range(k):
            members = matrix[labels == j]
            if len(members) == 0:
                # re-seed an empty cluster on the point worst-served so far
                worst = sims.max(axis=1).argmin()
                centroids[j] = matrix[worst]
                continue
            c = members.mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[j] = c / norm if norm > 0 else c
    return labels


def mine_intents(
    texts: list[str],
    vectors: np.ndarray,
    k: int = 3,
    min_cluster_size: int = 3,
    exemplars_per_intent: int = 5,
    seed: int = 0,
) -> list[IntentSuggestion]:
    """Cluster query texts into up to ``k`` candidate intents.

    Duplicate texts are collapsed before clustering (a query asked ten
    times should weigh as a theme, not dominate a centroid by sheer count
    — its presence is already captured by the dedup'd member).
    """
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        key = t.strip().lower()
        if key and key not in seen:
            seen[key] = i
    idx = sorted(seen.values())
    if len(idx) < max(min_cluster_size, 2):
        return []
    uniq_texts = [texts[i] for i in idx]
    mat = vectors[idx]
    # drop zero vectors (empty queries) — they carry no theme
    norms = np.linalg.norm(mat, axis=1)
    keep = norms > 0
    uniq_texts = [t for t, m in zip(uniq_texts, keep) if m]
    mat = mat[keep]
    if len(uniq_texts) < max(min_cluster_size, 2):
        return []

    k = min(k, len(uniq_texts))
    labels = _spherical_kmeans(mat, k, seed=seed)

    suggestions: list[IntentSuggestion] = []
    for j in range(k):
        member_idx = np.flatnonzero(labels == j)
        if len(member_idx) < min_cluster_size:
            continue
        members = mat[member_idx]
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        sims = members @ centroid
        order = np.argsort(sims)[::-1]
        suggestions.append(
            IntentSuggestion(
                size=int(len(member_idx)),
                coherence=float(sims.mean()),
                exemplars=[
                    uniq_texts[member_idx[i]] for i in order[:exemplars_per_intent]
                ],
            )
        )
    suggestions.sort(key=lambda s: (s.size, s.coherence), reverse=True)
    return suggestions
