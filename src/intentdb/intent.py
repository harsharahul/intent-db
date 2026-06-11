"""The intent dimension: intents, lenses, and intent inference.

An :class:`Intent` is a named retrieval purpose ("debugging", "cooking",
"legal research", ...) described by free text and optional exemplar
queries. From those we derive two things:

1. an **intent vector** ``t`` — the unit-norm centroid of the embedded
   description and exemplars. It lives in the same space as documents, so
   we can measure how much any document or query *belongs* to the intent
   (its *affinity*).

2. an **intent lens** — a per-dimension gate ``g`` over the embedding
   space. Applying the lens re-weights embedding dimensions that are
   characteristic of the intent, so the *effective vectorization* of both
   queries and documents changes when the intent is active. This is a
   diagonal (Mahalanobis-style) metric learned from the intent's examples:
   dimensions where the exemplars agree strongly (high mean magnitude, low
   variance) are amplified; the rest stay at weight 1.

The lensed similarity has a cheap closed form. With gate ``g``::

    sim_lens(q, d) = <q*g, d*g> / ||q*g||  =  <q * g^2, d> / ||q*g||

i.e. cosine in the lensed space on the query side, while the document side
keeps its base (unit) norm. The asymmetry is deliberate: re-normalizing
documents in the lensed space would *penalize* documents rich in
intent-relevant content (their lensed norm grows with every
intent-characteristic term they contain). With this form, query-document
overlap on intent-characteristic dimensions is amplified, overlap on
incidental dimensions is not — and the whole collection is scored with a
single matrix-vector product, since only the query is transformed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Default strength of the lens: gate values range in [1, 1 + LENS_STRENGTH].
DEFAULT_LENS_STRENGTH = 4.0


@dataclass
class IntentLens:
    """A per-dimension gate over the embedding space."""

    gate: np.ndarray  # shape (dim,), values >= 1

    @property
    def gate_sq(self) -> np.ndarray:
        return self.gate * self.gate

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Gate vectors (no re-normalization). Works on 1-D or 2-D input."""
        return vectors * self.gate

    def lensed_norms(self, vectors: np.ndarray) -> np.ndarray:
        """Norms of gated vectors; ``vectors`` is (n, dim) or (dim,)."""
        gated = self.apply(vectors)
        if gated.ndim == 1:
            return np.linalg.norm(gated)
        return np.linalg.norm(gated, axis=1)

    @staticmethod
    def fit(
        sample_vectors: np.ndarray,
        strength: float = DEFAULT_LENS_STRENGTH,
    ) -> "IntentLens":
        """Learn a gate from example vectors of an intent.

        Uses a diagonal Fisher-style relevance score: for each dimension,
        ``relevance_i = mean_i**2 / (var_i + eps)``. Dimensions that are
        consistently active across the intent's examples score high. The
        scores are normalized to [0, 1] and mapped to gates in
        ``[1, 1 + strength]``.

        With a single example the variance is zero everywhere and the score
        gracefully degrades to the squared magnitude profile of that vector.
        """
        mat = np.atleast_2d(np.asarray(sample_vectors, dtype=np.float64))
        mean = mat.mean(axis=0)
        var = mat.var(axis=0)
        eps = 1e-4
        relevance = (mean * mean) / (var + eps)
        peak = relevance.max()
        if peak <= 0:
            gate = np.ones(mat.shape[1])
        else:
            gate = 1.0 + strength * (relevance / peak)
        return IntentLens(gate=gate.astype(np.float32))


@dataclass
class Intent:
    """A named retrieval intent with its vector and lens.

    ``instruction`` is an optional natural-language task instruction (in the
    style of instruction-finetuned embedders such as INSTRUCTOR or
    nomic-embed). When the database's embedder supports instructions, the
    query is re-embedded conditioned on the active intent's instruction —
    the query's vectorization itself changes with intent. Defaults to the
    intent's description.
    """

    name: str
    description: str
    exemplars: list[str] = field(default_factory=list)
    instruction: str | None = None
    vector: np.ndarray | None = None  # unit-norm centroid, shape (dim,)
    lens: IntentLens | None = None
    lens_strength: float = DEFAULT_LENS_STRENGTH

    @staticmethod
    def build(
        name: str,
        description: str,
        exemplars: list[str],
        embed_batch,
        instruction: str | None = None,
        lens_strength: float = DEFAULT_LENS_STRENGTH,
    ) -> "Intent":
        """Embed the description/exemplars and fit the vector and lens."""
        texts = [description] + list(exemplars)
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise ValueError(f"intent {name!r} needs a description or exemplars")
        mat = np.asarray(embed_batch(texts), dtype=np.float64)
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return Intent(
            name=name,
            description=description,
            exemplars=list(exemplars),
            instruction=instruction if instruction is not None else description,
            vector=centroid.astype(np.float32),
            lens=IntentLens.fit(mat, strength=lens_strength),
            lens_strength=lens_strength,
        )

    def affinity(self, vectors: np.ndarray) -> np.ndarray:
        """Cosine affinity of unit-norm vectors to this intent (1-D or 2-D)."""
        return np.asarray(vectors) @ self.vector


def infer_intent(
    query_vector: np.ndarray,
    intents: list[Intent],
    threshold: float = 0.08,
) -> tuple[Intent | None, dict[str, float]]:
    """Pick the most plausible intent for a query, or ``None``.

    Returns the winning intent (if its affinity clears ``threshold`` and
    beats the runner-up meaningfully) plus the full affinity map so callers
    can expose the classifier's view of the query.
    """
    if not intents:
        return None, {}
    scores = {i.name: float(i.affinity(query_vector)) for i in intents}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]
    if best_score < threshold:
        return None, scores
    best = next(i for i in intents if i.name == best_name)
    return best, scores
