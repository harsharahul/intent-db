"""Information-retrieval metrics for the paired-intent benchmark.

All functions take a ranked list of ``doc_key`` strings (best first) and a
set (or dict ``key -> grade``) of relevant keys. Binary relevance is the
default; graded relevance is supported by passing a dict.

Metrics:

- :func:`ndcg_at_k` — normalized discounted cumulative gain, the standard
  top-heavy ranking quality measure.
- :func:`reciprocal_rank` — 1 / rank of the first relevant document.
- :func:`pmrr_delta` — paired reciprocal-rank delta, the FollowIR-style
  signal of intent sensitivity: how much higher a document ranks under the
  intent that makes it relevant versus an intent that does not. Range
  ``[-1, 1]``; positive means the system correctly favors the document
  under its own intent.
- :func:`robustness` — InstructIR-style worst-case quality: the mean over
  queries of each query's *minimum* nDCG across its intents.
- :func:`bootstrap_ci` — percentile bootstrap confidence interval, so the
  headline numbers ship with uncertainty rather than as bare points.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np


def _grade(relevant, key: str) -> float:
    if isinstance(relevant, Mapping):
        return float(relevant.get(key, 0.0))
    return 1.0 if key in relevant else 0.0


def rank_of(ranked_keys: list[str], key: str) -> int | None:
    """1-indexed rank of ``key`` in ``ranked_keys``, or ``None`` if absent."""
    for i, k in enumerate(ranked_keys, 1):
        if k == key:
            return i
    return None


def dcg_at_k(ranked_keys: list[str], relevant, k: int) -> float:
    return sum(
        _grade(relevant, key) / math.log2(i + 2)
        for i, key in enumerate(ranked_keys[:k])
    )


def ndcg_at_k(ranked_keys: list[str], relevant, k: int = 10) -> float:
    """Normalized DCG@k. Returns 0.0 when nothing is relevant."""
    dcg = dcg_at_k(ranked_keys, relevant, k)
    if isinstance(relevant, Mapping):
        ideal_grades = sorted((float(g) for g in relevant.values()), reverse=True)
    else:
        ideal_grades = [1.0] * len(relevant)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_grades[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_keys: list[str], relevant) -> float:
    """1 / rank of the first relevant document (0.0 if none retrieved)."""
    for i, key in enumerate(ranked_keys, 1):
        if _grade(relevant, key) > 0:
            return 1.0 / i
    return 0.0


def pmrr_delta(rank_relevant: int | None, rank_irrelevant: int | None) -> float:
    """Paired reciprocal-rank delta for one document across two intents.

    ``rank_relevant`` is the document's rank under the intent for which it
    is relevant; ``rank_irrelevant`` under an intent for which it is not.
    ``None`` means the document was not retrieved (reciprocal rank 0).
    Returns ``1/rank_relevant - 1/rank_irrelevant`` in ``[-1, 1]``.
    """
    rr_rel = 1.0 / rank_relevant if rank_relevant else 0.0
    rr_irr = 1.0 / rank_irrelevant if rank_irrelevant else 0.0
    return rr_rel - rr_irr


def robustness(per_query_ndcgs: Mapping[str, Iterable[float]]) -> float:
    """Mean over queries of each query's minimum nDCG across its intents."""
    mins = [min(v) for v in (list(vals) for vals in per_query_ndcgs.values()) if v]
    return mean(mins)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def bootstrap_ci(
    values: list[float],
    statistic=mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times and returns
    the ``(alpha/2, 1-alpha/2)`` percentiles of the resampled statistic.
    Deterministic for a fixed ``seed``.
    """
    values = list(values)
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    stats = np.array([statistic(arr[row]) for row in idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_delta_ci(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on the *paired* per-case difference ``a - b``.

    ``values_a[i]`` and ``values_b[i]`` are the same case scored under two
    configurations. Returns ``(mean_delta, lo, hi)``; when the whole
    interval is above zero, A beats B significantly. Pairing is preserved
    (the difference is taken case-by-case before resampling), so this tests
    the per-case improvement, not a difference of two independent means.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired_delta_ci needs equal-length inputs")
    deltas = [a - b for a, b in zip(values_a, values_b)]
    lo, hi = bootstrap_ci(deltas, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return (mean(deltas), lo, hi)
