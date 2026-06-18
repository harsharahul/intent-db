"""Learned fusion: per-intent signal weights from relevance feedback.

IntentDB blends three dense signals (lensed, affinity, base) with fixed
default weights. Bruch, Gai & Ingber (ACM TOIS 2023) showed that a tuned
convex combination of normalized scores beats rank-only fusion in- and
out-of-domain and is sample-efficient, a small number of labeled
examples suffices because there is only one parameter per signal.

This module learns those weights from recorded feedback: pairs of
(signal vector of a useful document, signal vector of a non-useful
document) for the same query. A tiny logistic regression on the pairwise
differences (Bradley-Terry style) yields weights, which are clipped to be
non-negative and normalized to sum to 1, keeping the result a convex
combination on the same scale as the defaults, so learned and default
intents stay comparable.
"""

from __future__ import annotations

import numpy as np

#: order of signals in the weight vectors handled here
SIGNALS = ("lensed", "affinity", "base")

#: minimum preference pairs before learning is trusted at all
MIN_PAIRS = 10


def learn_weights(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    defaults: dict[str, float],
    min_pairs: int = MIN_PAIRS,
    l2: float = 0.05,
    iterations: int = 400,
    learning_rate: float = 0.5,
) -> dict[str, float] | None:
    """Fit convex fusion weights from preference pairs.

    Each pair is ``(signals_of_useful_doc, signals_of_other_doc)`` for the
    same query, with signals ordered as :data:`SIGNALS`. Returns a weight
    dict, or ``None`` when there is not enough (or degenerate) data, the
    caller keeps the defaults in that case.

    The model is logistic regression on score differences with L2 pull
    toward the default weights: maximize
    ``sigma(w . (s_useful - s_other))`` over pairs.
    """
    if len(pairs) < min_pairs:
        return None
    x = np.asarray([p - n for p, n in pairs], dtype=np.float64)
    if not np.isfinite(x).all() or np.allclose(x, 0):
        return None

    w0 = np.array([defaults[s] for s in SIGNALS], dtype=np.float64)
    w = w0.copy()
    n = len(x)
    for _ in range(iterations):
        margins = x @ w
        sig = 1.0 / (1.0 + np.exp(-np.clip(margins, -30, 30)))
        grad = x.T @ (sig - 1.0) / n + l2 * (w - w0)
        w -= learning_rate * grad

    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return None
    w /= total
    return {s: round(float(v), 6) for s, v in zip(SIGNALS, w)}


def build_preference_pairs(
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross every useful-doc signal vector with every non-useful one."""
    return [(p, n) for p in positives for n in negatives]
