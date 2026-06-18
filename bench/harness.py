"""Run retrieval configurations over the paired-intent dataset.

A *config* is a named bundle of ``IntentDB.query`` keyword arguments (plus
an optional weight override) describing one point in the ablation grid:
plain cosine, intent lens only, affinity only, the full blend, and the
full blend plus hybrid BM25 and/or a cross-encoder reranker.

:func:`run_config` returns the raw ranked lists per case; :func:`evaluate`
aggregates them into the metric table used by ``bench/run.py``.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from typing import Any

from intentdb import IntentDB

from . import dataset, metrics

# Each config maps to query() kwargs. ``weights`` (when present) overrides
# the lensed/affinity/base blend to isolate a single signal.
CONFIGS: dict[str, dict[str, Any]] = {
    # plain vector search: no intent at all (the "normal vector DB" baseline)
    "plain": {"auto_intent": False},
    # intent inferred from the query text, never declared
    "auto-intent": {"auto_intent": True},
    # explicit intent, but only one signal active
    "lens-only": {"use_intent": True, "weights": {"lensed": 1.0, "affinity": 0.0, "base": 0.0}},
    "affinity-only": {"use_intent": True, "weights": {"lensed": 0.0, "affinity": 1.0, "base": 0.0}},
    # explicit intent, default blend, and the optional upgrades
    "full": {"use_intent": True},
    "full+hybrid": {"use_intent": True, "hybrid": True},
    "full+rerank": {"use_intent": True, "rerank": True},
    "full+hybrid+rerank": {"use_intent": True, "hybrid": True, "rerank": True},
}

#: configs whose name implies a cross-encoder (skipped if flashrank absent)
RERANK_CONFIGS = {name for name in CONFIGS if "rerank" in name}

K = 10


def build_db(embedder: str, path: str | None = None, data=dataset) -> IntentDB:
    """Build a fresh IntentDB over a benchmark corpus and its intents.

    ``data`` is a dataset module exposing ``DOCS`` and ``INTENTS`` (the
    easy track by default; pass ``bench.dataset_hard`` for the hard track).
    """
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".intentdb")
        os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    db = IntentDB(path, embedder=embedder)
    db.add_many([(text, key, None) for key, text in data.DOCS])
    for name, description, exemplars in data.INTENTS:
        db.register_intent(name, description=description, exemplars=exemplars)
    return db


def sample_feedback(data, intent: str, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Sample up to ``n`` (query, relevant-doc) training pairs for an intent.

    Drawn from ``data.TRAIN_CASES`` (disjoint from the eval ``CASES``), this
    is the feedback signal a feedback-driven model — for example a per-intent
    low-rank adapter — fits on. Deterministic for a fixed ``seed``; sweeping
    ``n`` traces how retrieval improves with accumulated feedback.
    """
    import numpy as np

    pool = [(q, gold) for q, i, gold in data.TRAIN_CASES if i == intent]
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in sorted(idx)]


def paired_cases(cases) -> list:
    """Keep only cases whose query appears under more than one intent.

    These are the ambiguous queries where intent is load-bearing — the
    same query string has different gold documents per intent, so a single
    fixed ranking cannot satisfy all of them.
    """
    counts: dict[str, int] = defaultdict(int)
    for query, _intent, _gold in cases:
        counts[query] += 1
    return [c for c in cases if counts[c[0]] > 1]


def run_config(db: IntentDB, cases, config: dict[str, Any]) -> list[dict]:
    """Run every case under ``config``; return ranked lists + gold per case."""
    cfg = dict(config)
    use_intent = cfg.pop("use_intent", False)
    rows = []
    for query, intent, gold in cases:
        kwargs = dict(cfg)
        if use_intent:
            kwargs["intent"] = intent
        # retrieve the whole corpus so rank-based metrics see every doc
        hits = db.query(query, k=len(dataset.DOCS), log=False, **kwargs)
        rows.append(
            {
                "query": query,
                "intent": intent,
                "gold": gold,
                "ranked": [h.doc_key for h in hits],
            }
        )
    return rows


def evaluate(db: IntentDB, cases, config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate metrics for one config over ``cases``."""
    rows = run_config(db, cases, config)

    ndcgs = [metrics.ndcg_at_k(r["ranked"], {r["gold"]}, K) for r in rows]
    rrs = [metrics.reciprocal_rank(r["ranked"], {r["gold"]}) for r in rows]
    top1 = [1.0 if r["ranked"] and r["ranked"][0] == r["gold"] else 0.0 for r in rows]

    # robustness: each query's worst nDCG across the intents it appears under
    by_query: dict[str, list[float]] = defaultdict(list)
    for r, nd in zip(rows, ndcgs):
        by_query[r["query"]].append(nd)

    # p-MRR: for each (query, gold) the rank under its own intent vs. under
    # the other intents the same query appears with (where it is not gold)
    rank_by_query_intent: dict[tuple[str, str], list[str]] = {
        (r["query"], r["intent"]): r["ranked"] for r in rows
    }
    intents_per_query: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        intents_per_query[r["query"]].append(r["intent"])
    pmrr_vals = []
    for r in rows:
        rel_rank = metrics.rank_of(r["ranked"], r["gold"])
        for other in intents_per_query[r["query"]]:
            if other == r["intent"]:
                continue
            irr_rank = metrics.rank_of(
                rank_by_query_intent[(r["query"], other)], r["gold"]
            )
            pmrr_vals.append(metrics.pmrr_delta(rel_rank, irr_rank))

    nd_lo, nd_hi = metrics.bootstrap_ci(ndcgs)
    return {
        "top1": metrics.mean(top1),
        "ndcg@10": metrics.mean(ndcgs),
        "ndcg@10_ci": (nd_lo, nd_hi),
        "ndcg_per_case": ndcgs,  # aligned with `cases`, for paired comparisons
        "mrr": metrics.mean(rrs),
        "p-mrr": metrics.mean(pmrr_vals) if pmrr_vals else 0.0,
        "robustness@10": metrics.robustness(by_query),
        "n": len(rows),
    }
