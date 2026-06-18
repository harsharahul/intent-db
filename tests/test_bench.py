"""Benchmark harness: metrics (TDD) + an end-to-end smoke run."""

import math

import pytest

from bench import metrics


# ---- nDCG ------------------------------------------------------------------

def test_ndcg_gold_at_rank_1_is_perfect():
    assert metrics.ndcg_at_k(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1.0)


def test_ndcg_gold_at_rank_2():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert metrics.ndcg_at_k(["b", "a", "c"], {"a"}, k=10) == pytest.approx(
        1.0 / math.log2(3)
    )


def test_ndcg_gold_below_cutoff_is_zero():
    assert metrics.ndcg_at_k(["b", "c", "a"], {"a"}, k=2) == pytest.approx(0.0)


def test_ndcg_two_relevant_docs():
    # ranked a(rel) x b(rel): DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (both at top) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    got = metrics.ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=10)
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert got == pytest.approx(expected)


def test_ndcg_no_relevant_returns_zero():
    assert metrics.ndcg_at_k(["a", "b"], set(), k=10) == 0.0


# ---- reciprocal rank -------------------------------------------------------

def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["b", "a", "c"], {"a"}) == pytest.approx(0.5)


def test_reciprocal_rank_absent_is_zero():
    assert metrics.reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_rank_of():
    assert metrics.rank_of(["b", "a", "c"], "a") == 2
    assert metrics.rank_of(["b", "c"], "a") is None


# ---- p-MRR (paired reciprocal-rank delta, FollowIR-style) ------------------

def test_pmrr_delta_relevant_intent_ranks_higher_is_positive():
    # doc at rank 1 under its relevant intent, rank 3 under the other
    assert metrics.pmrr_delta(1, 3) == pytest.approx(1.0 - 1.0 / 3.0)


def test_pmrr_delta_symmetric_zero():
    assert metrics.pmrr_delta(2, 2) == 0.0


def test_pmrr_delta_absent_under_irrelevant_intent():
    # not retrieved under the irrelevant intent -> 1/inf -> 0
    assert metrics.pmrr_delta(1, None) == pytest.approx(1.0)


def test_pmrr_delta_wrong_way_is_negative():
    # ranks higher under the intent where it is NOT relevant -> bad
    assert metrics.pmrr_delta(3, 1) == pytest.approx(1.0 / 3.0 - 1.0)


# ---- robustness@k ----------------------------------------------------------

def test_robustness_is_mean_of_per_query_minimums():
    # query1 nDCGs across its intents: [1.0, 0.5] -> min 0.5
    # query2 nDCGs: [1.0, 1.0] -> min 1.0 ; mean(0.5, 1.0) = 0.75
    per_query = {"python": [1.0, 0.5], "java": [1.0, 1.0]}
    assert metrics.robustness(per_query) == pytest.approx(0.75)


# ---- bootstrap CI ----------------------------------------------------------

def test_bootstrap_ci_constant_input_is_degenerate():
    lo, hi = metrics.bootstrap_ci([0.4, 0.4, 0.4, 0.4], seed=0)
    assert lo == pytest.approx(0.4)
    assert hi == pytest.approx(0.4)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    lo, hi = metrics.bootstrap_ci(values, seed=0)
    assert 0.0 <= lo <= metrics.mean(values) <= hi <= 1.0


def test_bootstrap_ci_is_deterministic_under_seed():
    values = [0.1, 0.9, 0.3, 0.7, 0.5]
    assert metrics.bootstrap_ci(values, seed=42) == metrics.bootstrap_ci(values, seed=42)


# ---- paired delta CI (significance between two configs) ---------------------

def test_paired_delta_ci_constant_gap():
    a = [0.6] * 12
    b = [0.5] * 12
    mean_d, lo, hi = metrics.paired_delta_ci(a, b, seed=0)
    assert mean_d == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_paired_delta_ci_zero_when_identical():
    a = [0.3, 0.7, 0.5, 0.9]
    mean_d, lo, hi = metrics.paired_delta_ci(a, a, seed=0)
    assert mean_d == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_delta_ci_detects_significant_improvement():
    # a is reliably above b -> the delta CI excludes zero
    a = [0.9, 0.8, 0.95, 0.85, 0.9, 0.88, 0.92, 0.87]
    b = [0.4, 0.5, 0.45, 0.5, 0.42, 0.48, 0.4, 0.5]
    mean_d, lo, hi = metrics.paired_delta_ci(a, b, seed=0)
    assert mean_d > 0
    assert lo > 0  # significant: the whole interval is above zero


def test_paired_delta_ci_is_paired_not_independent():
    # pairing matters: same values, shuffled, give a different (here zero) delta
    a = [0.1, 0.9]
    b = [0.9, 0.1]
    mean_d, _, _ = metrics.paired_delta_ci(a, b, seed=0)
    assert mean_d == pytest.approx(0.0)


# ---- dataset ---------------------------------------------------------------

def test_dataset_has_enough_paired_cases():
    from bench import dataset

    assert len(dataset.CASES) >= 50
    # every case references a real doc and a real intent
    doc_keys = {key for key, _ in dataset.DOCS}
    intent_names = {name for name, _, _ in dataset.INTENTS}
    for query, intent, gold in dataset.CASES:
        assert gold in doc_keys, f"unknown gold {gold!r}"
        assert intent in intent_names, f"unknown intent {intent!r}"


def test_paired_cases_keeps_only_multi_intent_queries():
    from bench import harness

    cases = [
        ("python", "coding", "py-lang"),
        ("python", "wildlife", "py-snake"),
        ("espresso crema", "culinary", "espresso"),  # single intent -> dropped
    ]
    paired = harness.paired_cases(cases)
    assert {(q, i, g) for q, i, g in paired} == {
        ("python", "coding", "py-lang"),
        ("python", "wildlife", "py-snake"),
    }


def test_hard_dataset_is_a_shared_topic_grid():
    from bench import dataset_hard as dh

    # every topic has exactly one doc per doc-type, every case is paired
    assert len(dh.DOCS) == len(dh.GRID) * len(dh.DOC_TYPES)
    by_query: dict[str, set] = {}
    for query, intent, gold in dh.CASES:
        by_query.setdefault(query, set()).add(intent)
    # each topic query appears under all four pragmatic intents
    assert all(len(intents) == len(dh.DOC_TYPES) for intents in by_query.values())
    assert len(dh.CASES) >= 40


def test_train_and_eval_queries_are_disjoint():
    from bench import dataset_hard as dh

    eval_q = {q for q, _, _ in dh.CASES}
    train_q = {q for q, _, _ in dh.TRAIN_CASES}
    assert eval_q.isdisjoint(train_q)


def test_sample_feedback_filters_by_intent_and_is_deterministic():
    from bench import dataset_hard as dh
    from bench import harness

    pairs = harness.sample_feedback(dh, "tutorial", n=5, seed=0)
    assert len(pairs) == 5
    assert all(gold.endswith("-tutorial") for _q, gold in pairs)
    assert harness.sample_feedback(dh, "tutorial", n=5, seed=0) == pairs  # deterministic
    # different intent -> different golds
    ref = harness.sample_feedback(dh, "reference", n=5, seed=0)
    assert all(gold.endswith("-reference") for _q, gold in ref)


def test_sample_feedback_caps_at_available():
    from bench import dataset_hard as dh
    from bench import harness

    available = sum(1 for _q, i, _g in dh.TRAIN_CASES if i == "concept")
    pairs = harness.sample_feedback(dh, "concept", n=10_000, seed=1)
    assert len(pairs) == available


def test_dataset_is_genuinely_paired():
    # at least some query strings appear under >1 intent with different gold
    from bench import dataset

    by_query: dict[str, set] = {}
    for query, intent, gold in dataset.CASES:
        by_query.setdefault(query, set()).add(gold)
    paired = [q for q, golds in by_query.items() if len(golds) > 1]
    assert len(paired) >= 5


# ---- end-to-end harness (hashing embedder; deterministic, no network) ------

@pytest.fixture(scope="module")
def built_db():
    from bench import harness

    db = harness.build_db("hashing:dim=512")
    yield db
    db.close()


def test_harness_runs_a_config(built_db):
    from bench import dataset, harness

    rows = harness.run_config(built_db, dataset.CASES, harness.CONFIGS["full"])
    assert len(rows) == len(dataset.CASES)
    assert all("ranked" in r and "gold" in r for r in rows)


def test_full_stack_beats_plain_cosine(built_db):
    # the headline claim, as a regression guard
    from bench import dataset, harness

    plain = harness.evaluate(built_db, dataset.CASES, harness.CONFIGS["plain"])
    full = harness.evaluate(built_db, dataset.CASES, harness.CONFIGS["full"])
    assert full["ndcg@10"] > plain["ndcg@10"]
    assert full["top1"] > plain["top1"]
