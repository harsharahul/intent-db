"""Relevance feedback recording and learned fusion weights."""

import numpy as np
import pytest

from intentdb import IntentDB
from intentdb.fusion import SIGNALS, build_preference_pairs, learn_weights

DEFAULTS = {"lensed": 0.6, "affinity": 0.25, "base": 0.15}


# ---- unit: the learner -------------------------------------------------------


def test_learn_weights_needs_enough_pairs():
    pairs = [(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))] * 3
    assert learn_weights(pairs, DEFAULTS, min_pairs=10) is None


def test_learn_weights_favors_separating_signal():
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(40):
        # affinity separates useful from useless; lensed is pure noise
        pos = np.array([rng.normal(), 0.8 + 0.1 * rng.normal(), 0.2])
        neg = np.array([rng.normal(), 0.1 + 0.1 * rng.normal(), 0.2])
        pairs.append((pos, neg))
    w = learn_weights(pairs, DEFAULTS)
    assert w is not None
    assert w["affinity"] > w["lensed"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in w.values())


def test_learn_weights_degenerate_data():
    z = np.zeros(3)
    assert learn_weights([(z, z)] * 20, DEFAULTS) is None


def test_build_preference_pairs_cross_product():
    p = [np.ones(3)] * 2
    n = [np.zeros(3)] * 3
    assert len(build_preference_pairs(p, n)) == 6


# ---- integration: the full loop ----------------------------------------------


@pytest.fixture()
def db(tmp_path):
    db = IntentDB(tmp_path / "fb.intentdb", embedder="hashing:dim=512")
    # "guide" matches research queries lexically but is shallow marketing;
    # "deep" docs carry the intent's vocabulary and are the useful ones
    db.add(
        "research guide research guide research methods overview brochure",
        doc_key="shallow",
    )
    db.add(
        "rigorous experiment methodology with controls and statistics",
        doc_key="deep1",
    )
    db.add(
        "study design methodology statistics hypothesis experiment controls",
        doc_key="deep2",
    )
    for i in range(4):
        db.add(f"cafeteria menu for week {i} soup salad sandwiches", doc_key=f"m{i}")
    db.register_intent(
        "research",
        description="rigorous scientific methodology, experiments, statistics",
        exemplars=["experiment design", "statistical controls"],
    )
    yield db
    db.close()


FEEDBACK_QUERIES = [
    "research methods",
    "research methodology",
    "how to research",
    "research approach",
    "research process",
]


def test_feedback_recorded_and_counted(db):
    db.record_feedback("research methods", "deep1", useful=True, intent="research")
    db.record_feedback("research methods", "shallow", useful=False, intent="research")
    assert db.stats()["feedback"] == 2


def test_learn_returns_none_without_enough_feedback(db):
    db.record_feedback("research methods", "deep1", useful=True, intent="research")
    out = db.learn_fusion_weights()
    assert out == {"research": None}
    assert db.stats()["learned_intents"] == []


def test_full_learning_loop_changes_weights_and_persists(db, tmp_path):
    for q in FEEDBACK_QUERIES:
        db.record_feedback(q, "deep1", useful=True, intent="research")
        db.record_feedback(q, "deep2", useful=True, intent="research")
        db.record_feedback(q, "shallow", useful=False, intent="research")
    out = db.learn_fusion_weights(intent="research")
    learned = out["research"]
    assert learned is not None
    assert set(learned) == set(SIGNALS)
    assert abs(sum(learned.values()) - 1.0) < 1e-5
    assert db.stats()["learned_intents"] == ["research"]

    # learned weights are applied automatically and survive reopen
    path = db.store.path
    db.close()
    reopened = IntentDB(path)
    assert reopened.fusion_weights()["research"] == learned
    reopened.close()


def test_learned_weights_improve_ranking(db):
    """Default weights may rank the lexically-loaded 'shallow' doc highly;
    after feedback says deep docs are what 'research' means, the useful
    docs must outrank it."""
    for q in FEEDBACK_QUERIES:
        db.record_feedback(q, "deep1", useful=True, intent="research")
        db.record_feedback(q, "deep2", useful=True, intent="research")
        db.record_feedback(q, "shallow", useful=False, intent="research")
    assert db.learn_fusion_weights(intent="research")["research"] is not None

    keys = [h.doc_key for h in db.query("research methods", intent="research", k=3)]
    pos_shallow = keys.index("shallow") if "shallow" in keys else 99
    assert keys.index("deep2") < pos_shallow or keys.index("deep1") < pos_shallow
    assert keys[0] in ("deep1", "deep2")


def test_explicit_weights_override_learned(db):
    for q in FEEDBACK_QUERIES:
        db.record_feedback(q, "deep1", useful=True, intent="research")
        db.record_feedback(q, "shallow", useful=False, intent="research")
    db.learn_fusion_weights(intent="research")
    # forcing all weight onto base similarity must change the blend
    hits = db.query(
        "research methods",
        intent="research",
        k=1,
        weights={"lensed": 0.0, "affinity": 0.0, "base": 1.0},
    )
    top = hits[0]
    assert abs(top.score - top.base_score) < 1e-9


def test_implicit_negatives_when_only_positives(db):
    for q in FEEDBACK_QUERIES:
        db.record_feedback(q, "deep1", useful=True, intent="research")
        db.record_feedback(q, "deep2", useful=True, intent="research")
    out = db.learn_fusion_weights(intent="research")
    assert out["research"] is not None  # implicit negatives were sampled


def test_remove_intent_drops_learned_weights(db):
    for q in FEEDBACK_QUERIES:
        db.record_feedback(q, "deep1", useful=True, intent="research")
        db.record_feedback(q, "shallow", useful=False, intent="research")
    db.learn_fusion_weights(intent="research")
    db.remove_intent("research")
    assert db.fusion_weights() == {}


def test_learn_unknown_intent_raises(db):
    with pytest.raises(KeyError):
        db.learn_fusion_weights(intent="nope")
