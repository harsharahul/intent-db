import numpy as np

from intentdb import IntentDB
from intentdb.embedders import HashingEmbedder
from intentdb.mining import mine_intents

COOKING = [
    "recipe for tomato pasta dinner",
    "how long to roast chicken dinner",
    "best recipe for vegetable soup",
    "easy dinner recipe ideas tonight",
]
CODING = [
    "python function throws exception",
    "debug stack trace in my function",
    "why does my python code crash",
    "fix exception in program code",
]


def _embed(texts):
    e = HashingEmbedder(dim=512)
    return np.stack([e.embed(t) for t in texts])


def test_mine_intents_finds_two_themes():
    texts = COOKING + CODING
    suggestions = mine_intents(texts, _embed(texts), k=2, min_cluster_size=3)
    assert len(suggestions) == 2
    joined = [" ".join(s.exemplars) for s in suggestions]
    # each cluster should be dominated by one theme
    assert any("recipe" in j and "exception" not in j for j in joined)
    assert any(("exception" in j or "debug" in j) and "recipe" not in j for j in joined)


def test_mine_intents_deduplicates():
    texts = ["same query"] * 50 + COOKING
    suggestions = mine_intents(texts, _embed(texts), k=2, min_cluster_size=3)
    for s in suggestions:
        assert s.size <= len(COOKING) + 1  # dupes collapsed to one member


def test_mine_intents_too_few_queries():
    texts = ["one", "two"]
    assert mine_intents(texts, _embed(texts), k=3, min_cluster_size=3) == []


def test_query_log_and_suggestions_end_to_end(tmp_path):
    db = IntentDB(tmp_path / "m.intentdb")
    db.add("pasta and soup recipes for dinner", doc_key="cook")
    db.add("debugging python exceptions and crashes", doc_key="code")

    for q in COOKING + CODING:
        db.query(q, k=1)
    assert db.stats()["logged_queries"] == len(COOKING) + len(CODING)

    suggestions = db.suggest_intents(k=2, min_cluster_size=3)
    assert len(suggestions) == 2
    assert all(len(s.exemplars) >= 3 for s in suggestions)

    # log=False keeps automated traffic out of the log
    before = db.stats()["logged_queries"]
    db.query("polling query", log=False)
    assert db.stats()["logged_queries"] == before
    db.close()


def test_explicit_intent_queries_excluded_from_mining(tmp_path):
    db = IntentDB(tmp_path / "x.intentdb")
    db.add("pasta recipes", doc_key="cook")
    db.register_intent("cooking", description="food recipes cooking dinner")
    for q in COOKING:
        db.query(q, intent="cooking", k=1)  # explicitly declared
    assert db.suggest_intents(k=2, min_cluster_size=2) == []
    db.close()


def test_suggestions_persist_in_log_across_reopen(tmp_path):
    path = tmp_path / "p.intentdb"
    db = IntentDB(path)
    db.add("doc", doc_key="d")
    for q in CODING:
        db.query(q, k=1)
    db.close()
    db = IntentDB(path)
    assert db.stats()["logged_queries"] == len(CODING)
    assert len(db.suggest_intents(k=1, min_cluster_size=3)) == 1
    db.close()
