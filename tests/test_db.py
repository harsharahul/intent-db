import numpy as np
import pytest

from intentdb import IntentDB

# A small corpus around an ambiguous term: "python".
CORPUS = [
    (
        "python-lang",
        "Python is a programming language; write code, functions, and modules, "
        "then debug your program.",
        {"topic": "software"},
    ),
    (
        "python-snake",
        "The python is a large snake, a reptile that lives in jungle habitats "
        "and hunts wildlife at night.",
        {"topic": "nature"},
    ),
    (
        "pip-doc",
        "Use pip to install python packages and manage code dependencies for "
        "your programming projects.",
        {"topic": "software"},
    ),
    (
        "zoo-doc",
        "At the zoo you can see a snake exhibit with a reptile house full of "
        "jungle animals and wildlife.",
        {"topic": "nature"},
    ),
]


@pytest.fixture()
def db(tmp_path):
    db = IntentDB(tmp_path / "test.intentdb", embedder="hashing:dim=512")
    for key, text, meta in CORPUS:
        db.add(text, doc_key=key, metadata=meta)
    db.register_intent(
        "coding",
        description="software programming, source code, debugging programs",
        exemplars=["how do I write code", "install a package", "debug my program"],
    )
    db.register_intent(
        "wildlife",
        description="animals, reptiles, snakes, jungle habitats and nature",
        exemplars=["what do snakes eat", "jungle animal habitats"],
    )
    yield db
    db.close()


def test_plain_query_returns_results(db):
    hits = db.query("python", k=2, auto_intent=False)
    assert len(hits) == 2
    assert hits[0].intent is None
    assert hits[0].score >= hits[1].score


def test_intent_changes_ranking(db):
    """The headline behavior: same query, different intent, different top hit."""
    coding = db.query("python", intent="coding", k=4)
    wildlife = db.query("python", intent="wildlife", k=4)
    assert coding[0].doc_key == "python-lang"
    assert wildlife[0].doc_key == "python-snake"


def test_intent_is_inferred_from_query(db):
    hits = db.query("python debug program code", k=2)
    assert hits[0].intent == "coding"
    assert hits[0].intent_inferred is True

    hits = db.query("python snake jungle reptile", k=2)
    assert hits[0].intent == "wildlife"
    assert hits[0].doc_key == "python-snake"


def test_unknown_intent_raises(db):
    with pytest.raises(KeyError):
        db.query("python", intent="nope")


def test_metadata_filter(db):
    hits = db.query(
        "python", k=4, auto_intent=False, where=lambda m: m.get("topic") == "nature"
    )
    assert {h.doc_key for h in hits} == {"python-snake", "zoo-doc"}


def test_persistence_across_reopen(db, tmp_path):
    path = db.store.path
    db.close()
    reopened = IntentDB(path)  # embedder spec restored from the store
    assert reopened.stats()["documents"] == len(CORPUS)
    assert set(reopened.stats()["intents"]) == {"coding", "wildlife"}
    hits = reopened.query("python", intent="wildlife", k=1)
    assert hits[0].doc_key == "python-snake"
    reopened.close()


def test_embedder_dim_mismatch_rejected(db):
    path = db.store.path
    db.close()
    with pytest.raises(ValueError, match="dim"):
        IntentDB(path, embedder="hashing:dim=128")


def test_upsert_replaces_document(db):
    db.add("Completely new text about cooking pasta.", doc_key="python-lang")
    assert db.stats()["documents"] == len(CORPUS)
    doc = db.get("python-lang")
    assert "pasta" in doc["text"]
    # the replaced doc should no longer top the coding query
    hits = db.query("python programming code", intent="coding", k=1)
    assert hits[0].doc_key != "python-lang"


def test_delete_document(db):
    assert db.delete("zoo-doc") is True
    assert db.delete("zoo-doc") is False
    assert db.stats()["documents"] == len(CORPUS) - 1
    hits = db.query("zoo reptile house", k=4, auto_intent=False)
    assert all(h.doc_key != "zoo-doc" for h in hits)


def test_delete_many(db):
    removed = db.delete_many(["zoo-doc", "pip-doc", "does-not-exist"])
    assert set(removed) == {"zoo-doc", "pip-doc"}  # only existing keys reported
    assert db.stats()["documents"] == len(CORPUS) - 2

    # the in-memory mirror is compacted to match (single-pass mask)
    remaining = len(CORPUS) - 2
    assert db._matrix.shape[0] == remaining
    assert len(db._keys) == len(db._ids) == len(db._texts) == remaining
    for aff in db._intent_affinities.values():
        assert aff.shape[0] == remaining

    # deleted docs no longer surface, and ranking still holds
    hits = db.query("python", intent="coding", k=4)
    keys = {h.doc_key for h in hits}
    assert {"zoo-doc", "pip-doc"}.isdisjoint(keys)
    assert hits[0].doc_key == "python-lang"
    assert all(np.isfinite(h.score) for h in hits)

    assert db.delete_many(["zoo-doc"]) == []  # idempotent


def test_add_many_new_and_replace_in_one_batch(db):
    n0 = db.stats()["documents"]
    keys = db.add_many(
        [
            ("A fresh note about gardening tomatoes outdoors.", "garden-doc", {"topic": "home"}),
            ("Replacement text: pip builds and installs wheels.", "pip-doc", {"topic": "software"}),
        ]
    )
    assert keys == ["garden-doc", "pip-doc"]
    assert db.stats()["documents"] == n0 + 1  # one new, one replaced

    # matrix and affinity arrays grew exactly once, staying aligned
    assert db._matrix.shape[0] == n0 + 1
    assert len(db._keys) == n0 + 1
    for aff in db._intent_affinities.values():
        assert aff.shape[0] == n0 + 1

    assert db.get("pip-doc")["text"].startswith("Replacement")
    hits = db.query("gardening tomatoes", k=2, auto_intent=False)
    assert any(h.doc_key == "garden-doc" for h in hits)


def test_busy_timeout_pragma(db):
    assert db.store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_batch_ops_cross_chunk_boundary(tmp_path):
    # Exercise the >900 bound-variable chunking in ids_for_keys / delete_documents.
    db = IntentDB(tmp_path / "big.intentdb", embedder="hashing:dim=64")
    try:
        n = 950
        db.add_many([(f"document number {i} body text", f"k{i}", {}) for i in range(n)])
        assert db.stats()["documents"] == n
        assert db._matrix.shape == (n, 64)

        to_delete = [f"k{i}" for i in range(920)] + ["missing-a", "missing-b"]
        removed = db.delete_many(to_delete)
        assert set(removed) == {f"k{i}" for i in range(920)}
        assert db.stats()["documents"] == n - 920
        assert db._matrix.shape[0] == n - 920
        # a surviving doc is still retrievable
        assert db.get("k949") is not None
    finally:
        db.close()


def test_add_after_intent_registration_is_indexed(db):
    db.add(
        "A cobra is a venomous snake found in jungle regions.",
        doc_key="cobra",
        metadata={"topic": "nature"},
    )
    hits = db.query("venomous cobra snake", intent="wildlife", k=1)
    assert hits[0].doc_key == "cobra"
    assert hits[0].intent_affinity is not None and hits[0].intent_affinity > 0


def test_remove_intent(db):
    assert db.remove_intent("wildlife") is True
    assert "wildlife" not in db.stats()["intents"]
    with pytest.raises(KeyError):
        db.query("python", intent="wildlife")


def test_empty_db_query(tmp_path):
    db = IntentDB(tmp_path / "empty.intentdb")
    assert db.query("anything") == []
    db.close()


def test_explain(db):
    out = db.explain("debug my python program")
    assert out["inferred_intent"] == "coding"
    assert set(out["intent_scores"]) == {"coding", "wildlife"}


def test_scores_are_finite(db):
    for intent in (None, "coding", "wildlife"):
        for hit in db.query("python snake code", intent=intent, k=4):
            assert np.isfinite(hit.score)
