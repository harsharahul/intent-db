import numpy as np

from intentdb import IntentDB
from intentdb.lexical import BM25Index, rrf_fuse


def make_index(docs):
    idx = BM25Index()
    for key, text in docs:
        idx.add(key, text)
    return idx, {key: pos for pos, (key, _) in enumerate(docs)}


def test_term_frequency_ranks_higher():
    docs = [
        ("a", "redis redis redis cache"),
        ("b", "redis appears once in this much longer document about caching"),
    ]
    idx, k2p = make_index(docs)
    scores = idx.scores("redis", k2p, 2)
    assert scores[0] > scores[1] > 0


def test_rare_terms_weigh_more_than_common():
    docs = [
        ("common", "database database systems store data in a database"),
        ("rare", "the ECONNREFUSED error appears in database logs"),
    ]
    idx, k2p = make_index(docs)
    scores = idx.scores("database ECONNREFUSED", k2p, 2)
    assert scores[1] > scores[0]  # the doc with the rare exact term wins


def test_no_match_is_zero():
    idx, k2p = make_index([("a", "alpha beta")])
    assert idx.scores("zeta", k2p, 1)[0] == 0.0
    assert idx.scores("zeta", {}, 0).shape == (0,)


def test_remove_and_reindex():
    idx, k2p = make_index([("a", "alpha beta"), ("b", "alpha gamma")])
    idx.remove("a")
    assert len(idx) == 1
    assert idx.scores("beta", k2p, 2)[0] == 0.0
    # re-adding a key replaces its old postings
    idx.add("b", "delta only now")
    assert idx.scores("gamma", k2p, 2)[1] == 0.0
    assert idx.scores("delta", k2p, 2)[1] > 0


def test_rrf_fuse_prefers_agreement():
    # doc 0 is ranked first by both lists; doc 1 and 2 split the lists
    fused = rrf_fuse([np.array([0, 1, 2]), np.array([0, 2, 1])], 3)
    assert fused[0] > fused[1]
    assert fused[0] > fused[2]


def test_hybrid_query_finds_exact_term_dense_misses(tmp_path):
    """An identifier-style token: hybrid must surface the exact match."""
    db = IntentDB(tmp_path / "h.intentdb", embedder="hashing:dim=64")
    # tiny dim -> hash collisions make dense ranking unreliable; BM25 is exact
    db.add("error code XK9417Q raised by the billing service", doc_key="hit")
    for i in range(8):
        db.add(f"unrelated filler document number {i} about other services", doc_key=f"f{i}")
    hits = db.query("XK9417Q", k=1, hybrid=True, auto_intent=False)
    assert hits[0].doc_key == "hit"
    assert hits[0].lexical_score is not None and hits[0].lexical_score > 0
    db.close()


def test_hybrid_score_is_rrf_and_reported(tmp_path):
    db = IntentDB(tmp_path / "r.intentdb")
    db.add("alpha beta gamma", doc_key="a")
    db.add("delta epsilon zeta", doc_key="b")
    hits = db.query("alpha", k=2, hybrid=True, auto_intent=False)
    assert hits[0].doc_key == "a"
    assert 0 < hits[0].score < 1  # RRF values live in (0, ~2/61]
    assert "lexical_score" in hits[0].to_dict()
    db.close()
