"""Whitened lens fitting, shrinkage, and Rocchio PRF (research items 1+2)."""

import numpy as np
import pytest

from intentdb import IntentDB
from intentdb.embedders import HashingEmbedder
from intentdb.intent import IntentLens, standardize


# ---- item 1a: shrinkage toward identity ------------------------------------


def test_shrinkage_grows_with_exemplar_count():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=64)
    one = IntentLens.fit(direction[None, :])
    many = IntentLens.fit(
        np.stack([direction + 0.05 * rng.normal(size=64) for _ in range(24)])
    )
    # a single exemplar barely bends the space; many exemplars bend it more
    assert one.gate.max() - 1.0 < 0.7
    assert many.gate.max() > one.gate.max()
    assert (one.gate >= 1.0).all() and (many.gate >= 1.0).all()


def test_shrinkage_bounds():
    rng = np.random.default_rng(1)
    n = 16
    lens = IntentLens.fit(rng.normal(size=(n, 32)), strength=4.0)
    # max gate = 1 + strength * n/(n+8)
    assert lens.gate.max() <= 1.0 + 4.0 * n / (n + 8) + 1e-5


# ---- item 1b: standardization kills rogue-dimension artifacts ---------------


def test_standardize_removes_shared_offset():
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(50, 16))
    mat[:, 3] += 100.0  # rogue dimension: huge shared offset, no signal
    mu, sigma = mat.mean(axis=0), np.maximum(mat.std(axis=0), 1e-3)
    s = standardize(mat, mu, sigma)
    assert abs(s[:, 3].mean()) < 1e-6
    assert abs(s[:, 3].std() - 1.0) < 1e-6


def test_lens_fit_on_standardized_basis_ignores_rogue_dimension():
    """A rogue dimension (large offset shared by the whole corpus) must not
    carry the peak gate once exemplars are standardized against
    corpus-scale statistics — but it dominates a raw-basis fit."""
    rng = np.random.default_rng(3)
    dim, rogue = 64, 0
    corpus = rng.normal(scale=0.1, size=(2000, dim))
    corpus[:, rogue] += 5.0  # anisotropy artifact shared by every vector
    mu = corpus.mean(axis=0)
    sigma = np.maximum(corpus.std(axis=0), 1e-3)

    # intent exemplars: rogue offset + genuine signal on dimension 7
    exemplars = rng.normal(scale=0.1, size=(6, dim))
    exemplars[:, rogue] += 5.0
    exemplars[:, 7] += 1.0

    raw_fit = IntentLens.fit(exemplars)
    std_fit = IntentLens.fit(standardize(exemplars, mu, sigma))
    assert int(np.argmax(raw_fit.gate)) == rogue       # the failure mode
    assert int(np.argmax(std_fit.gate)) == 7            # the fix
    assert std_fit.gate[rogue] < std_fit.gate[7]


def test_corpus_stats_shrink_with_collection_size(tmp_path):
    """mu/sigma must stay near the identity basis for tiny collections."""
    db = IntentDB(tmp_path / "shrink.intentdb")
    for i in range(4):
        db.add(f"document {i} alpha beta", doc_key=f"d{i}")
    db._ensure_loaded()
    mu, sigma = db._corpus_stats()
    lam = 4 / (4 + db.STATS_PSEUDO_COUNT)
    raw_mu = db._matrix.astype(np.float64).mean(axis=0)
    assert np.allclose(mu, lam * raw_mu)
    assert abs(float(sigma.mean()) - 1.0) < 0.1  # barely moved from identity
    db.close()


def test_intent_stats_persist_and_restore(tmp_path):
    path = tmp_path / "s.intentdb"
    db = IntentDB(path)
    db.add("alpha beta gamma delta", doc_key="a")
    db.add("epsilon zeta eta theta", doc_key="b")
    intent = db.register_intent("x", description="alpha beta")
    mu, sigma = intent.mu.copy(), intent.sigma.copy()
    db.close()
    db = IntentDB(path)
    assert [i["name"] for i in db.list_intents()] == ["x"]
    restored = db._intents["x"]
    assert np.allclose(restored.mu, mu)
    assert np.allclose(restored.sigma, sigma)
    hits = db.query("alpha", intent="x", k=1)
    assert hits and np.isfinite(hits[0].score)
    db.close()


def test_lensed_scores_remain_cosine_scaled(tmp_path):
    """The standardized lensed signal must stay comparable to cosine so the
    three-signal blend is not dominated by one term."""
    db = IntentDB(tmp_path / "scale.intentdb")
    for i in range(30):
        db.add(f"document about topic {i % 5} with content words", doc_key=f"d{i}")
    db.register_intent("t", description="topic content words", exemplars=["topic 1"])
    hits = db.query("topic content", intent="t", k=10)
    assert all(abs(h.lensed_score) < 5.0 for h in hits)
    db.close()


# ---- item 2: intent-aware Rocchio PRF ---------------------------------------


@pytest.fixture()
def prf_db(tmp_path):
    db = IntentDB(tmp_path / "prf.intentdb", embedder="hashing:dim=512")
    # a tight cluster sharing vocabulary with the query's topic...
    db.add("postgres mvcc transactions isolation snapshot", doc_key="c1")
    db.add("postgres transactions snapshot isolation levels", doc_key="c2")
    db.add("mvcc snapshot isolation concurrency transactions", doc_key="c3")  # no 'postgres'
    # ...and unrelated fillers
    for i in range(6):
        db.add(f"gardening tips for roses and tulips batch {i}", doc_key=f"g{i}")
    yield db
    db.close()


def test_prf_promotes_cluster_member_missing_query_term(prf_db):
    """'c3' shares the cluster's vocabulary but lacks the query token
    'postgres'; PRF should pull the query toward the cluster and improve
    c3's rank."""
    def rank_of(key, **kw):
        hits = prf_db.query("postgres", k=9, auto_intent=False, **kw)
        return [h.doc_key for h in hits].index(key)

    assert rank_of("c3", prf=True) <= rank_of("c3", prf=False)
    # and the cluster occupies the top after feedback
    top3 = [h.doc_key for h in prf_db.query("postgres", k=3, auto_intent=False, prf=True)]
    assert set(top3) == {"c1", "c2", "c3"}


def test_prf_with_intent_pushes_away_off_intent_docs(prf_db):
    prf_db.register_intent(
        "databases",
        description="database transactions mvcc isolation postgres",
        exemplars=["postgres transactions", "snapshot isolation"],
    )
    hits = prf_db.query("postgres", intent="databases", k=9, prf=True)
    keys = [h.doc_key for h in hits]
    assert set(keys[:3]) == {"c1", "c2", "c3"}
    assert all(k.startswith("g") for k in keys[3:])


def test_prf_noop_on_tiny_db(tmp_path):
    db = IntentDB(tmp_path / "tiny.intentdb")
    db.add("only document", doc_key="d")
    hits = db.query("document", prf=True, k=1)
    assert hits[0].doc_key == "d"
    db.close()


def test_prf_query_vector_stays_unit_norm(prf_db):
    q = prf_db.embedder.embed_query("postgres")
    top = np.array([0, 1, 2])
    q2 = prf_db._rocchio(q, top, np.array([0.5, 0.4, 0.3]), None)
    assert abs(float(np.linalg.norm(q2)) - 1.0) < 1e-5
