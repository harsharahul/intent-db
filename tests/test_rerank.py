"""Optional cross-encoder rerank stage on query()."""

import json

import numpy as np
import pytest

from intentdb import IntentDB
from intentdb.rerank import Reranker, get_reranker


class PreferenceReranker(Reranker):
    """Assigns scores from an explicit text -> score mapping; records calls."""

    def __init__(self, preferences: dict[str, float] | None = None):
        self.preferences = preferences or {}
        self.queries_seen: list[str] = []

    @property
    def spec(self) -> str:
        return "preference-stub"

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        self.queries_seen.append(query)
        return np.array(
            [self.preferences.get(t, 0.0) for t in texts], dtype=np.float64
        )


PROG = "python is a programming language for writing code"
SNAKE = "the python snake is a reptile that lives in the jungle"
BREAD = "bread recipes use flour and yeast"


@pytest.fixture()
def db(tmp_path):
    db = IntentDB(tmp_path / "rerank.intentdb")
    db.add(PROG, doc_key="prog")
    db.add(SNAKE, doc_key="snake", metadata={"kind": "wildlife"})
    db.add(BREAD, doc_key="bread", metadata={"kind": "food"})
    yield db
    db.close()


def test_rerank_reorders_results(db):
    plain = db.query("python", k=3, auto_intent=False)
    assert plain[0].doc_key != "bread"  # dense search has no reason to pick bread

    rr = PreferenceReranker({BREAD: 5.0, PROG: 1.0})
    out = db.query("python", k=3, auto_intent=False, rerank=rr)
    assert [r.doc_key for r in out] == ["bread", "prog", "snake"]
    assert out[0].rerank_score == 5.0
    assert out[0].score == 5.0  # score holds the value results were ranked by


def test_rerank_off_by_default(db):
    out = db.query("python", k=3, auto_intent=False)
    assert all(r.rerank_score is None for r in out)
    assert "rerank_score" not in out[0].to_dict()


def test_rerank_score_in_to_dict(db):
    rr = PreferenceReranker({PROG: 2.0})
    out = db.query("python", k=2, auto_intent=False, rerank=rr)
    d = out[0].to_dict()
    assert d["rerank_score"] == 2.0


def test_intent_instruction_prefixes_rerank_query(db):
    db.register_intent(
        "wildlife",
        description="animals and reptiles",
        instruction="find wildlife and animal facts",
    )
    rr = PreferenceReranker()
    db.query("python", intent="wildlife", k=2, rerank=rr)
    assert rr.queries_seen == ["find wildlife and animal facts: python"]


def test_plain_query_text_without_intent(db):
    rr = PreferenceReranker()
    db.query("python", k=2, auto_intent=False, rerank=rr)
    assert rr.queries_seen == ["python"]


def test_rerank_depth_windows_candidates(db):
    # precondition: bread ranks last for "python" in dense space
    plain = db.query("python", k=3, auto_intent=False)
    assert plain[-1].doc_key == "bread"
    # with a window of 2 it never reaches the reranker, however much the
    # reranker would like it
    rr = PreferenceReranker({BREAD: 9.0})
    out = db.query("python", k=2, auto_intent=False, rerank=rr, rerank_depth=2)
    assert "bread" not in [r.doc_key for r in out]


def test_rerank_depth_beyond_collection(db):
    rr = PreferenceReranker({BREAD: 9.0})
    out = db.query("python", k=3, auto_intent=False, rerank=rr, rerank_depth=100)
    assert [r.doc_key for r in out][0] == "bread"
    assert len(out) == 3


def test_rerank_composes_with_hybrid(db):
    rr = PreferenceReranker({BREAD: 9.0})
    out = db.query("python", k=3, auto_intent=False, hybrid=True, rerank=rr)
    assert out[0].doc_key == "bread"  # reranker has the last word
    assert out[0].rerank_score == 9.0


def test_rerank_depth_clamped_to_k(db):
    rr = PreferenceReranker()
    out = db.query("python", k=3, auto_intent=False, rerank=rr, rerank_depth=1)
    assert len(out) == 3  # window grows to k, results are not lost


def test_where_filter_composes_with_rerank(db):
    rr = PreferenceReranker({BREAD: 9.0})
    out = db.query(
        "python",
        k=3,
        auto_intent=False,
        rerank=rr,
        where=lambda m: m.get("kind") != "food",
    )
    assert "bread" not in [r.doc_key for r in out]
    assert len(out) == 2


def test_rerank_true_builds_default_and_caches(db, monkeypatch):
    stub = PreferenceReranker()
    specs_built: list[str] = []

    def fake_factory(spec):
        specs_built.append(spec)
        return stub

    monkeypatch.setattr("intentdb.db.get_reranker", fake_factory)
    db.query("python", k=1, auto_intent=False, rerank=True)
    db.query("python again", k=1, auto_intent=False, rerank=True)
    assert specs_built == ["flashrank"]  # built once, cached on the instance
    assert len(stub.queries_seen) == 2


def test_rerank_spec_string(db, monkeypatch):
    stub = PreferenceReranker()
    specs_built: list[str] = []
    monkeypatch.setattr(
        "intentdb.db.get_reranker",
        lambda spec: (specs_built.append(spec), stub)[1],
    )
    db.query("python", k=1, auto_intent=False, rerank="crossencoder:model=x")
    assert specs_built == ["crossencoder:model=x"]


def test_get_reranker_unknown():
    with pytest.raises(ValueError, match="unknown reranker"):
        get_reranker("nope")


class FakeRanker:
    """Stands in for flashrank.Ranker (no model, no dependency)."""

    def __init__(self, results):
        self.results = results

    def rerank(self, request):
        return self.results


def make_flashrank_wrapper(results):
    from intentdb.rerank import FlashRankReranker

    rr = FlashRankReranker.__new__(FlashRankReranker)
    rr.model = "stub-model"
    rr._ranker = FakeRanker(results)
    rr._request_cls = lambda **kw: kw
    return rr


def test_flashrank_scores_map_back_by_id():
    # flashrank returns passages sorted by score, not input order
    rr = make_flashrank_wrapper(
        [{"id": 2, "score": 0.9}, {"id": 0, "score": 0.5}, {"id": 1, "score": 0.1}]
    )
    out = rr.scores("q", ["a", "b", "c"])
    assert out.tolist() == [0.5, 0.1, 0.9]


def test_flashrank_scoreless_model_errors_helpfully():
    # listwise flashrank models return no per-passage scores; the wrapper
    # must say so instead of raising a bare KeyError
    rr = make_flashrank_wrapper([{"id": 0, "text": "a"}])
    with pytest.raises(ValueError, match="stub-model"):
        rr.scores("q", ["a"])


def test_flashrank_missing_dep_errors_helpfully():
    try:
        import flashrank  # noqa: F401

        pytest.skip("flashrank is installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match=r"intentdb\[rerank\]"):
        get_reranker("flashrank")


def test_crossencoder_missing_dep_errors_helpfully():
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers is installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match=r"intentdb\[sbert\]"):
        get_reranker("crossencoder")


def test_cli_rerank_flag(tmp_path, capsys, monkeypatch):
    from intentdb.cli import main

    stub = PreferenceReranker({BREAD: 7.0})
    monkeypatch.setattr("intentdb.db.get_reranker", lambda spec: stub)

    db_path = str(tmp_path / "cli.intentdb")
    main(["init", db_path])
    capsys.readouterr()
    main(["add", db_path, PROG, "--key", "prog"])
    main(["add", db_path, BREAD, "--key", "bread"])
    capsys.readouterr()

    code = main(["query", db_path, "python", "--rerank", "--json", "--no-auto-intent"])
    out = capsys.readouterr().out
    assert code == 0
    hits = json.loads(out)
    assert hits[0]["doc_key"] == "bread"
    assert hits[0]["rerank_score"] == 7.0
    assert len(stub.queries_seen) == 1


def test_mcp_query_rerank(tmp_path, monkeypatch):
    from intentdb.mcp_server import call_tool

    stub = PreferenceReranker({BREAD: 7.0})
    monkeypatch.setattr("intentdb.db.get_reranker", lambda spec: stub)

    db = IntentDB(tmp_path / "mcp.intentdb")
    db.add(PROG, doc_key="prog")
    db.add(BREAD, doc_key="bread")
    hits = call_tool(
        db, "intentdb_query", {"query": "python", "rerank": True, "auto_intent": False}
    )
    assert hits[0]["doc_key"] == "bread"
    assert hits[0]["rerank_score"] == 7.0
    db.close()
