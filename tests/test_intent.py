import numpy as np

from intentdb.embedders import HashingEmbedder
from intentdb.intent import Intent, IntentLens, infer_intent


def test_lens_gate_at_least_one():
    rng = np.random.default_rng(0)
    lens = IntentLens.fit(rng.normal(size=(5, 64)))
    assert lens.gate.shape == (64,)
    assert (lens.gate >= 1.0).all()
    assert lens.gate.max() > 1.0


def test_lens_single_example_degrades_gracefully():
    v = np.zeros(32)
    v[3] = 1.0
    lens = IntentLens.fit(v)
    # the active dimension gets the strongest gate
    assert lens.gate.argmax() == 3


def test_lensed_norm_identity_trick():
    """<q*g, d*g> must equal <q*g^2, d> — the optimization used at query time."""
    rng = np.random.default_rng(1)
    lens = IntentLens.fit(rng.normal(size=(4, 48)))
    q = rng.normal(size=48)
    d = rng.normal(size=48)
    direct = float(np.dot(q * lens.gate, d * lens.gate))
    optimized = float(np.dot(q * lens.gate_sq, d))
    assert abs(direct - optimized) < 1e-6


def test_intent_build_and_affinity():
    e = HashingEmbedder(dim=256)
    intent = Intent.build(
        "coding",
        description="software programming and source code",
        exemplars=["how to write a function", "debug my code"],
        embed_batch=e.embed_batch,
    )
    assert abs(float(np.linalg.norm(intent.vector)) - 1.0) < 1e-5
    code_doc = e.embed("programming a function in source code")
    food_doc = e.embed("recipe for tomato soup with basil")
    assert float(intent.affinity(code_doc)) > float(intent.affinity(food_doc))


def test_infer_intent_picks_right_one():
    e = HashingEmbedder(dim=512)
    coding = Intent.build(
        "coding",
        "software programming source code functions debugging",
        ["write code", "fix a bug in my program"],
        e.embed_batch,
    )
    cooking = Intent.build(
        "cooking",
        "recipes food cooking kitchen ingredients meals",
        ["how to cook dinner", "recipe ideas"],
        e.embed_batch,
    )
    q = e.embed("debugging a program function")
    best, scores = infer_intent(q, [coding, cooking])
    assert best is not None and best.name == "coding"
    assert scores["coding"] > scores["cooking"]


def test_infer_intent_returns_none_below_threshold():
    e = HashingEmbedder(dim=512)
    coding = Intent.build(
        "coding", "software programming source code", ["write code"], e.embed_batch
    )
    q = e.embed("zebra migration patterns serengeti")
    best, _ = infer_intent(q, [coding], threshold=0.08)
    assert best is None


def test_infer_intent_empty_list():
    best, scores = infer_intent(np.zeros(8), [])
    assert best is None and scores == {}
