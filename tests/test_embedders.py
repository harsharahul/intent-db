import numpy as np
import pytest

from intentdb.embedders import HashingEmbedder, get_embedder


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=256)
    b = HashingEmbedder(dim=256)
    v1 = a.embed("the quick brown fox")
    v2 = b.embed("the quick brown fox")
    assert np.allclose(v1, v2)


def test_unit_norm():
    e = HashingEmbedder(dim=128)
    v = e.embed("hello world, this is a test of normalization")
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_empty_text_is_zero_vector():
    e = HashingEmbedder(dim=64)
    v = e.embed("")
    assert float(np.linalg.norm(v)) == 0.0


def test_similar_texts_score_higher():
    e = HashingEmbedder(dim=512)
    base = e.embed("install python packages with pip")
    near = e.embed("installing a python package using pip")
    far = e.embed("grilled salmon with lemon butter sauce")
    assert float(base @ near) > float(base @ far)


def test_embed_batch_matches_single():
    e = HashingEmbedder(dim=256)
    texts = ["alpha beta", "gamma delta epsilon"]
    mat = e.embed_batch(texts)
    assert mat.shape == (2, 256)
    for row, t in zip(mat, texts):
        assert np.allclose(row, e.embed(t))


def test_spec_roundtrip():
    e = HashingEmbedder(dim=300, char_ngrams=False)
    rebuilt = get_embedder(e.spec)
    assert rebuilt.dim == 300
    assert np.allclose(e.embed("roundtrip test"), rebuilt.embed("roundtrip test"))


def test_get_embedder_rejects_unknown():
    with pytest.raises(ValueError):
        get_embedder("nonsense:foo=1")
