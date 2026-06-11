import pytest

from intentdb import IntentDB
from intentdb.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello world", max_chars=100) == ["hello world"]
    assert chunk_text("   ", max_chars=100) == []


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_text(text, max_chars=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_all_content_is_retained():
    paras = [f"unique-marker-{i} content here." for i in range(30)]
    chunks = chunk_text("\n\n".join(paras), max_chars=120, overlap=30)
    joined = "\n".join(chunks)
    for i in range(30):
        assert f"unique-marker-{i}" in joined


def test_overlap_carries_context():
    paras = [(w + " ") * 16 for w in ("alpha", "bravo", "charlie")]
    chunks = chunk_text("\n\n".join(p.strip() for p in paras), max_chars=200, overlap=60)
    assert len(chunks) == 2
    # chunk 0 ends with the bravo paragraph; its tail must lead chunk 1
    assert chunks[0].endswith("bravo")
    assert chunks[1].startswith("bravo") and "charlie" in chunks[1]


def test_zero_overlap_means_no_carry():
    paras = [(w + " ") * 16 for w in ("alpha", "bravo")]
    chunks = chunk_text("\n\n".join(p.strip() for p in paras), max_chars=120, overlap=0)
    assert len(chunks) == 2
    assert "alpha" not in chunks[1]


def test_runon_text_hard_wraps():
    chunks = chunk_text("x" * 5000, max_chars=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c.replace("\n\n", "")) for c in chunks) >= 5000


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=100, overlap=-1)


def test_default_overlap_is_clamped_for_small_chunks():
    # default overlap (200) exceeds max_chars; it must be clamped, not raise
    chunks = chunk_text("word " * 200, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)


def test_db_add_chunked(tmp_path):
    db = IntentDB(tmp_path / "c.intentdb")
    text = "\n\n".join(f"Section {i}: " + "details " * 40 for i in range(10))
    keys = db.add_chunked(text, doc_key="manual", metadata={"source": "test"},
                          max_chars=400, overlap=80)
    assert len(keys) > 1
    assert keys[0] == "manual#0"
    doc = db.get("manual#0")
    assert doc["metadata"]["parent"] == "manual"
    assert doc["metadata"]["chunk"] == 0
    assert doc["metadata"]["source"] == "test"
    hits = db.query("Section 7 details", k=3, auto_intent=False)
    assert any(h.metadata.get("parent") == "manual" for h in hits)
    db.close()
