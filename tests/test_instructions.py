"""Instruction-conditioned query embedding (INSTRUCTOR-style intents)."""

import numpy as np

from intentdb import IntentDB
from intentdb.embedders import HashingEmbedder, OllamaEmbedder, PREFIX_MODES


class InstructionAwareStub(HashingEmbedder):
    """Hashing embedder that honors instructions, for testing the wiring.

    Conditioning simply folds the instruction text into the query, which
    shifts the lexical vector toward the instruction's vocabulary, the
    same mechanism real instruction-aware embedders use semantically.
    """

    supports_instructions = True

    def __init__(self):
        super().__init__(dim=512)
        self.instructions_seen: list[str] = []

    def embed_query(self, text, instruction=None):
        if instruction:
            self.instructions_seen.append(instruction)
            return self.embed(f"{instruction}: {text}")
        return self.embed(text)


def test_instruction_changes_query_vector():
    e = InstructionAwareStub()
    plain = e.embed_query("python")
    conditioned = e.embed_query("python", instruction="find snake facts")
    assert not np.allclose(plain, conditioned)


def test_db_passes_intent_instruction_to_embedder(tmp_path):
    e = InstructionAwareStub()
    db = IntentDB(tmp_path / "instr.intentdb", embedder=e)
    db.add("snake reptile jungle", doc_key="snake")
    db.add("code program function", doc_key="code")
    db.register_intent(
        "wildlife",
        description="animals and reptiles",
        instruction="find wildlife and animal facts",
    )
    db.query("python", intent="wildlife", k=1)
    assert "find wildlife and animal facts" in e.instructions_seen
    db.close()


def test_instruction_defaults_to_description(tmp_path):
    db = IntentDB(tmp_path / "d.intentdb")
    db.register_intent("coding", description="software and code")
    assert db.list_intents()[0]["instruction"] == "software and code"
    db.close()


def test_instruction_persists_across_reopen(tmp_path):
    path = tmp_path / "p.intentdb"
    db = IntentDB(path)
    db.register_intent("x", description="desc", instruction="custom instruction")
    db.close()
    db = IntentDB(path)
    assert db.list_intents()[0]["instruction"] == "custom instruction"
    db.close()


def test_hashing_embedder_ignores_instructions():
    e = HashingEmbedder(dim=128)
    assert e.supports_instructions is False
    assert np.allclose(
        e.embed_query("hello", instruction="anything"), e.embed_query("hello")
    )


def test_prefix_modes_table():
    assert PREFIX_MODES["nomic"] == ("search_query: ", "search_document: ")
    assert PREFIX_MODES["e5"] == ("query: ", "passage: ")


def test_ollama_prefixing_without_server():
    """Exercise the prefix/instruction text construction (no network)."""
    e = OllamaEmbedder.__new__(OllamaEmbedder)
    e.model, e.host, e.dim = "nomic-embed-text", "http://x", 8
    e.prefix_mode = "nomic"
    e.query_prefix, e.doc_prefix = PREFIX_MODES["nomic"]
    sent = []
    e._request = lambda text: (sent.append(text), [0.0] * 8)[1]

    e.embed_document("doc text")
    e.embed_query("query text")
    e.embed_query("query text", instruction="find facts")
    assert sent == [
        "search_document: doc text",
        "search_query: query text",
        "search_query: find facts: query text",
    ]
    assert "prefix_mode=nomic" in e.spec
