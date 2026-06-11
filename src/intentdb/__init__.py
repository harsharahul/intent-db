"""IntentDB — a local-first, intent-aware vector database.

A vector database with an extra dimension: *intent*. Documents are embedded
once into a base vector space, but every registered intent defines a "lens"
(a per-dimension gating of the embedding space) so the effective
vectorization of both queries and documents changes with the intent that is
active at retrieval time.

Quick start::

    from intentdb import IntentDB

    db = IntentDB("./my.intentdb")
    db.add("Python is a popular programming language.", doc_key="py-lang")
    db.register_intent(
        "coding",
        description="Questions about software and programming",
        exemplars=["how do I write code", "debug my program"],
    )
    hits = db.query("python", intent="coding", k=3)
"""

from .db import IntentDB, QueryResult
from .embedders import Embedder, HashingEmbedder, get_embedder
from .intent import Intent, IntentLens

__all__ = [
    "IntentDB",
    "QueryResult",
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "Intent",
    "IntentLens",
]

__version__ = "0.1.0"
