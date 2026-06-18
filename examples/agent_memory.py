"""IntentDB as an LLM agent's memory: same store, recall by purpose.

The primary use case for IntentDB. A coding agent works in phases — *planning*,
*debugging*, *reviewing* — and writes memories into one IntentDB file as
it goes. When it recalls, it queries under its *current* phase, so the
same query surfaces a different memory depending on what the agent is
doing right now. The agent reports which memory it actually used
(``record_feedback``), and the database learns better per-phase ranking
over time.

This is the setting intent conditioning is built for: the agent declares
its phase for free (its control flow already knows it), and feedback is
automatic (it knows which memory it used).

Run: ``python examples/agent_memory.py``
See ``examples/AGENT_MEMORY.md`` to wire it to a real Claude Code agent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from intentdb import IntentDB

# The agent's work phases, registered as intents. Note these describe a
# *purpose*, not a topic — the differential recall is by what the agent is
# doing, not by keyword.
PHASES = ("planning", "debugging", "reviewing")

INTENTS = {
    "planning": dict(
        description="a design decision and its rationale: what we chose, the "
        "alternatives we rejected, and the tradeoffs we weighed",
        exemplars=[
            "design decision and rationale",
            "what did we choose and what did we reject",
            "the tradeoffs and alternatives we weighed",
            "why we decided on this approach",
        ],
    ),
    "debugging": dict(
        description="a bug and its fix: the error or crash, why it failed, the "
        "root cause we debugged, and the fix that resolved it",
        exemplars=[
            "the bug and its root cause",
            "why did it fail or crash",
            "the error and the fix",
            "what we debugged and fixed",
        ],
    ),
    "reviewing": dict(
        description="a review rule: the convention, standard, or required "
        "practice we must follow, and what review caught",
        exemplars=[
            "the review rule we must follow",
            "the required convention and standard",
            "what review caught",
            "the practice we are required to follow",
        ],
    ),
}

# Memories the agent wrote during the session. Keys are prefixed by the
# phase whose query *should* surface them (plan-/bug-/rule-), used only for
# narration and the feedback loop — the stored document carries no phase
# label; affinity to each phase is computed geometrically at ingest.
MEMORIES: list[tuple[str, str]] = [
    # -- storage --
    ("plan-storage", "Storage design decision: we chose SQLite in WAL mode and "
     "rejected Postgres and Redis. The rationale and the tradeoffs we weighed "
     "were one local file, many readers, and a single writer."),
    ("bug-storage", "Storage bug: a query returned stale results after a write "
     "and the error was intermittent. The root cause we debugged was an "
     "unsynced in-memory mirror; the fix landed in add_many."),
    ("rule-storage", "Storage review rule: every schema change must follow the "
     "migration convention, a backward-compatible column add, so old files "
     "keep opening. A required standard caught in review."),
    # -- scoring / fusion weights --
    ("plan-weights", "Fusion weights design decision: we chose lensed 0.6, "
     "affinity 0.25, base 0.15. The rationale was that the lens leads and "
     "affinity is a prior; we rejected an equal split after weighing tradeoffs."),
    ("bug-shrinkage", "Fusion weights bug: on a tiny corpus the scores "
     "inverted and ranking failed. The root cause we debugged was "
     "over-aggressive centering; the fix was statistic shrinkage."),
    ("rule-convex", "Fusion weights review rule: learned weights must stay "
     "convex, non-negative and summing to one, a required standard, or scores "
     "stop being comparable. Caught in review."),
    # -- reranker --
    ("plan-rerank", "Reranker design decision: we chose an optional CPU "
     "cross-encoder and decided to inject the intent instruction into the "
     "pair. The rationale was best quality given the tradeoffs."),
    ("bug-rerank", "Reranker bug: a listwise model returned no score and the "
     "wrapper crashed with an error. The root cause was found; the fix raises "
     "a clear message naming the model."),
    ("rule-rerank", "Reranker review rule: we must document that small "
     "cross-encoders only steer topically, not follow instructions, an honesty "
     "standard the review required."),
    # -- intent lens --
    ("plan-lens", "Lens design decision: we chose a diagonal gate applied "
     "query-side only. The rationale and tradeoff were one matvec for the "
     "whole corpus; we rejected per-document re-embedding."),
    ("bug-lens-shrink", "Lens bug: with few exemplars the gate overfit and the "
     "query term was drowned, a ranking failure. The root cause was debugged; "
     "the fix shrinks the gate toward identity."),
    ("rule-asymmetric", "Lens review rule: we must never re-normalize documents "
     "in the lensed space, a required convention, because it penalizes "
     "intent-rich documents. Caught in review."),
    # -- MCP / API --
    ("plan-mcp", "MCP server design decision: we chose a stdio JSON-RPC server "
     "exposing query, add, and feedback tools. The rationale was that any "
     "agent can mount it; we weighed the tradeoffs."),
    ("bug-mcp", "MCP server bug: an unknown method crashed the server instead "
     "of replying, an error in dispatch. The root cause was found and fixed by "
     "returning a method-not-found response."),
    ("rule-feedback", "MCP server review rule: the agent must call "
     "record_feedback after using a memory, a required practice, so ranking "
     "improves. A standard enforced in review."),
    # -- mining / conventions --
    ("plan-mining", "Mining design decision: we chose spherical k-means over "
     "the query log to suggest intents. The rationale was to surface recurring "
     "themes; we rejected manual-only definition."),
    ("rule-commits", "Commit review rule: messages must end with the fixed "
     "sign-off and follow the type-prefixed form, a required convention and "
     "standard checked in review."),
]

# Topic-keyword queries whose correct answer depends entirely on the phase.
PROBE_QUERIES = [
    "storage",
    "fusion weights",
    "reranker",
    "lens",
    "MCP server",
]
DEMO_QUERY = "storage"


def build_memory(path, embedder: str = "hashing:dim=512") -> IntentDB:
    """Create a memory store: register the phases, write the memories."""
    if Path(path).exists():
        Path(path).unlink()
    db = IntentDB(path, embedder=embedder)
    db.add_many([(text, key, None) for key, text in MEMORIES])
    for name, spec in INTENTS.items():
        db.register_intent(name, **spec)
    return db


def run_feedback_loop(db: IntentDB) -> dict:
    """Simulate the agent reporting which memories it used while debugging.

    For each debugging probe, the matching ``bug-*`` memory was useful and
    the competing ``plan-*`` / ``rule-*`` memories were not. Feeding that
    back lets the database fit better per-phase fusion weights.
    """
    bug_keys = [k for k, _ in MEMORIES if k.startswith("bug-")]
    other_keys = [k for k, _ in MEMORIES if not k.startswith("bug-")]
    for query in PROBE_QUERIES:
        hits = db.query(query, intent="debugging", k=5, log=False)
        used = next((h.doc_key for h in hits if h.doc_key in bug_keys), None)
        if used:
            db.record_feedback(query, used, useful=True, intent="debugging")
        for h in hits:
            if h.doc_key in other_keys:
                db.record_feedback(query, h.doc_key, useful=False, intent="debugging")
    return db.learn_fusion_weights(intent="debugging")


def _show(db: IntentDB, query: str) -> None:
    print(f'\nquery: "{query}"  — same store, recall depends on the phase')
    for phase in PHASES:
        top = db.query(query, intent=phase, k=1)[0]
        text = top.text if len(top.text) <= 88 else top.text[:85] + "..."
        print(f"  [{phase:9}] ({top.doc_key}) {text}")


def main() -> None:
    path = Path(tempfile.mkdtemp()) / "agent.intentdb"
    db = build_memory(path)
    print(f"agent memory: {db.stats()['documents']} memories, "
          f"phases = {list(PHASES)}")

    # The headline: same query, different phase, different memory.
    for query in ("storage", "reranker", "lens"):
        _show(db, query)

    # Plain search (no phase) can't tell them apart — the gap this fills.
    print('\nwithout a phase, the same query is answered the same way every time:')
    for h in db.query(DEMO_QUERY, auto_intent=False, k=2):
        print(f"  ({h.doc_key}) {h.text[:85]}")

    # The agent reports what it used; the database learns the phase's ranking.
    before = db.fusion_weights().get("debugging")
    learned = run_feedback_loop(db)
    print("\nfeedback loop (debugging phase):")
    print(f"  default weights : lensed 0.60 affinity 0.25 base 0.15")
    w = learned.get("debugging")
    if w:
        print(f"  learned weights : lensed {w['lensed']:.2f} affinity "
              f"{w['affinity']:.2f} base {w['base']:.2f}  "
              f"(from {db.stats()['feedback']} feedback marks)")
    db.close()


if __name__ == "__main__":
    main()
