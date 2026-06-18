"""The agent-memory reference example: differential recall + feedback loop.

Verifies the core claim — the same memory store returns different memories
depending on the agent's current phase (debugging / planning / reviewing).
"""

import pytest

from examples import agent_memory


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    path = tmp_path_factory.mktemp("agentmem") / "agent.intentdb"
    db = agent_memory.build_memory(path)
    yield db
    db.close()


def test_same_query_different_phase_different_memory(mem):
    # the headline: one query, three phases, three different top memories
    tops = {
        phase: mem.query(agent_memory.DEMO_QUERY, intent=phase, k=1)[0].doc_key
        for phase in agent_memory.PHASES
    }
    assert len(set(tops.values())) == 3, f"recall did not differ by phase: {tops}"


def test_each_phase_surfaces_its_own_kind_of_memory(mem):
    # across every probe query, the top hit under a phase is a memory of that
    # phase's kind (keys are prefixed plan-/bug-/rule- by their phase)
    prefix = {"planning": "plan-", "debugging": "bug-", "reviewing": "rule-"}
    hits = misses = 0
    for query in agent_memory.PROBE_QUERIES:
        for phase in agent_memory.PHASES:
            top = mem.query(query, intent=phase, k=1)[0].doc_key
            if top.startswith(prefix[phase]):
                hits += 1
            else:
                misses += 1
    # the lexical hashing embedder is not perfect; require a strong majority
    assert hits / (hits + misses) >= 0.7, f"phase-appropriate recall too low: {hits}/{hits + misses}"


def test_plain_search_cannot_tell_phases_apart(mem):
    # without an intent, the same query returns the same ranking every time —
    # the gap the agent-memory use case fills
    a = [h.doc_key for h in mem.query(agent_memory.DEMO_QUERY, auto_intent=False, k=3)]
    b = [h.doc_key for h in mem.query(agent_memory.DEMO_QUERY, auto_intent=False, k=3)]
    assert a == b


def test_feedback_loop_learns_weights(tmp_path):
    db = agent_memory.build_memory(tmp_path / "fb.intentdb")
    learned = agent_memory.run_feedback_loop(db)
    assert learned.get("debugging") is not None  # enough pairs -> weights fit
    w = learned["debugging"]
    assert w["lensed"] + w["affinity"] + w["base"] == pytest.approx(1.0, abs=1e-6)
    db.close()
