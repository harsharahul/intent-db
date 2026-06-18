# IntentDB as agent memory: recall by purpose

Most agent-memory tools store memories and rank them by similarity to the
query. IntentDB adds the missing coordinate: the agent's **current goal**.
The same memory store returns a different memory depending on whether the
agent is *planning*, *debugging*, or *reviewing* — and it learns which
memories are useful in each phase from the agent's own feedback.

This is the use case intent conditioning is built for. The agent declares
its phase for free (its control flow already knows it), and feedback is
automatic (it knows which memory it actually used).

## Run the demo (no setup)

```bash
python examples/agent_memory.py
```

```
query: "storage"  — same store, recall depends on the phase
  [planning ] (plan-storage) Storage design decision: we chose SQLite in WAL mode and rejected Postgres...
  [debugging] (bug-storage)  Storage bug: a query returned stale results after a write and the error...
  [reviewing] (rule-storage) Storage review rule: every schema change must follow the migration...

without a phase, the same query is answered the same way every time.

feedback loop (debugging phase):
  default weights : lensed 0.60 affinity 0.25 base 0.15
  learned weights : lensed 0.22 affinity 0.69 base 0.09  (from 19 feedback marks)
```

One query, three phases, three memories — and after the agent reports what
it used, the database shifts weight onto the signal that actually carries
the phase (affinity). Plain similarity search returns the same ranking
every time, regardless of what the agent is doing; that is the gap this
fills.

> The demo uses the zero-dependency hashing embedder, so its memories are
> written with explicit phase vocabulary ("Bug… root cause… fixed",
> "Decided… rejected") — which is how an agent naturally writes them. For
> free-form phrasing, use a semantic embedder
> (`ollama:model=nomic-embed-text` or `sbert:...`); phase recall is much
> more robust there.

## Wire it to a real Claude Code agent

### 1. Create the memory and register the phases (once)

```bash
intentdb init agent.intentdb --embedder ollama:model=nomic-embed-text

intentdb intent add agent.intentdb planning \
  --description "a design decision and its rationale: what we chose, the alternatives we rejected, and the tradeoffs" \
  --exemplar "design decision and rationale" --exemplar "what did we choose and reject"

intentdb intent add agent.intentdb debugging \
  --description "a bug and its fix: the error, why it failed, the root cause, and the fix" \
  --exemplar "the bug and its root cause" --exemplar "the error and the fix"

intentdb intent add agent.intentdb reviewing \
  --description "a review rule: the convention, standard, or required practice to follow" \
  --exemplar "the review rule we must follow" --exemplar "the required convention and standard"
```

### 2. Mount the MCP server (`.mcp.json` in your project)

```jsonc
{
  "mcpServers": {
    "memory": {
      "command": "intentdb",
      "args": ["serve-mcp", "agent.intentdb"]
    }
  }
}
```

### 3. Teach the agent the protocol (`CLAUDE.md`)

```markdown
## Memory protocol (intentdb MCP)

You have a purpose-aware memory. Use the phase you are in as the intent.

- When you make a design decision, write it:
  `intentdb_add(text="<decision, rationale, rejected alternatives>")`
- When you fix a bug, write it:
  `intentdb_add(text="<the error, root cause, and the fix>")`
- When you learn a convention, write it:
  `intentdb_add(text="<the rule or required practice>")`

- Before acting, recall under your current phase:
  `intentdb_query(query="<topic>", intent="planning" | "debugging" | "reviewing")`
- After you use (or discard) a recalled memory, report it:
  `intentdb_record_feedback(query="<topic>", doc_key="<id>", useful=true, intent="<phase>")`

Periodically the operator runs `intentdb learn agent.intentdb` to turn that
feedback into better per-phase ranking.
```

That is the whole loop: the agent writes memories as it works, recalls them
conditioned on what it is doing, and the database gets sharper at each phase
the more the agent uses it. Memories are embedded once; registering a new
phase over the whole memory is a single matrix-vector product.

See the top-level [README](../README.md) for the engine, and
[`bench/`](../bench/) for the measured retrieval quality.
