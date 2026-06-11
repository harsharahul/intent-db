# IntentDB

**A local-first vector database with an extra dimension: intent.**

A normal vector database embeds your data once and answers every query
against that single fixed geometry. IntentDB makes the geometry
*conditional on intent*: the same corpus and the same query return
different results depending on what the asker is trying to do — and when
no intent is declared, the most plausible one is inferred from the query
itself. Storage is general-purpose; retrieval is designed for an LLM
consumer (structured hits, score breakdowns, and a built-in MCP server).

```text
query: "python"                          query: "python"
intent: coding                           intent: wildlife
──────────────────────────               ──────────────────────────
1. Python is a programming               1. The python is a large
   language; write code...                  snake, a reptile that...
```

- Single SQLite file, pure Python + NumPy. No services, no model downloads
  required (a deterministic hashing embedder ships in the box; plug in
  Ollama or sentence-transformers for semantic embeddings).
- Three intent mechanisms, fused into one score:
  **instruction conditioning** (the query is re-embedded under the
  intent's instruction, [INSTRUCTOR](https://arxiv.org/abs/2212.09741)-style,
  when the embedder supports it), an **intent lens** (a per-intent
  diagonal re-weighting of embedding space — model-agnostic), and
  **intent affinity** (precomputed `cos(doc, intent)` per document).
- Adding an intent never re-embeds your corpus: documents are embedded
  once; intents cost one matrix-vector product to index.

See [PLAN.md](PLAN.md) for the architecture, the math, the research it is
grounded in, and known limitations.

## Install

```bash
pip install -e .          # from this repo; needs Python 3.10+, numpy
pip install -e .[dev]     # + pytest
```

## Python API

```python
from intentdb import IntentDB

db = IntentDB("knowledge.intentdb")            # hashing embedder by default
# db = IntentDB("knowledge.intentdb", embedder="ollama:model=nomic-embed-text")

db.add("Python is a programming language; write code and debug programs.",
       doc_key="py-lang", metadata={"topic": "software"})
db.add("The python is a large snake, a reptile of jungle habitats.",
       doc_key="py-snake", metadata={"topic": "nature"})

db.register_intent(
    "coding",
    description="software programming, source code, debugging",
    exemplars=["how do I write code", "debug my program"],
)
db.register_intent(
    "wildlife",
    description="animals, reptiles, snakes, jungle habitats",
    exemplars=["what do snakes eat"],
)

db.query("python", intent="coding")[0].doc_key    # -> "py-lang"
db.query("python", intent="wildlife")[0].doc_key  # -> "py-snake"

# No intent? It is inferred from the query (and reported on the result):
hit = db.query("python snake habitat")[0]
hit.intent, hit.intent_inferred                   # -> ("wildlife", True)

# Every hit carries its score breakdown:
hit.score, hit.base_score, hit.lensed_score, hit.intent_affinity

# Metadata filters, custom signal weights:
db.query("python", intent="coding", k=3,
         where=lambda m: m.get("topic") == "software",
         weights={"lensed": 0.5, "affinity": 0.3, "base": 0.2})
```

## CLI

```bash
intentdb init kb.intentdb
intentdb add kb.intentdb "Postgres uses MVCC for concurrency" --key pg-mvcc
intentdb add kb.intentdb --file notes.txt --split-paragraphs
intentdb intent add kb.intentdb debugging \
    --description "diagnosing errors and failures in software" \
    --exemplar "why is my service crashing"
intentdb query kb.intentdb "postgres locks" --intent debugging -k 3
intentdb query kb.intentdb "postgres locks" --json     # for machines
intentdb explain kb.intentdb "why does my app crash"   # intent classifier view
intentdb stats kb.intentdb
```

## Use from an LLM (MCP server)

IntentDB ships an MCP stdio server, so any MCP client (Claude Code, Claude
Desktop, local agents) can mount a database as a retrieval tool:

```jsonc
// .mcp.json
{
  "mcpServers": {
    "intentdb": {
      "command": "intentdb",
      "args": ["serve-mcp", "/path/to/kb.intentdb"]
    }
  }
}
```

Exposed tools: `intentdb_query`, `intentdb_add`, `intentdb_register_intent`,
`intentdb_list_intents`, `intentdb_explain`, `intentdb_stats`.

## Embedders

| Spec | What it is |
|---|---|
| `hashing:dim=512` (default) | Deterministic lexical hashing; zero deps, no downloads |
| `ollama:model=nomic-embed-text` | Any local Ollama embedding model; task prefixes (`prefix_mode=nomic\|e5\|none`) and intent instructions supported |
| `sbert:model=all-MiniLM-L6-v2` | sentence-transformers models (`pip install .[sbert]`) |

The embedder spec is stored in the database and restored on reopen;
mismatched dimensions are rejected.

## How intent changes the vectorization

For query `q`, document `d` (embedded once, unit norm), active intent with
vector `t`, instruction `i`, and lens gate `g` (a per-dimension weighting
fit from the intent's description and exemplars):

```text
q        = embed_query(text, instruction=i)      # moves with intent (neural embedders)
lensed   = ⟨q·g, d·g⟩ / ‖q·g‖                    # overlap on intent dimensions amplified
affinity = cos(d, t)                              # precomputed at ingest
base     = cos(q, d)                              # plain vector search
score    = 0.6·lensed + 0.25·affinity + 0.15·base # weights tunable per query
```

The identity `⟨q·g, d·g⟩ = ⟨q·g², d⟩` means only the query is transformed
at search time: intent-conditioned retrieval over the whole collection
costs one extra matrix-vector product. Without intents (or below the
inference confidence threshold) IntentDB behaves as a plain cosine vector
store.

## Development

```bash
python -m pytest          # 46 tests
python examples/demo.py   # end-to-end demo
```
