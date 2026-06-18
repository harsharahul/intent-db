# IntentDB

**A local-first vector database with an extra dimension: intent.**

[![CI](https://github.com/harsharahul/intent-db/actions/workflows/ci.yml/badge.svg)](https://github.com/harsharahul/intent-db/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A conventional vector database embeds your data once and answers every
query against that single, fixed geometry. IntentDB makes the geometry
conditional on *intent*: the same corpus and the same query return
different results depending on what the asker is trying to do. When no
intent is declared, the most plausible one is inferred from the query
itself. Storage is general-purpose; retrieval is designed for an LLM
consumer: structured hits, per-signal score breakdowns, a built-in MCP
server, and a feedback loop the database learns from.

The use case it is built for is **agent memory**: one store that returns
the bug fix while the agent is *debugging*, the decision rationale while
it is *planning*, and the convention while it is *reviewing*. The same
query, ranked by what the agent is doing, sharpens each phase's
ranking from the agent's own feedback. A normal vector store returns the
right fact at the wrong moment; this one conditions on the moment. See the
[agent-memory walkthrough](examples/AGENT_MEMORY.md) (`python
examples/agent_memory.py`).

```text
query: "python"                       query: "python"                       query: "python"
intent: coding                        intent: wildlife                      intent: comedy
---------------------------           ---------------------------          ---------------------------
1. Python is a programming            1. The python is a large             1. Monty Python was a
   language; write code,                 snake, a reptile that                British comedy group
   functions, and modules...             lives in jungle habitats...          famous for sketch humor...
```

One SQLite file. Pure Python and NumPy. No services, no model downloads
required. A deterministic hashing embedder ships in the box, and Ollama
or sentence-transformers plug in for semantic embeddings.

**Does it work?** On a paired-intent benchmark (`bench/`) where the same
query under different intents has a different correct answer each time,
measured on the ambiguous queries where intent is load-bearing, with a real embedder
(`nomic-embed-text`):

| configuration | top-1 | p-MRR (intent sensitivity) |
|---|---|---|
| plain cosine (a normal vector DB) | 44% | +0.000 |
| **IntentDB (intent-conditioned)** | **96%** | **+0.766** |

Plain cosine is blind to intent by construction: it returns the identical
ranking whatever the intent, so its p-MRR is exactly zero. A harder track
(pragmatic intents over a shared-topic corpus) keeps real headroom. See
[`bench/`](bench/) to reproduce, and [`bench/RESULTS.md`](bench/RESULTS.md)
for the full ablation grid.

## Contents

- [Why intent-aware retrieval](#why-intent-aware-retrieval)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Use from an LLM (MCP server)](#use-from-an-llm-mcp-server)
- [The learning loop](#the-learning-loop)
- [How it works](#how-it-works)
- [Embedders](#embedders)
- [Reranking](#reranking)
- [Benchmark](#benchmark)
- [Docker](#docker)
- [Performance and limitations](#performance-and-limitations)
- [References](#references)
- [Development](#development)

## Why intent-aware retrieval

The same query rarely means one thing. "python" is a language, a snake,
and a comedy troupe; "java" is code, coffee, and an island. A plain vector
database collapses all of those into one similarity ranking. The standard
workarounds (rewriting queries by hand, separate collections per topic)
push the problem onto the caller.

IntentDB instead makes intent a first-class, registered object. Each
intent contributes three retrieval signals, fused into one score:

1. **Instruction conditioning.** With an instruction-aware embedder, the
   query is re-embedded under the intent's natural-language instruction,
   so the query vector itself moves (the INSTRUCTOR pattern).
2. **The intent lens.** Each intent owns a per-dimension gate over the
   embedding space, fitted from its description and exemplar queries.
   Query-document overlap on intent-characteristic dimensions is
   amplified. Model-agnostic, and only the query is transformed, so
   intent retrieval over the whole collection costs one extra
   matrix-vector product.
3. **Intent affinity.** Every document's cosine to every intent vector is
   precomputed at ingest: a prior for "does this document belong to this
   intent at all", independent of the query.

Documents are embedded exactly once. Registering a new intent over a
million documents is one matrix-vector product, not a million model
calls. Without intents, IntentDB degrades gracefully to an ordinary
cosine vector store.

## Installation

From source (Python 3.10+, the only runtime dependency is NumPy):

```bash
git clone https://github.com/harsharahul/intent-db
cd intent-db
pip install -e .          # library + intentdb CLI
pip install -e .[dev]     # + pytest
pip install -e .[rerank]  # + flashrank cross-encoder reranking
```

Or use the container image (see [Docker](#docker)).

## Quick start

```python
from intentdb import IntentDB

db = IntentDB("knowledge.intentdb")   # hashing embedder by default
# db = IntentDB("knowledge.intentdb", embedder="ollama:model=nomic-embed-text")

# Ingest. Documents are embedded once; re-adding a doc_key replaces it.
db.add("Python is a programming language; write code and debug programs.",
       doc_key="py-lang", metadata={"topic": "software"})
db.add("The python is a large snake, a reptile of jungle habitats.",
       doc_key="py-snake", metadata={"topic": "nature"})

# Long documents: chunk on paragraph/sentence boundaries with overlap.
db.add_chunked(long_text, doc_key="manual", max_chars=1200, overlap=200)

# Register intents: a description, optional exemplar queries, and an
# optional instruction for instruction-aware embedders.
db.register_intent("coding",
                   description="software programming, source code, debugging",
                   exemplars=["how do I write code", "debug my program"])
db.register_intent("wildlife",
                   description="animals, reptiles, snakes, jungle habitats",
                   exemplars=["what do snakes eat"])

# The headline behavior: same query, different intent, different results.
db.query("python", intent="coding")[0].doc_key     # -> "py-lang"
db.query("python", intent="wildlife")[0].doc_key   # -> "py-snake"

# No intent declared? It is inferred (and reported on the result).
hit = db.query("python snake habitat")[0]
hit.intent, hit.intent_inferred                    # -> ("wildlife", True)

# Every hit explains itself.
hit.score, hit.base_score, hit.lensed_score, hit.intent_affinity

# Optional retrieval upgrades, all local:
db.query("ECONNREFUSED billing", hybrid=True)      # BM25 + dense, RRF-fused
db.query("postgres", intent="coding", prf=True)    # pseudo-relevance feedback
db.query("python", intent="coding", rerank=True)   # cross-encoder rerank stage
db.query("python", intent="coding",
         where=lambda m: m.get("topic") == "software")   # metadata filter
```

## Command-line interface

```bash
intentdb init kb.intentdb
intentdb add kb.intentdb "Postgres uses MVCC for concurrency" --key pg-mvcc
intentdb add kb.intentdb --file manual.txt --chunk --key manual
intentdb intent add kb.intentdb debugging \
    --description "diagnosing errors and failures in software" \
    --exemplar "why is my service crashing"
intentdb query kb.intentdb "postgres locks" --intent debugging -k 3
intentdb query kb.intentdb "ECONNREFUSED" --hybrid --prf
intentdb query kb.intentdb "postgres locks" --intent debugging --rerank
intentdb query kb.intentdb "postgres locks" --json
intentdb explain kb.intentdb "why does my app crash"
intentdb feedback kb.intentdb "postgres locks" pg-mvcc --intent debugging
intentdb learn kb.intentdb
intentdb suggest-intents kb.intentdb
intentdb stats kb.intentdb
```

Run `intentdb <command> --help` for the full set of options.

## Use from an LLM (MCP server)

IntentDB ships a Model Context Protocol stdio server, so any MCP client
(Claude Code, Claude Desktop, local agent frameworks) can mount a database
as a retrieval tool:

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

| Tool | Purpose |
|---|---|
| `intentdb_query` | Search, optionally under a named intent (with hybrid/PRF options) |
| `intentdb_add` | Store a document |
| `intentdb_register_intent` | Register or redefine an intent |
| `intentdb_list_intents` | List registered intents |
| `intentdb_explain` | Show which intent the classifier infers for a query |
| `intentdb_record_feedback` | Report whether a retrieved document was useful |
| `intentdb_learn_fusion` | Learn per-intent signal weights from feedback |
| `intentdb_suggest_intents` | Mine the query log for undeclared intents |
| `intentdb_stats` | Database statistics |

**Agent memory** is the use case this is built for: one store, recalled by
the agent's current phase (planning / debugging / reviewing), so the same
query returns the decision rationale, the bug fix, or the review rule
depending on what the agent is doing, and the store learns each phase's
ranking from feedback. See
[`examples/AGENT_MEMORY.md`](examples/AGENT_MEMORY.md) for a runnable demo
(`python examples/agent_memory.py`) and a Claude Code wiring recipe.

## The learning loop

IntentDB improves from use, entirely locally:

1. **Query logging.** Every query is recorded (capped, `log=False` opts
   out for automated traffic).
2. **Intent mining.** `suggest_intents()` clusters the queries that ran
   without a declared intent and proposes new intents with exemplar
   queries; an LLM or a human names them and calls `register_intent`.
3. **Relevance feedback.** The consumer reports which results were
   actually useful via `record_feedback(query, doc_key, useful, intent)`.
4. **Learned fusion.** `learn_fusion_weights()` fits each intent's blend
   of the three signals from that feedback (a small pairwise logistic
   model; tuned linear fusion is the literature's sample-efficient
   winner). Learned weights persist and apply automatically; intents
   without enough feedback keep the defaults.

```python
db.record_feedback("research methods", "deep-paper", useful=True, intent="research")
db.record_feedback("research methods", "marketing-page", useful=False, intent="research")
db.learn_fusion_weights()        # {'research': {'lensed': ..., 'affinity': ..., 'base': ...}}
```

## How it works

```text
              WRITE PATH                              READ PATH
  add(text) ──> embed once ──> SQLite          query(text, intent?)
                  │             ├ documents+vectors      │
                  │             ├ intents (vector,       ▼
   per intent:    │             │  lens, mu/sigma)   declared? ──no──> infer from
   affinity ──────┘             ├ doc_intent             │             intent vectors
                                ├ query_log              ▼
   register_intent(...)         └ feedback     embed query (conditioned on
   fits vector + lens,                         the intent's instruction)
   indexes all docs in                                   │
   one matvec                                            ▼
                                       score = w_l * lensed + w_a * affinity
                                             + w_b * base   (+ BM25 via RRF,
                                               + Rocchio PRF second pass)
```

For query `q`, document `d` (both unit norm), intent vector `t`, lens
gate `g`, and the intent's corpus statistics `(mu, sigma)`:

```text
base      = cos(q, d)
lensed    = <q_s * g, d_s * g> / (|q_s * g| * mean-doc-norm)
            where x_s = (x - mu) / sigma         (standardized basis)
affinity  = cos(d, t)                            (precomputed at ingest)
score     = w_lensed * lensed + w_affinity * affinity + w_base * base
```

Design decisions worth knowing:

- **One matvec per intent.** The identity `<q*g, d*g> = <q*g^2, d>` means
  only the query is transformed at search time.
- **Standardized lens basis.** Dense-embedding dimensions are dominated by
  anisotropy artifacts ("rogue dimensions"); lenses are fitted and applied
  in a corpus-standardized basis so gates measure intent, not artifacts.
- **Double shrinkage.** Gates shrink toward the identity when an intent
  has few exemplars, and the corpus statistics themselves shrink toward
  the raw basis when the collection is small. Both estimators are only
  trusted in proportion to their data.
- **Asymmetric lensed similarity.** The document side keeps its base norm;
  re-normalizing documents in the lensed space would penalize documents
  rich in intent-relevant content.

## Embedders

| Spec | Description |
|---|---|
| `hashing:dim=512` (default) | Deterministic lexical feature hashing. Zero dependencies, no downloads, fully reproducible. Quality is lexical: good for tests, demos, keyword-ish corpora. |
| `ollama:model=nomic-embed-text` | Any local Ollama embedding model. Task prefixes via `prefix_mode=nomic\|e5\|none`; intent instructions are injected into the query. |
| `sbert:model=all-MiniLM-L6-v2` | sentence-transformers models (`pip install -e .[sbert]`). |

The embedder spec is stored inside the database and restored on reopen;
dimension mismatches are rejected. Note that small bi-encoders treat
instructions mostly as additional keywords (a soft topical bias) rather
than true semantic constraints. The lens and affinity signals carry the
intent conditioning for those models. See [REFERENCES.md](REFERENCES.md).

## Reranking

`query(rerank=True)` re-scores the top candidates (default 20) with a
cross-encoder and orders results by that score, the best-documented
quality jump over pure bi-encoder retrieval. When an intent is active its
instruction is prefixed to the query before scoring; unlike small
bi-encoders, a cross-encoder reads the query and document jointly, so
this is where the intent text actually changes the model's judgment.

| Spec | Description |
|---|---|
| `flashrank` (default for `rerank=True`) | ONNX cross-encoders on CPU, no torch (`pip install -e .[rerank]`). The default model, ms-marco-TinyBERT-L-2-v2, is a ~4 MB download on first use. |
| `crossencoder:model=cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers cross-encoders (`pip install -e .[sbert]`). |

```python
db.query("python", intent="coding", rerank=True)
db.query("python", intent="coding", rerank="crossencoder:model=cross-encoder/ms-marco-MiniLM-L-6-v2",
         rerank_depth=50)
```

Reranking composes with `hybrid` and `prf` (it always runs last), respects
`where` filters, and reports its value on each hit as `rerank_score`.
Models are loaded once per database instance and cached.

## Benchmark

[`bench/`](bench/) is a paired-intent benchmark: the same query appears
under several intents with a different correct document each time, so a
retriever with one fixed geometry cannot satisfy them. It reports an
ablation grid (plain cosine, inferred intent, each signal in isolation, the
full blend, and the hybrid/rerank upgrades) with `nDCG@10`, `MRR`, `p-MRR`,
and `robustness@10`, each with a bootstrap 95% CI, plus a paired bootstrap
CI on the full-vs-plain delta (significance).

```bash
python -m bench.run                                          # both tracks, deterministic
python -m bench.run --embedder ollama:model=nomic-embed-text # semantic embedder
python -m bench.run --embedder hashing:dim=512,ollama:model=nomic-embed-text
```

Two tracks:

- **Easy**: topical ambiguity (`python` the snake vs. the language). A
  diagonal lens solves it by gating topic dimensions, so the full stack
  nearly saturates: **plain cosine 44% → intent 96% top-1** on the
  ambiguous slice (nomic-embed-text), delta +0.21 nDCG@10 [+0.13, +0.29],
  significant. The honest finding here: with a well-fit lens the **lens
  alone** carries the gain, and stacking hybrid/rerank on top *dilutes* an
  already-strong signal. That argues for learned, per-intent fusion over
  blind signal-stacking.
- **Hard**: pragmatic intents (`tutorial` / `reference` / `troubleshooting`
  / `concept`) over a shared-topic corpus, where every document for a topic
  shares its vocabulary so the intent, not the query, must pick the answer.
  This leaves headroom: the full stack reaches **62% top-1 / 0.80 nDCG@10**
  (nomic), well below ceiling, with the diagonal lens doing most of the work
  and roughly twenty points still open for future ranking refinements such
  as per-intent low-rank query adapters.

Full numbers in [`bench/RESULTS.md`](bench/RESULTS.md).

## Docker

A multi-stage image is built by CI and published to GitHub Container
Registry on pushes to `main` and on version tags:

```bash
docker pull ghcr.io/harsharahul/intent-db:latest
```

Or build locally:

```bash
docker build -t intentdb .
```

The image's entrypoint is the `intentdb` CLI and `/data` is the working
volume:

```bash
# CLI usage
docker run --rm -v "$PWD/data:/data" intentdb init /data/kb.intentdb
docker run --rm -v "$PWD/data:/data" intentdb add /data/kb.intentdb "some text" --key t1
docker run --rm -v "$PWD/data:/data" intentdb query /data/kb.intentdb "some text"

# MCP server over stdio (note -i)
docker run --rm -i -v "$PWD/data:/data" intentdb serve-mcp /data/kb.intentdb
```

To use the containerized MCP server from an MCP client:

```jsonc
{
  "mcpServers": {
    "intentdb": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-v", "/abs/path/data:/data",
               "ghcr.io/harsharahul/intent-db:latest",
               "serve-mcp", "/data/kb.intentdb"]
    }
  }
}
```

## Performance and limitations

Honest numbers and trade-offs, by design:

- **Exact search, O(N·d) per query.** Brute-force NumPy scoring is exact
  and fast into the hundreds of thousands of documents on a laptop. ANN
  indexing (IVF/HNSW) is on the roadmap for beyond that.
- **SQLite WAL: many readers, one writer.** Right for a local or
  per-agent database; wrong for a multi-writer server.
- **BM25 scoring is a Python loop over postings.** Only runs with
  `hybrid=True`; fine for short queries on local corpora.
- **The lens is diagonal.** It re-weights dimensions but cannot rotate
  the space. Per the sample-complexity literature this is the only sound
  regime for few-exemplar intents; the documented upgrade path is
  per-intent low-rank query adapters once feedback accumulates.
- **The hashing embedder is lexical.** Plug in Ollama or
  sentence-transformers for semantic retrieval.

## References

IntentDB's design builds on prior work in instruction-conditioned
embeddings (INSTRUCTOR, TART, Promptriever; the FollowIR and InstructIR
benchmarks), diagonal and low-rank metric learning (Schultz & Joachims;
ITML; Verma & Branson), embedding-space corrections (Timkey & van
Schijndel; whitening), pseudo-relevance feedback (Rocchio; vector-PRF),
score fusion (RRF; Bruch et al.), and query-log intent mining (Beeferman &
Berger through TnT-LLM). See [REFERENCES.md](REFERENCES.md) for the full
list with links.

## Development

```bash
pip install -e .[dev]
python -m pytest          # 91 tests
python examples/demo.py   # end-to-end demo
```

Repository layout:

```text
src/intentdb/
  db.py           core engine: scoring, PRF, feedback, learned fusion
  intent.py       intents, lenses, standardization, inference
  embedders.py    hashing / Ollama / sentence-transformers adapters
  store.py        SQLite persistence and migrations
  lexical.py      incremental BM25 index and RRF
  fusion.py       learned fusion weights from preference pairs
  mining.py       query-log clustering for intent suggestions
  chunking.py     paragraph/sentence chunker with overlap
  cli.py          command-line interface
  mcp_server.py   MCP stdio server for LLM clients
tests/            pytest suite
examples/demo.py  runnable end-to-end demonstration
```

CI runs the test suite on Python 3.10 through 3.13 and builds the Docker
image on every push and pull request; images publish to GHCR from `main`
and version tags.

## License

[MIT](LICENSE)
