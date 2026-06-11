# IntentDB — Design Plan

## 1. Vision

A vector database with one extra dimension: **intent**. Ordinary vector
databases embed text once and answer every query against that single, fixed
geometry. IntentDB makes the geometry *conditional*: the same corpus, the
same query, but a different active intent produces different effective
vectorizations — and therefore different results. Data collection is
general-purpose; retrieval is built for an LLM consumer (structured
results, score breakdowns, an MCP server interface).

## 2. What the research says (and how it shaped the design)

The plan was checked against current retrieval research and the
architecture of existing local vector stores before finalizing.

**Intent-conditioned embeddings are real and state-of-the-art.**
[INSTRUCTOR](https://arxiv.org/abs/2212.09741) ("One Embedder, Any Task")
embeds every text *together with a task instruction*, so the same input
maps to different vectors per task. Production embedders are already
instruction/prefix-aware: [nomic-embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
requires `search_query:` / `search_document:` task prefixes, e5 uses
`query:` / `passage:`. **Design consequence:** IntentDB's embedder API is
asymmetric (`embed_query` vs `embed_document`) and instruction-aware; an
intent carries an `instruction` that re-conditions the query embedding
when the model supports it. For embedders that can't use instructions, the
*lens* mechanism (below) provides intent conditioning model-agnostically.

**Per-intent document re-embedding does not scale.** Embedding every
document once per intent costs `O(docs × intents)` embeddings and storage,
and adding an intent would force a full re-embed of the corpus.
**Design consequence:** documents are embedded **once**; intent
conditioning happens (a) on the query side (instructions, lens) and (b)
via cheap precomputed per-(doc, intent) scalars (affinity). Registering a
new intent over a million documents is one matrix–vector product, not a
million model calls.

**Local architecture: SQLite + in-memory matrix is the proven shape.**
This is essentially [how Chroma works under the hood](https://dev.to/svemaraju/vector-database-what-is-chromadb-doing-under-the-hood-177j)
(SQLite for storage, an in-memory index for search). Brute-force exact
search is fine into the hundreds of thousands of vectors; ANN indexes
(HNSW) only pay off [at larger scales](https://anandtopu.medium.com/vector-databases-from-scratch-part-3-chromadb-hands-on-872bf28a4a58),
at the cost of memory-resident graphs and approximate recall.
**Design consequence:** v1 uses exact NumPy scoring (one matvec per query,
plus one per active intent) over a float32 matrix mirrored from SQLite.
HNSW is roadmap, not v1 — correctness and the intent mechanism first.

**Modern RAG fuses multiple ranked signals.** Best practice is
[hybrid dense+sparse retrieval fused with RRF, then cross-encoder reranking](https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/)
([overview](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)).
**Design consequence:** IntentDB's score is an explicit weighted fusion of
three signals (lensed similarity, intent affinity, base similarity) with
user-tunable weights, and every result reports its per-signal breakdown so
a downstream LLM or reranker can re-fuse. Cross-encoder reranking and RRF
are roadmap stages that slot in after candidate generation.

## 3. Architecture

```
                ┌────────────────────────────────────────────────┐
                │                   IntentDB                     │
                │                                                │
  add(text) ───▶  Embedder.embed_document ──▶ base vector ─────▶│ SQLite
                │                              │                 │  documents
                │            per registered intent:              │  intents
                │            affinity = cos(doc, intent) ───────▶│  doc_intent
                │                                                │
 register_intent│  Intent.build: vector = centroid(desc+exemplars)
      ─────────▶│               lens   = diagonal gate fit on    │
                │                        exemplar statistics     │
                │                                                │
 query(text,    │  q = embed_query(text [, intent.instruction])  │
   intent?) ───▶│  no intent given? infer from intent vectors    │
                │                                                │
                │  score = w_l · <q·g², D> / ‖q·g‖   (lensed)    │
                │        + w_a · affinity(D, intent)             │
                │        + w_b · cos(q, D)            (base)     │
                └────────────────────────────────────────────────┘
                          │
                          ▼
        Python API  ·  CLI (`intentdb`)  ·  MCP stdio server (for LLMs)
```

### The three intent mechanisms

1. **Instruction conditioning** (model-level, strongest with neural
   embedders): the active intent's instruction is folded into query
   embedding, INSTRUCTOR-style. The query's vector itself moves.
2. **The lens** (model-agnostic): each intent owns a diagonal gate `g ≥ 1`
   over embedding dimensions, fit from the intent's description/exemplars
   with a Fisher-style relevance score `mean²/(var+ε)` — dimensions the
   intent's examples consistently activate get amplified. Lensed
   similarity is `⟨q·g, d·g⟩ / ‖q·g‖`: overlap on intent-characteristic
   dimensions counts more. The document side keeps its base norm
   deliberately — re-normalizing would *penalize* documents rich in
   intent-relevant content (this was caught and fixed by test). The
   identity `⟨q·g, d·g⟩ = ⟨q·g², d⟩` means only the query is transformed:
   intent retrieval costs one extra matvec, nothing per-document.
3. **Affinity** (precomputed): `cos(doc, intent_vector)`, stored per
   (document, intent) at ingest — "does this document belong to this
   intent at all", independent of the query.

### Intent inference

Queries without a declared intent are classified against registered intent
vectors; below a confidence threshold the database degrades gracefully to
plain cosine search. `explain()` exposes the classifier's view.

## 4. Known limitations (deliberate v1 trade-offs)

- **Exact brute-force search** — O(N·d) per query. Fine locally to ~10⁵–10⁶
  docs; beyond that an ANN index is needed (roadmap).
- **Diagonal lens** — a diagonal metric can only re-weight existing
  dimensions, not rotate the space. With strongly distributed neural
  embeddings its effect is milder than with lexical embeddings; that is
  exactly why instruction conditioning is the primary mechanism for neural
  embedders and the lens is the universal fallback.
- **Hashing embedder is lexical** — token/n-gram overlap, not semantics.
  It exists so the system runs with zero dependencies; plug Ollama or
  sentence-transformers for true semantic retrieval.
- **SQLite WAL = single writer** — many readers, one writer; fine for a
  local/per-agent database, not for multi-writer servers.
- **BM25 scoring is a Python loop over postings** — fast for short
  queries on local corpora, slower than the vectorized dense path on very
  large vocabularies; only runs when ``hybrid=True``.

## 5. Implementation phases

- **Phase 1 — Core engine (done):** embedder abstraction (hashing /
  Ollama / sentence-transformers), instruction-aware asymmetric embedding,
  intents with lens + affinity, fused scoring, intent inference, SQLite
  persistence, metadata filtering, upsert/delete.
- **Phase 2 — Interfaces (done):** Python API, full CLI, MCP stdio server
  exposing query/add/intents/explain/stats tools to any MCP client.
- **Phase 3 — Quality (done):** 46-test suite covering the math identities,
  ranking flips under intent, persistence, CLI, and MCP protocol; demo
  script.
- **Phase 4 — Scale (roadmap):** HNSW or IVF index behind the same query
  API; memory-mapped vector matrix; batched/streaming ingest.
- **Phase 5 — Retrieval quality (done):** incremental BM25 index over the
  corpus with hybrid dense+sparse retrieval fused by Reciprocal Rank
  Fusion (`query(..., hybrid=True)`); paragraph/sentence chunking with
  overlap (`add_chunked`, CLI `--chunk`). Still roadmap: cross-encoder
  reranking; per-intent learned weights from relevance feedback.
- **Phase 6 — Intent learning (done, first iteration):** every query is
  logged (capped, opt-out with `log=False`); `suggest_intents()` clusters
  undeclared queries with spherical k-means and proposes intents with
  exemplar queries (CLI `suggest-intents`, MCP `intentdb_suggest_intents`)
  — an LLM or human names them and calls `register_intent`. Still
  roadmap: online lens updates from clicks / LLM feedback.
