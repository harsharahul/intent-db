# Changelog

All notable changes to IntentDB are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [0.2.3] - 2026-06-24

### Added
- `delete_many(doc_keys)`: remove many documents in one pass. The store deletes
  them in a single transaction (chunked to stay under SQLite's bound-variable
  limit) and the in-memory matrix and intent-affinity arrays are compacted once
  with a boolean mask, instead of one O(N) rebuild per key.

### Changed
- `add_many` now persists every document in a single transaction and grows the
  in-memory matrix and affinity arrays once per batch, instead of committing and
  reallocating per row. This makes bulk indexing of a large corpus markedly
  faster while preserving upsert (replace-by-key) semantics.
- The store sets `PRAGMA busy_timeout=5000`, so a second process writing to the
  same index waits briefly for the lock instead of failing with "database is
  locked".

## [0.2.2] - 2026-06-23

### Fixed
- `OllamaEmbedder` now batches embeddings through the `/api/embed` endpoint with
  a retry on transient 5xx, instead of one `/api/embeddings` request per
  document. This is much faster and avoids the HTTP 500s that a long burst of
  single-document calls could trigger on the Ollama server.

## [0.2.1] - 2026-06-23

### Changed
- Published to PyPI as `intent-vector-db`. The import package `intentdb` and the
  `intentdb` CLI command are unchanged; only the distribution name differs.
- Added a PyPI release workflow using trusted publishing (OIDC).

## [0.2.0] - 2026-06-17

### Added
- Optional cross-encoder reranker stage: `query(rerank=...)` with
  `rerank_depth`. FlashRank (ONNX, CPU) and sentence-transformers backends
  selected by spec string, with the active intent's instruction injected
  into each (query, document) pair. Available through the CLI (`--rerank`,
  `--reranker`, `--rerank-depth`) and the MCP query tool.
- Evaluation benchmark (`bench/`): a paired-intent suite with an ablation
  grid (plain, inferred-intent, lens-only, affinity-only, full, and the
  hybrid/rerank variants), `nDCG@10`, `MRR`, `p-MRR`, and `robustness@10`,
  each with a bootstrap confidence interval, plus a paired significance
  test, two difficulty tracks, and a train/test split with a feedback
  sampler. Runs as `python -m bench.run`.
- Agent-memory example (`examples/agent_memory.py`,
  `examples/AGENT_MEMORY.md`): one store recalled under the agent's current
  phase, with a Claude Code / MCP integration recipe.
- `CONTRIBUTING.md` and this changelog.

### Changed
- README now opens with the agent-memory use case and the benchmark
  results.
- Added a curated references list (`REFERENCES.md`).

## [0.1.0] - 2026-06

### Added
- Initial release: intent-aware vector database with the diagonal intent
  lens, intent affinity, hybrid BM25 + RRF fusion, intent-aware
  pseudo-relevance feedback, feedback-learned fusion weights, query-log
  intent mining, pluggable embedders (hashing / Ollama /
  sentence-transformers), SQLite (WAL) persistence, a CLI, and an MCP
  stdio server. NumPy-only core, MIT licensed.
