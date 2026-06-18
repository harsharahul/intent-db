# Changelog

All notable changes to IntentDB are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

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
