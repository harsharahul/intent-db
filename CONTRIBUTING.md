# Contributing to IntentDB

Thanks for your interest. IntentDB is a small, deliberately focused
project; contributions that keep it that way are very welcome.

## Principles

- **Pure-NumPy core.** The only required runtime dependency is NumPy.
  Anything heavier (embedding models, rerankers) goes behind an optional
  dependency and degrades gracefully when absent.
- **Every change lands with tests.** The suite runs in ~1s; there is no
  excuse to skip it. New behavior gets a test that fails before the change
  and passes after.
- **Accuracy over hype.** Claims in the README and docs should hold up
  against the literature; cite sources in `REFERENCES.md`. If a measurement
  is noisy or a result is negative, say so.

## Development

```bash
pip install -e .[dev]        # NumPy + pytest
python -m pytest -q          # the test suite (should be all green)
python examples/demo.py      # end-to-end sanity
python examples/agent_memory.py
python -m bench.run --quick  # the benchmark (hashing, no extra deps)
```

Optional extras: `.[ollama]` (local Ollama embeddings, stdlib only),
`.[sbert]` (sentence-transformers), `.[rerank]` (FlashRank cross-encoder;
needs a Python where `onnxruntime` has a wheel).

## Submitting changes

1. Branch off `main`.
2. Make the change with tests; keep the suite green and the example
   scripts running.
3. Match the surrounding style: type hints, concise docstrings that state
   constraints, no emojis in code or docs.
4. Open a pull request describing what changed and why. CI runs the test
   matrix (Python 3.10-3.13), the examples, and the benchmark smoke test.

## Scope

IntentDB is the *ranking* layer for intent-conditioned retrieval and agent
memory. Things that fit: retrieval signals, embedders, evaluation, the MCP
surface, performance within the pure-NumPy budget. Things that do not:
becoming a general-purpose ANN engine, a multi-writer server, or a full
memory-lifecycle platform. When in doubt, open an issue first.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
