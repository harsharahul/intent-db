"""IntentDB evaluation harness.

A paired-intent benchmark: the same query under different intents has
different correct answers, so retrieval that ignores intent cannot score
well. This package measures how much each retrieval signal (lens,
affinity, hybrid BM25, cross-encoder rerank) contributes, with standard
IR metrics and bootstrap confidence intervals.

Not part of the shipped library — a research/CI tool. Run it with
``python -m bench.run`` (see ``bench/README.md``).
"""
