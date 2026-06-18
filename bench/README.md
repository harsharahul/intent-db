# IntentDB benchmark

A paired-intent benchmark: the same query under different intents has a
different correct document, so a retriever with one fixed geometry cannot
score well on the ambiguous cases. This harness measures how much each
retrieval signal contributes, with standard IR metrics and bootstrap
confidence intervals.

## Run it

```bash
# zero-dependency, deterministic (the lexical hashing embedder)
python -m bench.run

# headline numbers on a real semantic embedder (needs a local Ollama)
python -m bench.run --embedder ollama:model=nomic-embed-text --out bench/RESULTS.md

# skip the cross-encoder rerank configs (no optional model needed)
python -m bench.run --quick
```

Rerank configs need the optional `flashrank` dependency
(`pip install -e .[rerank]`); they are skipped automatically when it is
absent. `flashrank` requires `onnxruntime`, which has no wheel for some
newest Python builds, so use a Python 3.11-3.13 environment for the rerank
rows.

## What it reports

Two slices, each as an ablation grid:

- **Full set**: ambiguous bare terms mixed with phrased and single-sense
  queries (the realistic mix a corpus sees). Shows that intent
  conditioning does not harm the easy cases.
- **Ambiguous slice**: only queries that appear under more than one intent
  with a different gold document. This isolates what intent conditioning
  buys, because a single fixed ranking provably cannot satisfy it.

Configs: `plain` (cosine, no intent, the normal-vector-DB baseline),
`auto-intent` (intent inferred from the query), `lens-only` /
`affinity-only` (one signal isolated), `full` (the default blend), and the
`+hybrid` / `+rerank` upgrades.

Metrics: top-1 accuracy, `nDCG@10` (with a percentile bootstrap 95% CI),
`MRR`, `p-MRR` (FollowIR-style paired reciprocal-rank delta, intent
sensitivity), and `robustness@10` (each query's worst nDCG across its
intents). See [RESULTS.md](RESULTS.md) for a committed run.

## Headline (nomic-embed-text, ambiguous slice)

| config | top-1 | p-MRR |
|---|---|---|
| plain cosine | 44% | +0.000 |
| intent (full) | 96% | +0.766 |

Plain cosine is blind to intent by construction (identical ranking for
every intent, so p-MRR is exactly 0). Intent conditioning more than
doubles top-1 on the cases where intent matters. A finding worth keeping
honest: with a well-fit lens on a strong embedder, the lens alone carries
almost all of the gain. Stacking hybrid BM25 or a small cross-encoder on
top *dilutes* an already-strong signal here rather than helping, which is
the argument for learned, per-intent fusion rather than blindly combining
every signal.

## Layout

```
bench/
  dataset.py   paired-intent corpus, intents, and (query, intent, gold) cases
  metrics.py   ndcg@k, reciprocal rank, p-MRR, robustness, bootstrap CI
  harness.py   build_db, run_config, evaluate, paired_cases
  run.py       ablation grid -> console table + markdown report
  RESULTS.md   a committed run on nomic-embed-text
```

The metrics and harness are covered by `tests/test_bench.py`, including a
regression guard that the full stack beats plain cosine.
