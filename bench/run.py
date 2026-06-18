"""Run the ablation grid across tracks and embedders, with significance.

    python -m bench.run                              # both tracks, hashing
    python -m bench.run --embedder ollama:model=nomic-embed-text
    python -m bench.run --embedder hashing:dim=512,ollama:model=nomic-embed-text
    python -m bench.run --track hard                 # only the hard track
    python -m bench.run --quick                      # skip rerank configs
    python -m bench.run --out bench/RESULTS.md       # also write a report

Two tracks:

- **easy** — topical ambiguity ("python" the snake vs. the language); a
  diagonal lens solves it by gating topic dimensions, so the full stack
  nearly saturates.
- **hard** — pragmatic intents (tutorial / reference / troubleshooting /
  concept) over a shared-topic corpus; every document for a topic shares
  its vocabulary, so the intent, not the query, must pick the answer. This
  leaves headroom for evaluating further ranking refinements.

Each track reports the ablation grid over its load-bearing (paired) cases,
and a paired bootstrap CI on the full-vs-plain nDCG@10 delta (significance).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from . import dataset, dataset_hard, harness, metrics

TRACKS = {"easy": dataset, "hard": dataset_hard}


def _flashrank_available() -> bool:
    return importlib.util.find_spec("flashrank") is not None


def _ascii_bar(value: float, width: int = 24) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "-" * (width - filled)


def run_track(embedder: str, data, configs: list[str]) -> dict:
    """Evaluate every config on a track's paired (load-bearing) cases."""
    db = harness.build_db(embedder, data=data)
    try:
        paired = harness.paired_cases(data.CASES)
        results = {}
        for name in configs:
            results[name] = harness.evaluate(db, paired, harness.CONFIGS[name])
        # significance: full vs plain, paired per-case nDCG@10
        delta, lo, hi = metrics.paired_delta_ci(
            results["full"]["ndcg_per_case"], results["plain"]["ndcg_per_case"]
        )
        return {"n": len(paired), "results": results, "sig": (delta, lo, hi)}
    finally:
        db.close()


def _print_track(track: str, embedder: str, run: dict, configs: list[str]) -> None:
    print(f"\n## {track} track | {embedder} | {run['n']} paired cases")
    for name in configs:
        r = run["results"][name]
        lo, hi = r["ndcg@10_ci"]
        print(
            f"  {name:20s} top1={r['top1']:.0%}  "
            f"nDCG@10={r['ndcg@10']:.3f} [{lo:.3f},{hi:.3f}]  p-MRR={r['p-mrr']:+.3f}"
        )
    d, lo, hi = run["sig"]
    verdict = "significant" if lo > 0 else "not significant"
    print(f"  full vs plain nDCG@10 delta: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] ({verdict})")


def _md_track(track: str, embedder: str, run: dict, configs: list[str]) -> list[str]:
    lines = [
        f"### {track} track · `{embedder}` · {run['n']} paired cases",
        "",
        "| config | top-1 | nDCG@10 (95% CI) | p-MRR |",
        "|---|---|---|---|",
    ]
    for name in configs:
        r = run["results"][name]
        lo, hi = r["ndcg@10_ci"]
        lines.append(
            f"| {name} | {r['top1']:.0%} | {r['ndcg@10']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"| {r['p-mrr']:+.3f} |"
        )
    lines += ["", "```"]
    for name in configs:
        r = run["results"][name]
        lines.append(f"{name:20s} {_ascii_bar(r['ndcg@10'])} {r['ndcg@10']:.3f}")
    lines += ["```", ""]
    d, lo, hi = run["sig"]
    verdict = "**significant**" if lo > 0 else "not significant"
    lines += [
        f"Full vs plain nDCG@10 delta (paired bootstrap 95% CI): "
        f"**{d:+.3f}** [{lo:+.3f}, {hi:+.3f}] — {verdict}.",
        "",
    ]
    return lines


def render_markdown(matrix: list[tuple[str, str, dict]], configs: list[str]) -> str:
    lines = [
        "# IntentDB benchmark results",
        "",
        "Two tracks. **Easy** = topical ambiguity (a diagonal lens nearly "
        "saturates it). **Hard** = pragmatic intents (tutorial / reference / "
        "troubleshooting / concept) over a shared-topic corpus, which leaves "
        "headroom for evaluating further ranking refinements.",
        "",
        f"Easy track: {len(dataset.DOCS)} docs, {len(dataset.INTENTS)} intents. "
        f"Hard track: {len(dataset_hard.DOCS)} docs, {len(dataset_hard.INTENTS)} "
        "intents (one tutorial/reference/troubleshooting/concept doc per topic).",
        "Reproduce: `python -m bench.run --embedder <spec> --out bench/RESULTS.md`",
        "",
    ]
    last_track = None
    for track, embedder, run in matrix:
        if track != last_track:
            lines.append(f"## {track.capitalize()} track")
            lines.append("")
            last_track = track
        lines += _md_track(track, embedder, run, configs)
    lines += [
        "## Notes",
        "",
        "- **p-MRR** is the paired reciprocal-rank delta (FollowIR-style); ~0 "
        "means the configuration is blind to intent (plain cosine returns the "
        "same ranking for every intent, so its p-MRR is exactly 0).",
        "- The significance line is a paired bootstrap CI on the per-case "
        "nDCG@10 difference between the full stack and plain cosine.",
        "- Rerank rows use FlashRank's TinyBERT — topical steering from the "
        "injected intent text, not true instruction following (see RESEARCH.md).",
        "- The hard track's headroom (the full stack well below 1.0) leaves "
        "room for future ranking refinements such as per-intent low-rank adapters.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="IntentDB ablation benchmark")
    p.add_argument("--embedder", default="hashing:dim=512",
                   help="one spec, or several comma-separated")
    p.add_argument("--track", choices=["easy", "hard", "both"], default="both")
    p.add_argument("--quick", action="store_true", help="skip rerank configs")
    p.add_argument("--out", help="write a markdown report to this path")
    args = p.parse_args(argv)

    configs = list(harness.CONFIGS)
    if args.quick or not _flashrank_available():
        skipped = [c for c in configs if c in harness.RERANK_CONFIGS]
        configs = [c for c in configs if c not in harness.RERANK_CONFIGS]
        if skipped:
            reason = "--quick" if args.quick else "flashrank not installed"
            print(f"(skipping rerank configs: {reason})")

    embedders = [e.strip() for e in args.embedder.split(",") if e.strip()]
    tracks = ["easy", "hard"] if args.track == "both" else [args.track]

    matrix = []
    for track in tracks:
        for embedder in embedders:
            run = run_track(embedder, TRACKS[track], configs)
            _print_track(track, embedder, run, configs)
            matrix.append((track, embedder, run))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(matrix, configs) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
