"""Smoke test for route detector feature extraction.

Doesn't require sklearn — just shows feature distributions per route to verify
the routes are linearly separable on (latency, num_sources). If this looks
clean, the decision tree should hit >95% accuracy.
"""

import json
import statistics
from pathlib import Path


def main() -> None:
    base = Path("results/final")
    routes = ("A", "B", "C")
    by_route = {r: {"latency_ms": [], "num_sources": [], "answer_words": []} for r in routes}

    for r in routes:
        path = base / f"baseline_musique_route_{r}.jsonl"
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                by_route[r]["latency_ms"].append(float(rec.get("latency_ms") or 0.0))
                by_route[r]["num_sources"].append(int(rec.get("num_sources") or 0))
                by_route[r]["answer_words"].append(len((rec.get("system_answer") or "").split()))

    for r in routes:
        d = by_route[r]
        n = len(d["latency_ms"])
        if not n:
            print(f"Route {r}: no data")
            continue
        print(f"Route {r} (n={n})")
        for feat in ("latency_ms", "num_sources", "answer_words"):
            vals = d[feat]
            print(
                f"  {feat:18s}  mean={statistics.mean(vals):8.1f}  "
                f"median={statistics.median(vals):8.1f}  "
                f"min={min(vals):8.1f}  max={max(vals):8.1f}"
            )
        print()

    print("=" * 60)
    print("Naive rule-based detector (sanity check):")
    print("=" * 60)
    rule = lambda lat, src: "A" if src == 0 else ("B" if lat < 12000 else "C")
    correct = 0
    total = 0
    for r in routes:
        d = by_route[r]
        for lat, src in zip(d["latency_ms"], d["num_sources"]):
            pred = rule(lat, src)
            total += 1
            if pred == r:
                correct += 1
    print(f"Naive rule accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
