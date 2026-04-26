"""
Analyze Phase 1 attack results.

Produces:
  - Per-route attack success rates (with bootstrap CIs)
  - Mechanism breakdown: route_flip vs same_route_answer_changed vs failed
  - Average queries-to-success
  - Common successful swap patterns
  - Cases where baseline was correct but attack flipped to wrong-by-EM
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib


def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def extract_features(record: Dict[str, Any]) -> List[float]:
    answer = record.get("system_answer") or ""
    latency = float(record.get("latency_ms") or 0.0)
    num_sources = int(record.get("num_sources") or 0)
    return [
        latency,
        float(np.log1p(latency)),
        float(num_sources),
        float(len(answer)),
        float(len(answer.split())),
    ]


def predict_route(detector, record: Dict[str, Any]) -> str:
    """Use detector to label what route a perturbed-query observation came from."""
    feats = np.asarray(extract_features(record)).reshape(1, -1)
    return str(detector.predict(feats)[0])


def bootstrap_ci(values: List[int], n_iter: int = 10000, alpha: float = 0.05) -> tuple:
    """Bootstrap confidence interval for proportion or mean."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values)
    means = []
    n = len(arr)
    for _ in range(n_iter):
        sample = np.random.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", type=Path, default=Path("results/attacks/phase1"))
    ap.add_argument(
        "--detector",
        type=Path,
        default=Path("results/final/route_detector.joblib"),
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/attacks/phase1/phase1_analysis.json"),
    )
    args = ap.parse_args()

    # Load route detector
    print(f"[1/4] Loading route detector from {args.detector}...")
    bundle = joblib.load(args.detector)
    detector = bundle["model"]
    print(f"  classes: {bundle['classes']}")

    # Load attack results
    print(f"\n[2/4] Loading attack results from {args.phase1_dir}...")
    attacks_by_route = defaultdict(list)
    for route in ("A", "B", "C"):
        path = args.phase1_dir / f"phase1_{route}.jsonl"
        if not path.exists():
            print(f"  {route}: skipped (no file)")
            continue
        with open(path) as f:
            for line in f:
                attacks_by_route[route].append(json.loads(line))
        print(f"  {route}: {len(attacks_by_route[route])} attack records")

    # Per-attack analysis
    print(f"\n[3/4] Analyzing each attack...")
    results = {
        "per_route": {},
        "mechanism_breakdown": {},
        "successful_swaps": Counter(),
        "questions_with_route_flip": [],
    }

    for route, attacks in attacks_by_route.items():
        per_route = {
            "n": len(attacks),
            "successes": 0,
            "failures": 0,
            "queries_used": [],
            "queries_to_success": [],
            "candidates_generated": [],
            "candidates_filtered_sim": [],
            "candidates_filtered_ppl": [],
            "wall_clock_s": [],
            "outcome_counts": Counter(),
        }
        for atk in attacks:
            success = bool(atk.get("attack_success_strict") or atk.get("attack_success"))
            per_route["queries_used"].append(int(atk.get("queries_used") or 1))
            per_route["candidates_generated"].append(int(atk.get("num_candidates_generated") or 0))
            per_route["candidates_filtered_sim"].append(int(atk.get("num_candidates_filtered_sim") or 0))
            per_route["candidates_filtered_ppl"].append(int(atk.get("num_candidates_filtered_ppl") or 0))
            per_route["wall_clock_s"].append(float(atk.get("wall_clock_s") or 0.0))
            if success:
                per_route["successes"] += 1
                per_route["queries_to_success"].append(int(atk["queries_used"]))
                # Track successful swap
                swap = atk.get("successful_swap") or {}
                if swap.get("original") and swap.get("replacement"):
                    results["successful_swaps"][f"{swap['original']} -> {swap['replacement']}"] += 1
            else:
                per_route["failures"] += 1

            # Mechanism analysis: did the perturbed query route to a different system?
            baseline_route = atk.get("baseline_actual_route", route)
            # Look at the perturbed query's observables (last non-baseline call)
            perturbed_route = None
            for entry in reversed(atk.get("oracle_log", [])):
                if not entry.get("is_baseline"):
                    perturbed_route = predict_route(detector, entry)
                    break
            if perturbed_route is None:
                perturbed_route = baseline_route  # only baseline call existed
            if success:
                if perturbed_route != baseline_route:
                    outcome = f"success_route_flip_{baseline_route}_to_{perturbed_route}"
                    results["questions_with_route_flip"].append({
                        "qid": atk["qid"],
                        "baseline_route": baseline_route,
                        "perturbed_route": perturbed_route,
                        "swap": atk.get("successful_swap"),
                    })
                else:
                    outcome = f"success_same_route_{baseline_route}"
            else:
                outcome = f"failure_route_{baseline_route}"
            per_route["outcome_counts"][outcome] += 1

        # Compute summary stats
        per_route["success_rate"] = per_route["successes"] / per_route["n"] if per_route["n"] else 0.0
        ci_lo, ci_hi = bootstrap_ci(
            [1 if i < per_route["successes"] else 0 for i in range(per_route["n"])]
        )
        per_route["success_rate_ci_95"] = [ci_lo, ci_hi]
        per_route["mean_queries_used"] = float(np.mean(per_route["queries_used"])) if per_route["queries_used"] else 0.0
        per_route["mean_queries_to_success"] = float(np.mean(per_route["queries_to_success"])) if per_route["queries_to_success"] else 0.0

        # Convert Counter to dict for JSON
        per_route["outcome_counts"] = dict(per_route["outcome_counts"])
        results["per_route"][route] = per_route

    # Aggregate breakdown
    all_outcomes = Counter()
    for route in attacks_by_route:
        for outcome, count in results["per_route"][route]["outcome_counts"].items():
            all_outcomes[outcome] += count
    results["mechanism_breakdown"] = dict(all_outcomes)

    # Top successful swaps
    results["top_successful_swaps"] = dict(results["successful_swaps"].most_common(15))

    # Print summary
    print(f"\n[4/4] Summary")
    print("=" * 70)
    for route in ("A", "B", "C"):
        if route not in results["per_route"]:
            continue
        d = results["per_route"][route]
        ci_lo, ci_hi = d["success_rate_ci_95"]
        print(f"\nRoute {route}: {d['successes']}/{d['n']} = {100*d['success_rate']:.1f}% "
              f"(95% CI: {100*ci_lo:.1f}–{100*ci_hi:.1f})")
        print(f"  Mean queries used:        {d['mean_queries_used']:.1f}")
        if d["queries_to_success"]:
            print(f"  Mean queries to success:  {d['mean_queries_to_success']:.1f}")
        print(f"  Outcome breakdown:")
        for outcome, count in sorted(d["outcome_counts"].items()):
            print(f"    {outcome}: {count}")

    print(f"\n{'='*70}")
    print("Route flips (successful attacks that changed the routed system):")
    if results["questions_with_route_flip"]:
        for rec in results["questions_with_route_flip"]:
            swap = rec.get("swap") or {}
            print(f"  {rec['baseline_route']} -> {rec['perturbed_route']}: qid={rec['qid']}")
            print(f"    swap: {swap.get('original')!r} -> {swap.get('replacement')!r}")
    else:
        print("  (none)")

    print(f"\n{'='*70}")
    print("Top successful swaps:")
    for swap, count in results["successful_swaps"].most_common(10):
        print(f"  {count}x  {swap}")

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Convert Counter etc. to JSON-serializable
    results["successful_swaps"] = dict(results["successful_swaps"])
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis to {args.output_json}")


if __name__ == "__main__":
    main()
