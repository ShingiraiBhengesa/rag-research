"""
Combined analysis: Phase 1 + Phase 2 + Phase 3 attack results.

Produces paper-ready statistics including:
- Per-route success rates with 95% bootstrap CIs (n=107 across all phases)
- Mechanism breakdown (route flip vs same-route LLM degradation)
- Phase comparison (P1, P2, P3)
- Filter rejection analysis (Phase 3 finding: filter-bound, not budget-bound)
- Top successful swaps across all phases
- Markdown summary printed to stdout, full JSON saved
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
    """Same feature extraction as route_detector.py training."""
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
    feats = np.asarray(extract_features(record)).reshape(1, -1)
    return str(detector.predict(feats)[0])


def bootstrap_ci(values: List[int], n_iter: int = 10000, alpha: float = 0.05):
    """Bootstrap 95% CI for proportion."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = []
    rng = np.random.default_rng(seed=42)  # reproducible
    for _ in range(n_iter):
        sample_idx = rng.integers(0, n, size=n)
        means.append(float(arr[sample_idx].mean()))
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def is_strict_success(atk: Dict[str, Any]) -> bool:
    """Strict success: answer changed AND not in gold spans."""
    return bool(atk.get("attack_success_strict") or atk.get("attack_success"))


def load_phase(paths: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                out.append(json.loads(line))
    return out


def analyze_attacks(
    attacks: List[Dict[str, Any]],
    detector,
    label: str,
) -> Dict[str, Any]:
    """Compute per-route stats for a set of attacks."""
    by_route = defaultdict(list)
    for atk in attacks:
        r = atk.get("baseline_actual_route") or "?"
        by_route[r].append(atk)

    stats = {"label": label, "total": len(attacks), "by_route": {}}
    overall_outcomes = [1 if is_strict_success(a) else 0 for a in attacks]
    overall_success = sum(overall_outcomes)
    ci_lo, ci_hi = bootstrap_ci(overall_outcomes)
    stats["overall"] = {
        "successes": overall_success,
        "n": len(attacks),
        "rate": overall_success / max(len(attacks), 1),
        "ci_95": [ci_lo, ci_hi],
    }

    for route in ("A", "B", "C"):
        route_atks = by_route.get(route, [])
        if not route_atks:
            continue
        outcomes = [1 if is_strict_success(a) else 0 for a in route_atks]
        succ = sum(outcomes)
        ci = bootstrap_ci(outcomes)

        # Mechanism analysis (route flip vs same-route)
        outcome_counts = Counter()
        successful_swaps = Counter()
        queries_to_success = []
        candidates_total = []
        candidates_filtered_sim = []
        candidates_filtered_ppl = []

        for atk in route_atks:
            success = is_strict_success(atk)
            baseline_route = atk.get("baseline_actual_route", route)
            # Look at last non-baseline oracle call to infer perturbed route
            perturbed_route = None
            for entry in reversed(atk.get("oracle_log") or []):
                if not entry.get("is_baseline"):
                    perturbed_route = predict_route(detector, entry)
                    break
            if perturbed_route is None:
                perturbed_route = baseline_route

            if success:
                if perturbed_route != baseline_route:
                    outcome = f"success_flip_{baseline_route}_to_{perturbed_route}"
                else:
                    outcome = f"success_same_route_{baseline_route}"
                queries_to_success.append(int(atk.get("queries_used") or 0))
                swap = atk.get("successful_swap") or {}
                if swap.get("original") and swap.get("replacement"):
                    successful_swaps[f"{swap['original']} -> {swap['replacement']}"] += 1
            else:
                outcome = f"failure_{baseline_route}"

            outcome_counts[outcome] += 1
            candidates_total.append(int(atk.get("num_candidates_generated") or 0))
            candidates_filtered_sim.append(int(atk.get("num_candidates_filtered_sim") or 0))
            candidates_filtered_ppl.append(int(atk.get("num_candidates_filtered_ppl") or 0))

        stats["by_route"][route] = {
            "n": len(route_atks),
            "successes": succ,
            "rate": succ / len(route_atks),
            "ci_95": list(ci),
            "outcome_counts": dict(outcome_counts),
            "successful_swaps": dict(successful_swaps.most_common(20)),
            "mean_queries_to_success": float(np.mean(queries_to_success)) if queries_to_success else 0.0,
            "mean_candidates_generated": float(np.mean(candidates_total)) if candidates_total else 0.0,
            "mean_filtered_sim": float(np.mean(candidates_filtered_sim)) if candidates_filtered_sim else 0.0,
            "mean_filtered_ppl": float(np.mean(candidates_filtered_ppl)) if candidates_filtered_ppl else 0.0,
        }
    return stats


def print_markdown_table(stats: Dict[str, Any], header: str):
    print(f"\n### {header}\n")
    print("| Route | n | Successes | Rate | 95% CI |")
    print("|-------|---|-----------|------|--------|")
    for route in ("A", "B", "C"):
        if route not in stats["by_route"]:
            continue
        d = stats["by_route"][route]
        ci_lo, ci_hi = d["ci_95"]
        print(f"| {route} | {d['n']} | {d['successes']} | "
              f"{100*d['rate']:.1f}% | [{100*ci_lo:.1f}, {100*ci_hi:.1f}] |")
    o = stats["overall"]
    ci_lo, ci_hi = o["ci_95"]
    print(f"| **Total** | **{o['n']}** | **{o['successes']}** | "
          f"**{100*o['rate']:.1f}%** | **[{100*ci_lo:.1f}, {100*ci_hi:.1f}]** |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--detector",
        type=Path,
        default=Path("results/route_detector.joblib"),
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/combined_analysis.json"),
    )
    args = ap.parse_args()

    # Load route detector
    print(f"Loading route detector from {args.detector}...")
    bundle = joblib.load(args.detector)
    detector = bundle["model"]
    print(f"  detector classes: {bundle['classes']}\n")

    # Load all phases
    phase1_attacks = load_phase([
        args.results_dir / "phase1" / "phase1_A.jsonl",
        args.results_dir / "phase1" / "phase1_B.jsonl",
        args.results_dir / "phase1" / "phase1_C.jsonl",
    ])
    phase2_attacks = load_phase([
        args.results_dir / "phase2" / "phase2_C.jsonl",
    ])
    phase3_attacks = load_phase([
        args.results_dir / "phase3" / "phase3.jsonl",
    ])

    print(f"Loaded: Phase 1 = {len(phase1_attacks)}, "
          f"Phase 2 = {len(phase2_attacks)}, "
          f"Phase 3 = {len(phase3_attacks)}")
    print(f"Combined Phase 1+2 = {len(phase1_attacks) + len(phase2_attacks)} attacks\n")

    # Analyze each phase + combined
    p1_stats = analyze_attacks(phase1_attacks, detector, "Phase 1 (n=53)")
    p2_stats = analyze_attacks(phase2_attacks, detector, "Phase 2 (n=54, Route C continuation)")
    p3_stats = analyze_attacks(phase3_attacks, detector, "Phase 3 (n=39, extended budget)")
    combined = analyze_attacks(phase1_attacks + phase2_attacks, detector, "Combined Phase 1+2")
    all_phases = analyze_attacks(
        phase1_attacks + phase2_attacks + phase3_attacks, detector, "All phases"
    )

    # =====================================================
    # MARKDOWN OUTPUT FOR LAB MEETING / PAPER
    # =====================================================
    print("=" * 70)
    print("BLACK-BOX ATTACK PIPELINE: COMBINED ANALYSIS")
    print("=" * 70)

    print_markdown_table(combined, "Combined Phase 1+2 (Primary Result)")
    print_markdown_table(p1_stats, "Phase 1 (n=53, baseline-correct, budget=30)")
    print_markdown_table(p2_stats, "Phase 2 (n=54, Route C continuation, budget=30)")
    print_markdown_table(p3_stats, "Phase 3 (n=39, hardest failures, budget=200)")

    # =====================================================
    # MECHANISM BREAKDOWN
    # =====================================================
    print("\n### Mechanism Breakdown (Combined Phase 1+2)\n")
    all_outcomes = Counter()
    for route in ("A", "B", "C"):
        for outcome, count in combined["by_route"].get(route, {}).get("outcome_counts", {}).items():
            all_outcomes[outcome] += count

    total_success = sum(v for k, v in all_outcomes.items() if k.startswith("success"))
    print(f"Total successful attacks: {total_success}")
    print()

    same_route = sum(v for k, v in all_outcomes.items() if "same_route" in k)
    flips = sum(v for k, v in all_outcomes.items() if "flip" in k)

    print(f"| Mechanism | Count | % of Successes |")
    print(f"|-----------|-------|----------------|")
    print(f"| Same-route LLM degradation | {same_route} | {100*same_route/max(total_success,1):.1f}% |")
    print(f"| Route flip (escalation/de-escalation) | {flips} | {100*flips/max(total_success,1):.1f}% |")

    # Detail per route flip type
    print("\nDetailed route flip breakdown:")
    for k, v in sorted(all_outcomes.items()):
        if "flip" in k:
            print(f"  {k}: {v}")

    # =====================================================
    # FILTER ANALYSIS (Phase 3 finding)
    # =====================================================
    print("\n### Filter Analysis (Why Phase 3 Failed)\n")
    print("Phase 3 budget=200 vs Phase 1+2 budget=30:")
    p3 = p3_stats["by_route"]
    p12 = combined["by_route"]

    print("\n| Phase | Route | Mean candidates generated | Mean filtered (sim) | Mean filtered (PPL) | Mean passing |")
    print("|-------|-------|---------------------------|---------------------|---------------------|--------------|")
    for route in ("B", "C"):
        for label, stats in [("P1+P2", p12), ("P3 (hard)", p3)]:
            d = stats.get(route, {})
            if not d:
                continue
            cg = d.get("mean_candidates_generated", 0)
            cs = d.get("mean_filtered_sim", 0)
            cp = d.get("mean_filtered_ppl", 0)
            passing = max(cg - cs - cp, 0)
            print(f"| {label} | {route} | {cg:.1f} | {cs:.1f} | {cp:.1f} | {passing:.1f} |")

    # =====================================================
    # TOP SUCCESSFUL SWAPS
    # =====================================================
    print("\n### Top Successful Swaps (Combined Phase 1+2)\n")
    all_swaps = Counter()
    for route in ("A", "B", "C"):
        for swap, count in combined["by_route"].get(route, {}).get("successful_swaps", {}).items():
            all_swaps[swap] += count
    print("| Count | Swap |")
    print("|-------|------|")
    for swap, count in all_swaps.most_common(15):
        print(f"| {count}x | {swap} |")

    # =====================================================
    # SAVE JSON
    # =====================================================
    output = {
        "phase1": p1_stats,
        "phase2": p2_stats,
        "phase3": p3_stats,
        "combined_p1_p2": combined,
        "all_phases": all_phases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nFull analysis saved to {args.output_json}")


if __name__ == "__main__":
    main()
