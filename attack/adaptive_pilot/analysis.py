"""Analysis for the adaptive-cue pilot.

Reuses the canonical PCR implementation unchanged
(analysis/canonical/canonical_completion.py): every arm's trace becomes a
CanonicalCompletionRecord via build_canonical_record, so PCR eligibility,
terminal routes, and T_i are defined exactly as in the thesis. This module adds
only the pilot-specific pairing (Static vs Control, Dynamic vs Static,
Optimized vs Dynamic), the first-vs-final optimization view, and bootstrap
intervals. It introduces no new PCR definition.

Evidence is measured independently of attacker feedback via
analysis/evidence.py build_evidence_record. Answer accuracy is intentionally
not computed here (the report-to-answer adjudication is out of pilot scope).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from analysis.canonical.canonical_completion import (
    PCR_NUMERATOR_ROUTES,
    build_canonical_record,
)
from analysis.completion_inventory import RowRecord

PCR_CHECKPOINTS = (1, 2, 3, 5, 10)


def _row_record(model: str, dataset: str, condition: str, example_id: str, run_id: str, trace_path: str) -> RowRecord:
    exists = bool(trace_path) and Path(trace_path).exists()
    return RowRecord(
        model=model, dataset=dataset, condition=condition, shard_index=0, example_id=example_id,
        final_status="ok" if exists else "error", error="", trace_path=trace_path or "",
        trace_exists=exists, valid_answer=None, row_index_in_shard=0, run_id=run_id,
    )


@dataclass
class ArmCompletion:
    arm: str
    condition: str
    example_id: str
    pcr_eligible: bool
    terminal_route: str
    t_i: Optional[int]
    completed: bool
    exclusion_reason: str


def load_cells(results_path: Path) -> list[dict]:
    return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def arm_completions(cell: dict) -> dict[str, ArmCompletion]:
    out: dict[str, ArmCompletion] = {}
    for arm, info in cell.get("arms", {}).items():
        trace_path = info.get("trace_path") or ""
        rec = build_canonical_record(
            _row_record(cell["victim_model"], cell["dataset"], info.get("condition", cell["condition"]),
                        cell["example_id"], info.get("run_id", ""), trace_path)
        )
        out[arm] = ArmCompletion(
            arm=arm, condition=info.get("condition", cell["condition"]), example_id=cell["example_id"],
            pcr_eligible=rec.pcr_eligible, terminal_route=rec.terminal_route,
            t_i=rec.first_terminal_post_observation_step,
            completed=rec.terminal_route in PCR_NUMERATOR_ROUTES and rec.pcr_eligible,
            exclusion_reason=rec.pcr_exclusion_reason,
        )
    return out


def _pcr_at_k(completions: Iterable[ArmCompletion], k: int) -> Optional[float]:
    elig = [c for c in completions if c.pcr_eligible]
    if not elig:
        return None
    hit = sum(1 for c in elig if c.completed and c.t_i is not None and c.t_i <= k)
    return 100.0 * hit / len(elig)


def _bootstrap_ci(paired_diffs: list[float], iters: int = 2000, seed: int = 20261005) -> tuple[float, float, float]:
    import numpy as np

    if not paired_diffs:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(paired_diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(cells: list[dict], attack_arm: str, baseline_arm: str, k: int) -> dict:
    """Question-level paired PCR@k delta (attack - baseline), both arms eligible."""
    diffs: list[float] = []
    n_pairs = 0
    for cell in cells:
        comps = arm_completions(cell)
        a, b = comps.get(attack_arm), comps.get(baseline_arm)
        if not a or not b or not (a.pcr_eligible and b.pcr_eligible):
            continue
        n_pairs += 1
        a_hit = 1.0 if (a.completed and a.t_i is not None and a.t_i <= k) else 0.0
        b_hit = 1.0 if (b.completed and b.t_i is not None and b.t_i <= k) else 0.0
        diffs.append(a_hit - b_hit)
    mean, lo, hi = _bootstrap_ci([d * 100 for d in diffs])
    return {
        "attack_arm": attack_arm, "baseline_arm": baseline_arm, "k": k, "paired_n": n_pairs,
        "mean_delta_pcr": mean, "ci95_lo": lo, "ci95_hi": hi,
    }


def optimization_view(cells: list[dict], k: int) -> dict:
    """First-attempt (Dynamic) vs final-attempt (Optimized), including regressions."""
    improved = worsened = unchanged = no_optimized = 0
    paired = 0
    for cell in cells:
        comps = arm_completions(cell)
        dyn, opt = comps.get("dynamic"), comps.get("optimized")
        if not opt or not opt.pcr_eligible or not dyn or not dyn.pcr_eligible:
            no_optimized += 1
            continue
        paired += 1
        d = (1 if dyn.completed and dyn.t_i is not None and dyn.t_i <= k else 0)
        o = (1 if opt.completed and opt.t_i is not None and opt.t_i <= k else 0)
        if o > d:
            improved += 1
        elif o < d:
            worsened += 1
        else:
            unchanged += 1
    return {
        "k": k, "paired_n": paired, "revision_improved": improved,
        "revision_worsened": worsened, "revision_unchanged": unchanged,
        "no_eligible_optimized": no_optimized,
    }


def route_and_failure_counts(cells: list[dict]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        for arm, comp in arm_completions(cell).items():
            d = counts.setdefault(arm, {"eligible": 0, "ineligible": 0})
            d[comp.terminal_route] = d.get(comp.terminal_route, 0) + 1
            d["eligible" if comp.pcr_eligible else "ineligible"] += 1
    return counts


def attacker_cost_summary(cells: list[dict]) -> dict:
    gen_tokens = refine_tokens = gen_calls = refine_calls = 0
    gen_fail = refine_fail = 0
    for cell in cells:
        g = cell.get("attacker_generation") or {}
        gen_calls += 1
        gen_tokens += (g.get("completion_tokens") or 0) + (g.get("prompt_tokens") or 0)
        gen_fail += 0 if g.get("accepted") else 1
        r = cell.get("attacker_refinement")
        if r:
            refine_calls += 1
            refine_tokens += (r.get("completion_tokens") or 0) + (r.get("prompt_tokens") or 0)
            refine_fail += 0 if r.get("accepted") else 1
    return {
        "generation_calls": gen_calls, "generation_tokens": gen_tokens, "generation_validation_failures": gen_fail,
        "refinement_calls": refine_calls, "refinement_tokens": refine_tokens, "refinement_validation_failures": refine_fail,
    }


def summarize(cells: list[dict]) -> dict:
    by_cond_model: dict[tuple, list[dict]] = {}
    for cell in cells:
        by_cond_model.setdefault((cell["victim_model"], cell["dataset"], cell["condition"]), []).append(cell)

    comparisons = []
    for (model, dataset, condition), group in sorted(by_cond_model.items()):
        for attack_arm, base_arm in (("static", "control"), ("dynamic", "static"), ("optimized", "dynamic")):
            for k in PCR_CHECKPOINTS:
                d = paired_delta(group, attack_arm, base_arm, k)
                d.update(model=model, dataset=dataset, condition=condition)
                comparisons.append(d)

    return {
        "schema_version": "adaptive_cue_pilot_analysis_v1",
        "n_cells": len(cells),
        "paired_comparisons": comparisons,
        "optimization_view": {k: optimization_view(cells, k) for k in PCR_CHECKPOINTS},
        "route_and_failure_counts": route_and_failure_counts(cells),
        "attacker_cost": attacker_cost_summary(cells),
        "shared_prefix_ok_fraction": (
            sum(1 for c in cells if c.get("shared_prefix_ok")) / len(cells) if cells else None
        ),
    }
