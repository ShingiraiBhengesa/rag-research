"""Steps 3-4: canonical run-level completion + full step-1-29 PCR/route curves.

Reuses `canonical_completion.build_canonical_records` (step 3) and
`compute_pcr_curve`/`compute_route_specific_curves` (step 4) completely
unchanged -- both are already parametrized by an arbitrary `k_max`, they are
just never called with `k_max=29` anywhere in the existing module (every
existing CSV writer only calls them at the 8 checkpoints
`(1,2,3,5,10,15,20,30)`). This module adds the missing full-sweep call site
and writers; it adds no new statistical logic.
"""

from __future__ import annotations

from pathlib import Path

from prematurestop.evaluation import completion as cc

FULL_STEP_MAX = 29
END_OF_BUDGET_STEP = 30  # PCR@30 retained as the end-of-budget summary (spec 3.1)


def write_pcr_run_level_csv(records: list[cc.CanonicalCompletionRecord], path: Path) -> None:
    """`data/pcr_run_level.csv` -- one row per run, verbatim from
    `canonical_completion.record_to_dict`, just retargeted to the bundle path.
    """
    cc.write_run_level_csv(records, path)


def _cell_key(r: cc.CanonicalCompletionRecord) -> tuple[str, str, str]:
    return (r.model, r.dataset, r.condition)


def write_pcr_stepwise_full_csv(records: list[cc.CanonicalCompletionRecord], path: Path) -> None:
    """`data/pcr_stepwise.csv` -- spec 5.2. One row per
    (model, dataset, condition, step) for step in 1..29, plus a synthesized
    step-30 row (the "end-of-budget summary" PCR@30 spec 3.1 asks to retain).
    `pcr = research_complete_pcr + no_tool_exit_pcr` holds by construction
    (both come from the same `compute_route_specific_curves` call, which
    already asserts internally that the two sum to the union curve).
    """
    grouped = cc._group(records, _cell_key)  # noqa: SLF001 -- reusing the module's own grouping helper
    rows: list[dict] = []
    for (model, dataset, condition), recs in sorted(grouped.items()):
        curve = cc.compute_route_specific_curves(recs, k_max=END_OF_BUDGET_STEP)
        n_eligible = curve["eligible_n"]
        for step_row in curve["per_step"]:
            k = step_row["k"]
            rc_pct = step_row["rc_pct"]
            nt_pct = step_row["no_tool_pct"]
            union_pct = step_row["union_pct"]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "condition": condition,
                    "step": k,
                    "n_total": len(recs),
                    "n_eligible": n_eligible,
                    "n_completed_by_step": step_row["rc_cum_n"] + step_row["no_tool_cum_n"],
                    "pcr": (union_pct / 100.0) if union_pct is not None else None,
                    "n_research_complete_by_step": step_row["rc_cum_n"],
                    "research_complete_pcr": (rc_pct / 100.0) if rc_pct is not None else None,
                    "n_no_tool_exit_by_step": step_row["no_tool_cum_n"],
                    "no_tool_exit_pcr": (nt_pct / 100.0) if nt_pct is not None else None,
                }
            )
    cc._write_csv(path, rows)  # noqa: SLF001 -- reusing the module's own CSV writer


def write_route_decomposition_stepwise_csv(records: list[cc.CanonicalCompletionRecord], path: Path) -> None:
    """`data/route_decomposition_stepwise.csv` -- spec 5.4. Deliberately does
    NOT repeat `pcr`/`n_total`/`n_eligible` (already in `pcr_stepwise.csv`);
    joins to it on (model,dataset,condition,step). Carries only the
    route-specific per-step counts plus final tallies for all 7 canonical
    routes (see plan decision log: this repo's route taxonomy already has
    exactly 7 entries, matching spec 5.4's "seven terminal routes").
    """
    grouped = cc._group(records, _cell_key)  # noqa: SLF001
    rows: list[dict] = []
    for (model, dataset, condition), recs in sorted(grouped.items()):
        curve = cc.compute_route_specific_curves(recs, k_max=END_OF_BUDGET_STEP)
        decomp = cc.route_decomposition(recs)
        for step_row in curve["per_step"]:
            row = {
                "model": model,
                "dataset": dataset,
                "condition": condition,
                "step": step_row["k"],
                "n_research_complete_by_step": step_row["rc_cum_n"],
                "n_no_tool_exit_by_step": step_row["no_tool_cum_n"],
            }
            # Final (non-step-indexed) route tallies, repeated on every row of
            # this cell for easy joining/filtering -- see TABLE_INDEX.md for
            # the documented no-duplication convention with pcr_stepwise.csv.
            for route in cc.CANONICAL_ROUTES:
                row[f"final_n_{route}"] = decomp[f"n_{route}"]
            rows.append(row)
    cc._write_csv(path, rows)  # noqa: SLF001
