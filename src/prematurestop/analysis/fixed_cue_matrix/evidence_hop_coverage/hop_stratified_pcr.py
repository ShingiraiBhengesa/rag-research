"""PCR curves and paired deltas, cut by hop/evidence-count stratum.

`results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/pcr_stepwise.csv` and
`pcr_paired_deltas_stepwise.csv` are grouped by (model, dataset, condition)
only -- never further split by hop count within a dataset (2-hop vs 4-hop
2WikiMultiHopQA, 2/3/4-hop MuSiQue, 2/3/4 required-evidence-count
MultiHopRAG). This module adds that missing stratification, reusing:

- `prematurestop.evaluation.completion.compute_route_specific_curves` /
  `.build_strict_pairs` -- unchanged, called on hop-stratum-filtered
  record subsets rather than whole-dataset ones. No new PCR/pairing logic.
- `prematurestop.analysis.fixed_cue_matrix.results_bundle.paired_deltas._t_i_array` /
  `._bootstrap_all_steps` -- the exact same vectorized paired-bootstrap
  method already used for the unstratified paired deltas, unchanged.
- `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS` -- the same
  reconciled comparison manifest, no new list.

Does NOT re-parse any trace. `CanonicalCompletionRecord` objects are
reconstructed directly from the already-computed, already gate-validated
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/pcr_run_level.csv`; hop-stratum values are
read from the already-computed, already gate-validated
`results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/run_level_evidence_hops.csv`. Both are pure
CSV reads (seconds, not the ~10-15 minute trace-parsing cost each of those
two files already paid once).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from prematurestop.evaluation import config
from prematurestop.evaluation import completion as cc
from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import COMPARISONS
from prematurestop.analysis.fixed_cue_matrix.results_bundle.paired_deltas import STEPWISE_BOOTSTRAP_ITERATIONS, _bootstrap_all_steps, _t_i_array
from prematurestop.paths import RESULTS_E1_QWEN_LLAMA

END_OF_BUDGET_STEP = 30


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_bool(v: str) -> bool:
    return v == "True"


def _to_int_or_none(v: str):
    return None if v in ("", "None") else int(v)


def load_canonical_records_from_csv(pcr_run_level_path: Path) -> list[cc.CanonicalCompletionRecord]:
    """Reconstructs `CanonicalCompletionRecord` objects from
    `record_to_dict`'s own CSV output -- the exact inverse of that function,
    so the reconstructed objects are field-for-field identical to the ones
    `build_canonical_records` produced (verified: every field on
    `CanonicalCompletionRecord` is a plain str/bool/int/None, so this round
    trip is lossless)."""
    records = []
    for r in _read_csv(pcr_run_level_path):
        records.append(
            cc.CanonicalCompletionRecord(
                model=r["model"], dataset=r["dataset"], condition=r["condition"], example_id=r["example_id"],
                run_id=r["run_id"], trace_path="", researcher_episode_id=r["researcher_episode_id"] or None,
                anchor_found=_to_bool(r["anchor_found"]), anchor_event_index=_to_int_or_none(r["anchor_event_index"]),
                anchor_step=_to_int_or_none(r["anchor_step"]),
                injection_applied=None if r["injection_applied"] in ("", "None") else _to_bool(r["injection_applied"]),
                first_terminal_post_observation_step=_to_int_or_none(r["first_terminal_post_observation_step"]),
                terminal_route=r["terminal_route"],
                research_complete_step=_to_int_or_none(r["research_complete_step"]),
                no_tool_direct_exit_step=_to_int_or_none(r["no_tool_direct_exit_step"]),
                post_observation_researcher_decisions=int(r["post_observation_researcher_decisions"]),
                post_observation_search_calls=int(r["post_observation_search_calls"]),
                post_observation_think_calls=int(r["post_observation_think_calls"]),
                supervisor_redelegated=_to_bool(r["supervisor_redelegated"]),
                final_status=r["final_status"], parse_valid=_to_bool(r["parse_valid"]),
                pcr_eligible=_to_bool(r["pcr_eligible"]), pcr_exclusion_reason=r["pcr_exclusion_reason"],
            )
        )
    return records


def load_hop_stratum_lookup(run_level_evidence_hops_path: Path) -> dict[tuple[str, str, str, str], int]:
    """(model, dataset, condition, example_id) -> dataset_stratum_value.
    Read once from `run_level_evidence_hops.csv` -- never recomputed."""
    out: dict[tuple[str, str, str, str], int] = {}
    for r in _read_csv(run_level_evidence_hops_path):
        v = r["dataset_stratum_value"]
        if v not in ("", "None"):
            out[(r["model"], r["dataset"], r["condition"], r["example_id"])] = int(v)
    return out


def group_by_stratum(
    records: list[cc.CanonicalCompletionRecord], stratum_lookup: dict[tuple[str, str, str, str], int]
) -> tuple[dict[tuple[str, str, int], list[cc.CanonicalCompletionRecord]], int]:
    grouped: dict[tuple[str, str, int], list[cc.CanonicalCompletionRecord]] = {}
    n_no_stratum = 0
    for r in records:
        stratum = stratum_lookup.get((r.model, r.dataset, r.condition, r.example_id))
        if stratum is None:
            n_no_stratum += 1
            continue
        grouped.setdefault((r.model, r.dataset, stratum), []).append(r)
    return grouped, n_no_stratum


def write_hop_stratified_pcr_stepwise_csv(
    records: list[cc.CanonicalCompletionRecord], stratum_lookup: dict, path: Path
) -> int:
    grouped, n_no_stratum = group_by_stratum(records, stratum_lookup)
    rows: list[dict] = []
    by_model_dataset_stratum_condition: dict[tuple, list] = {}
    for (model, dataset, stratum), recs in grouped.items():
        by_condition: dict[str, list] = {}
        for r in recs:
            by_condition.setdefault(r.condition, []).append(r)
        for condition, condition_recs in by_condition.items():
            by_model_dataset_stratum_condition[(model, dataset, stratum, condition)] = condition_recs

    for (model, dataset, stratum, condition), recs in sorted(by_model_dataset_stratum_condition.items()):
        curve = cc.compute_route_specific_curves(recs, k_max=END_OF_BUDGET_STEP)
        n_eligible = curve["eligible_n"]
        for step_row in curve["per_step"]:
            k = step_row["k"]
            rc_pct, nt_pct, union_pct = step_row["rc_pct"], step_row["no_tool_pct"], step_row["union_pct"]
            rows.append(
                {
                    "model": model, "dataset": dataset, "hop_stratum": stratum, "condition": condition,
                    "step": k, "n_total": len(recs), "n_eligible": n_eligible,
                    "n_completed_by_step": step_row["rc_cum_n"] + step_row["no_tool_cum_n"],
                    "pcr": (union_pct / 100.0) if union_pct is not None else None,
                    "n_research_complete_by_step": step_row["rc_cum_n"],
                    "research_complete_pcr": (rc_pct / 100.0) if rc_pct is not None else None,
                    "n_no_tool_exit_by_step": step_row["no_tool_cum_n"],
                    "no_tool_exit_pcr": (nt_pct / 100.0) if nt_pct is not None else None,
                }
            )
    _write_csv(path, rows)
    return n_no_stratum


def write_hop_stratified_pcr_paired_deltas_csv(records: list[cc.CanonicalCompletionRecord], stratum_lookup: dict, path: Path) -> None:
    grouped, _ = group_by_stratum(records, stratum_lookup)
    steps = np.arange(1, END_OF_BUDGET_STEP + 1)
    rows: list[dict] = []
    for (model, dataset, stratum), recs in sorted(grouped.items()):
        for comp in COMPARISONS:
            pairs = cc.build_strict_pairs(recs, comp.cue_condition, comp.control_condition)
            n_paired = len(pairs)
            cond_all = [r for r in recs if r.condition == comp.cue_condition]
            base_all = [r for r in recs if r.condition == comp.control_condition]
            if not cond_all or not base_all:
                continue
            n_cue_eligible = sum(1 for r in cond_all if r.pcr_eligible)
            n_control_eligible = sum(1 for r in base_all if r.pcr_eligible)

            if n_paired == 0:
                for step in steps:
                    rows.append(
                        {
                            "model": model, "dataset": dataset, "hop_stratum": stratum,
                            "comparison_name": comp.comparison_name,
                            "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                            "step": int(step),
                            "n_control_eligible": n_control_eligible, "n_cue_eligible": n_cue_eligible,
                            "n_paired": 0,
                            "control_pcr": None, "cue_pcr": None, "delta_pcr": None,
                            "ci_lower": None, "ci_upper": None,
                        }
                    )
                continue

            cue_t_i = _t_i_array([a for a, _b in pairs])
            control_t_i = _t_i_array([b for _a, b in pairs])
            control_pcr, cue_pcr, delta_pcr, ci_lo, ci_hi = _bootstrap_all_steps(
                cue_t_i, control_t_i, steps, STEPWISE_BOOTSTRAP_ITERATIONS, config.DETERMINISTIC_SEED
            )
            for i, step in enumerate(steps):
                rows.append(
                    {
                        "model": model, "dataset": dataset, "hop_stratum": stratum,
                        "comparison_name": comp.comparison_name,
                        "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                        "step": int(step),
                        "n_control_eligible": n_control_eligible, "n_cue_eligible": n_cue_eligible,
                        "n_paired": n_paired,
                        "control_pcr": float(control_pcr[i]), "cue_pcr": float(cue_pcr[i]), "delta_pcr": float(delta_pcr[i]),
                        "ci_lower": float(ci_lo[i]), "ci_upper": float(ci_hi[i]),
                    }
                )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    output_dir = RESULTS_E1 / "evidence_hop_coverage"
    pcr_run_level_path = RESULTS_E1_QWEN_LLAMA / "data" / "pcr_run_level.csv"
    run_level_evidence_hops_path = output_dir / "run_level_evidence_hops.csv"

    print("Loading canonical records from pcr_run_level.csv (no trace re-parsing)...")
    records = load_canonical_records_from_csv(pcr_run_level_path)
    print(f"  {len(records)} records")

    print("Loading hop-stratum lookup from run_level_evidence_hops.csv...")
    stratum_lookup = load_hop_stratum_lookup(run_level_evidence_hops_path)
    print(f"  {len(stratum_lookup)} rows with a defined stratum")

    print("Writing hop_stratified_pcr_stepwise.csv...")
    n_no_stratum = write_hop_stratified_pcr_stepwise_csv(records, stratum_lookup, output_dir / "hop_stratified_pcr_stepwise.csv")
    print(f"  {n_no_stratum} records had no stratum value (excluded, not imputed)")

    print("Writing hop_stratified_pcr_paired_deltas.csv (bootstrap CIs)...")
    write_hop_stratified_pcr_paired_deltas_csv(records, stratum_lookup, output_dir / "hop_stratified_pcr_paired_deltas.csv")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
