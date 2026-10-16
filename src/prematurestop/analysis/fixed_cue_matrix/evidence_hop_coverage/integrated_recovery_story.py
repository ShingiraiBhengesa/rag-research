"""Integrates the three pieces the checklist flagged as "not ready" /
scattered: supervisor recovery (previously split across two disconnected
fields in two packages), the two evidence boundaries (previously two
separate files with no side-by-side view), and hop-stratified PCR
(previously nonexistent, built in `hop_stratified_pcr.py`) -- into one
consolidated summary CSV pair plus one narrative report.

Reused, unmodified:
  - `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/process_metrics.csv::later_evidence_recovery`
    (the categorical recovery label, `src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation/recovery.py`'s
    RECOVERY_* constants, already computed by
    `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/process_metrics.py::classify_recovery`)
  - `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv` (final-report-input
    boundary, unchanged)
  - `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/paired_evidence_deltas.csv`
    (local-completion boundary, unchanged)
  - `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/hop_stratified_pcr_stepwise.csv` /
    `hop_stratified_pcr_paired_deltas.csv` (this session's own prior step)

New here: only the aggregation/merge logic (grouping recovery labels into
counts, joining the two evidence-boundary CSVs on their shared key,
appending 4 new validation gates) -- no new PCR, evidence, or recovery
computation.
"""

from __future__ import annotations

import csv
from pathlib import Path

from prematurestop.analysis.fixed_cue_matrix.paper_evaluation.recovery import (
    RECOVERY_NO_REDELEGATION,
    RECOVERY_REDELEGATED_FULL,
    RECOVERY_REDELEGATED_NO_RESTORE,
    RECOVERY_REDELEGATED_PARTIAL,
    RECOVERY_UNRESOLVED,
)
from prematurestop.paths import RESULTS_E1_QWEN_LLAMA

QLB_DATA = RESULTS_E1_QWEN_LLAMA / "data"
EHC_DIR = RESULTS_E1 / "evidence_hop_coverage"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


# ---------------------------------------------------------------------------
# supervisor_recovery_summary.csv
# ---------------------------------------------------------------------------

_CATEGORY_COLS = {
    RECOVERY_NO_REDELEGATION: "n_no_redelegation",
    RECOVERY_REDELEGATED_FULL: "n_redelegated_fully_restored",
    RECOVERY_REDELEGATED_PARTIAL: "n_redelegated_partially_restored",
    RECOVERY_REDELEGATED_NO_RESTORE: "n_redelegated_did_not_restore",
    RECOVERY_UNRESOLVED: "n_redelegated_unresolved",
}


def build_supervisor_recovery_summary() -> list[dict]:
    rows = _read_csv(QLB_DATA / "process_metrics.csv")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model"], r["dataset"], r["condition"]), []).append(r)

    out = []
    for (model, dataset, condition), recs in sorted(grouped.items()):
        n_total = len(recs)
        counts = {col: 0 for col in _CATEGORY_COLS.values()}
        for r in recs:
            col = _CATEGORY_COLS.get(r["later_evidence_recovery"])
            if col:
                counts[col] += 1
            else:
                counts.setdefault("n_unrecognized_label", 0)
                counts["n_unrecognized_label"] += 1
        n_redelegated = n_total - counts["n_no_redelegation"]
        n_restored = counts["n_redelegated_fully_restored"] + counts["n_redelegated_partially_restored"]
        out.append(
            {
                "model": model, "dataset": dataset, "condition": condition,
                "n_total": n_total,
                "n_redelegated": n_redelegated,
                **counts,
                "pct_redelegated_fully_restored": (100.0 * counts["n_redelegated_fully_restored"] / n_redelegated) if n_redelegated else None,
                "pct_redelegated_partially_restored": (100.0 * counts["n_redelegated_partially_restored"] / n_redelegated) if n_redelegated else None,
                "pct_redelegated_did_not_restore": (100.0 * counts["n_redelegated_did_not_restore"] / n_redelegated) if n_redelegated else None,
                "pct_redelegated_any_restoration": (100.0 * n_restored / n_redelegated) if n_redelegated else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# evidence_boundary_paired_results.csv
# ---------------------------------------------------------------------------


def build_evidence_boundary_paired_results() -> list[dict]:
    final_input_rows = {
        (r["model"], r["dataset"], r["comparison_name"]): r for r in _read_csv(QLB_DATA / "evidence_summary.csv")
    }
    local_rows = {
        (r["model"], r["dataset"], r["comparison_name"]): r for r in _read_csv(EHC_DIR / "paired_evidence_deltas.csv")
    }
    all_keys = sorted(set(final_input_rows) | set(local_rows))

    out = []
    for model, dataset, comparison_name in all_keys:
        fi = final_input_rows.get((model, dataset, comparison_name))
        loc = local_rows.get((model, dataset, comparison_name))
        out.append(
            {
                "model": model, "dataset": dataset, "comparison_name": comparison_name,
                "control_condition": (fi or loc or {}).get("control_condition"),
                "cue_condition": (fi or loc or {}).get("cue_condition"),
                "final_report_input_n_paired": fi["n_paired"] if fi else None,
                "final_report_input_control_mean_coverage": fi["control_mean_coverage"] if fi else None,
                "final_report_input_cue_mean_coverage": fi["cue_mean_coverage"] if fi else None,
                "final_report_input_paired_evidence_loss": fi["paired_evidence_loss"] if fi else None,
                "final_report_input_ci_lower": fi["ci_lower"] if fi else None,
                "final_report_input_ci_upper": fi["ci_upper"] if fi else None,
                "local_completion_n_paired": loc["n_paired"] if loc else None,
                "local_completion_control_mean_coverage": loc["control_mean_coverage"] if loc else None,
                "local_completion_cue_mean_coverage": loc["cue_mean_coverage"] if loc else None,
                "local_completion_paired_evidence_loss": loc["paired_evidence_loss"] if loc else None,
                "local_completion_full_evidence_rate_loss": loc["full_evidence_rate_loss"] if loc else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Validation gates (appended to the existing evidence_hop_coverage gate file)
# ---------------------------------------------------------------------------


class GateResult:
    def __init__(self, gate_id, description, passed, detail):
        self.gate_id = gate_id
        self.description = description
        self.passed = passed
        self.detail = detail


def gate_15_hop_pcr_monotonic() -> GateResult:
    rows = _read_csv(EHC_DIR / "hop_stratified_pcr_stepwise.csv")
    grouped: dict[tuple, list[tuple[int, float]]] = {}
    for r in rows:
        pcr = r["pcr"]
        if pcr in ("", "None"):
            continue
        grouped.setdefault((r["model"], r["dataset"], r["hop_stratum"], r["condition"]), []).append((int(r["step"]), float(pcr)))
    violations = []
    for key, pts in grouped.items():
        pts.sort()
        for (k1, p1), (k2, p2) in zip(pts, pts[1:]):
            if p2 < p1 - 1e-9:
                violations.append((key, k1, k2))
    return GateResult(15, "Every hop-stratified PCR curve is monotonic nondecreasing", not violations, f"{len(violations)} violations" + (f": {violations[:3]}" if violations else ""))


def gate_16_hop_pairs_subset_of_unstratified() -> GateResult:
    unstratified = {
        (r["model"], r["dataset"], r["comparison_name"], r["step"]): int(r["n_paired"])
        for r in _read_csv(QLB_DATA / "pcr_paired_deltas_stepwise.csv")
    }
    violations = []
    for r in _read_csv(EHC_DIR / "hop_stratified_pcr_paired_deltas.csv"):
        key = (r["model"], r["dataset"], r["comparison_name"], r["step"])
        unstrat_n = unstratified.get(key)
        if unstrat_n is not None and int(r["n_paired"]) > unstrat_n:
            violations.append(key)
    return GateResult(16, "Hop-stratified paired N never exceeds the unstratified paired N for the same cell", not violations, f"{len(violations)} violations" + (f": {violations[:3]}" if violations else ""))


def gate_17_recovery_counts_sum(rows: list[dict]) -> GateResult:
    violations = []
    for r in rows:
        category_sum = sum(r[col] for col in _CATEGORY_COLS.values()) + r.get("n_unrecognized_label", 0)
        if category_sum != r["n_total"]:
            violations.append((r["model"], r["dataset"], r["condition"]))
        if r["n_redelegated"] != r["n_total"] - r["n_no_redelegation"]:
            violations.append((r["model"], r["dataset"], r["condition"], "n_redelegated_mismatch"))
    return GateResult(17, "Recovery category counts sum to n_total, n_redelegated = n_total - n_no_redelegation", not violations, f"{len(violations)} violations")


def gate_18_evidence_boundary_merge_complete(rows: list[dict]) -> GateResult:
    missing_either = [r for r in rows if r["final_report_input_n_paired"] is None or r["local_completion_n_paired"] is None]
    return GateResult(
        18, "Every (model,dataset,comparison) cell has both evidence boundaries present",
        not missing_either, f"{len(missing_either)}/{len(rows)} cells missing one boundary" + (f": {[(r['model'],r['dataset'],r['comparison_name']) for r in missing_either[:3]]}" if missing_either else ""),
    )


def append_gates_csv(new_results: list[GateResult], path: Path) -> None:
    existing = _read_csv(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gate_id", "description", "status", "detail"])
        for r in existing:
            writer.writerow([r["gate_id"], r["description"], r["status"], r["detail"]])
        for g in new_results:
            writer.writerow([g.gate_id, g.description, "PASS" if g.passed else "FAIL", g.detail])


# ---------------------------------------------------------------------------
# EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md
# ---------------------------------------------------------------------------


def write_report(recovery_rows: list[dict], boundary_rows: list[dict], new_gates: list[GateResult]) -> None:
    n_pass = sum(1 for g in new_gates if g.passed)
    lines = [
        "# EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md",
        "",
        "Integrates three pieces that were previously either scattered across "
        "two packages or missing entirely: supervisor recovery (one summary "
        "instead of two disconnected fields), the two evidence boundaries "
        "(side by side instead of two separate files), and hop-stratified "
        "PCR (new -- PCR had never been cut by hop/evidence-count stratum "
        "before this). Locked PCR, evidence matching rules, and comparison "
        "manifest are unchanged from `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/README.md` "
        "and `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` -- this report only "
        "combines already-validated numbers, it does not recompute them.",
        "",
        "## 1. Hop-stratified PCR",
        "",
        "See `hop_stratified_pcr_stepwise.csv` / `hop_stratified_pcr_paired_deltas.csv` "
        "-- PCR curves and paired deltas, grouped by `(model, dataset, hop_stratum, "
        "condition)` instead of just `(model, dataset, condition)`. Reuses "
        "`canonical_completion.compute_route_specific_curves`/`.build_strict_pairs` "
        "unchanged; no new PCR logic. Stratum values per dataset: HotpotQA={2} "
        "(fixed, not a real stratification axis), 2WikiMultiHopQA={2,4}, "
        "MuSiQue={2,3,4}, MultiHopRAG={2,3,4} (required-evidence-count, not "
        "ordered hops).",
        "",
        "## 2. Evidence boundaries, side by side",
        "",
        "See `evidence_boundary_paired_results.csv` -- one row per "
        "`(model, dataset, comparison_name)`, with both "
        "`final_report_input_*` columns (from `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv`) "
        "and `local_completion_*` columns (from `paired_evidence_deltas.csv`) "
        "present together. **These two boundaries are not expected to agree** "
        "(see `README.md`'s reconciliation section) -- this file exists so a "
        "reader can see both numbers in one place without cross-referencing "
        "two files by hand.",
        "",
        "## 3. Supervisor recovery, unified",
        "",
        "See `supervisor_recovery_summary.csv` -- one row per "
        "`(model, dataset, condition)`, built from "
        "`process_metrics.csv::later_evidence_recovery`'s categorical label "
        "(`src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation/recovery.py`'s `RECOVERY_*` "
        "constants). Percentages are restricted to `n_redelegated` rows "
        "(redelegation alone is never counted as recovery), matching "
        "`figures.py::plot_supervisor_recovery`'s existing convention.",
        "",
        f"Pooled across all {len(recovery_rows)} (model,dataset,condition) cells: "
        f"{sum(r['n_redelegated'] for r in recovery_rows)} total redelegated rows, "
        f"{sum(r['n_redelegated_fully_restored'] for r in recovery_rows)} fully restored, "
        f"{sum(r['n_redelegated_partially_restored'] for r in recovery_rows)} partially restored, "
        f"{sum(r['n_redelegated_did_not_restore'] for r in recovery_rows)} did not restore.",
        "",
        "## 4. Validation",
        "",
        f"**{n_pass}/{len(new_gates)} new gates passed** (appended to "
        "`validation_gates.csv` as gates 15-18, alongside the original 14):",
        "",
        "| Gate | Description | Status |",
        "|---|---|---|",
    ]
    for g in new_gates:
        lines.append(f"| {g.gate_id} | {g.description} | {'PASS' if g.passed else 'FAIL'} |")

    lines += [
        "",
        "## 5. What this is NOT",
        "",
        "This is a consolidation of already-validated numbers, not a new "
        "causal analysis and not Chapter 5. It does not resolve the "
        "final-report-input vs local-completion boundary difference (both "
        "are reported, on purpose, side by side) and it does not make an "
        "answer-harm claim (see `EVIDENCE_VS_ANSWER_HARM_NOTES.md` for that, "
        "still preliminary). `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` "
        "sections 7.4-7.6 remain the actual Chapter-5-facing narrative, still "
        "`_TODO(human)`.",
    ]
    (EHC_DIR / "EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("Building supervisor_recovery_summary.csv...")
    recovery_rows = build_supervisor_recovery_summary()
    _write_csv(EHC_DIR / "supervisor_recovery_summary.csv", recovery_rows)
    print(f"  {len(recovery_rows)} rows")

    print("Building evidence_boundary_paired_results.csv...")
    boundary_rows = build_evidence_boundary_paired_results()
    _write_csv(EHC_DIR / "evidence_boundary_paired_results.csv", boundary_rows)
    print(f"  {len(boundary_rows)} rows")

    print("Running new validation gates (15-18)...")
    new_gates = [
        gate_15_hop_pcr_monotonic(),
        gate_16_hop_pairs_subset_of_unstratified(),
        gate_17_recovery_counts_sum(recovery_rows),
        gate_18_evidence_boundary_merge_complete(boundary_rows),
    ]
    for g in new_gates:
        print(f"  [{'PASS' if g.passed else 'FAIL'}] gate {g.gate_id}: {g.description} -- {g.detail}")
    append_gates_csv(new_gates, EHC_DIR / "validation_gates.csv")

    print("Writing EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md...")
    write_report(recovery_rows, boundary_rows, new_gates)

    print("Done.")
    return 0 if all(g.passed for g in new_gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
