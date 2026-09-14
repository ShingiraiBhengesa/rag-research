"""Phase 6: reconcile the canonical implementation against the pre-existing
`analysis/pcr.py` outputs already on disk at
results/evaluation/
v3_final_completed_qwen_llama_20260910/pcr/*.csv (generated 2026-09-10,
read-only input to this script -- never modified or regenerated here).

Comparison is at CELL granularity (model, dataset, condition), because
pcr.py's outputs are pre-aggregated (no row-level example_id is retained in
its CSVs) -- this is stated explicitly rather than silently treated as a
row-level diff.
"""

from __future__ import annotations

import csv
from pathlib import Path

from analysis import config

LEGACY_DIR = config.DEFAULT_OUTPUT_ROOT / "pcr"
HERE = Path(__file__).resolve().parent


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    legacy_denom = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(LEGACY_DIR / "pcr_denominator_audit.csv")}
    legacy_cdf = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(LEGACY_DIR / "pcr_completion_cdf_wide.csv")}
    legacy_unc = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(LEGACY_DIR / "pcr_unconditional_incidence_wide.csv")}
    canonical = {(r["model"], r["dataset"], r["condition"]): r for r in _read_csv(HERE / "canonical_pcr_by_condition.csv")}

    checkpoints = (1, 3, 5)
    rows_out = []
    n_cells_compared = 0
    n_cells_eligible_n_differs = 0
    n_cells_pcr1_differs_from_5_1 = 0
    n_cells_pcr1_differs_from_5_2 = 0
    max_abs_diff_5_1 = 0.0
    max_abs_diff_5_2 = 0.0

    all_keys = sorted(set(legacy_denom) & set(canonical))
    for key in all_keys:
        model, dataset, condition = key
        d = legacy_denom[key]
        cdf = legacy_cdf.get(key, {})
        unc = legacy_unc.get(key, {})
        can = canonical[key]

        legacy_n_total = int(d["n_valid_completed_rows"])
        legacy_n_cdf_denom = int(d["n_completion_cdf_denominator"])
        legacy_n_unc_denom = int(d["n_unconditional_incidence_denominator"])
        canonical_n_total = int(can["n_total"])
        canonical_n_eligible = int(can["n_eligible"])

        row = {
            "model": model, "dataset": dataset, "condition": condition,
            "legacy_n_valid_completed_rows": legacy_n_total,
            "legacy_completion_time_denominator_5_1": legacy_n_cdf_denom,
            "legacy_unconditional_denominator_5_2": legacy_n_unc_denom,
            "canonical_n_total": canonical_n_total,
            "canonical_n_eligible": canonical_n_eligible,
            "n_execution_failure_canonical": int(can.get("n_EXECUTION_FAILURE", 0)),
            "n_max_budget_exit_canonical": int(can.get("n_MAX_BUDGET_EXIT", 0)),
        }
        n_cells_compared += 1
        if canonical_n_eligible != legacy_n_cdf_denom:
            n_cells_eligible_n_differs += 1

        for cp in checkpoints:
            cdf_key = f"combined_pct@{cp}"
            unc_key = f"cum_pct@{cp}"
            can_key = f"pcr@{cp}"
            legacy_5_1 = float(cdf[cdf_key]) if cdf.get(cdf_key) not in (None, "") else None
            legacy_5_2 = float(unc[unc_key]) if unc.get(unc_key) not in (None, "") else None
            canon = float(can[can_key]) if can.get(can_key) not in (None, "") else None
            row[f"legacy_5_1_pcr@{cp}"] = legacy_5_1
            row[f"legacy_5_2_pcr@{cp}"] = legacy_5_2
            row[f"canonical_pcr@{cp}"] = canon
            if canon is not None and legacy_5_1 is not None:
                diff = abs(canon - legacy_5_1)
                row[f"abs_diff_vs_5_1@{cp}"] = diff
                max_abs_diff_5_1 = max(max_abs_diff_5_1, diff)
                if cp == 1 and diff > 0.05:
                    n_cells_pcr1_differs_from_5_1 += 1
            if canon is not None and legacy_5_2 is not None:
                diff2 = abs(canon - legacy_5_2)
                row[f"abs_diff_vs_5_2@{cp}"] = diff2
                max_abs_diff_5_2 = max(max_abs_diff_5_2, diff2)
                if cp == 1 and diff2 > 0.05:
                    n_cells_pcr1_differs_from_5_2 += 1
        rows_out.append(row)

    fieldnames: list[str] = []
    seen = set()
    for r in rows_out:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(HERE / "canonical_vs_previous_row_differences.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    summary = {
        "n_cells_compared": n_cells_compared,
        "n_cells_eligible_n_differs_from_5_1_denominator": n_cells_eligible_n_differs,
        "n_cells_pcr_at_1_differs_from_5_1_by_gt_0.05pp": n_cells_pcr1_differs_from_5_1,
        "n_cells_pcr_at_1_differs_from_5_2_by_gt_0.05pp": n_cells_pcr1_differs_from_5_2,
        "max_abs_diff_vs_5_1_any_checkpoint": max_abs_diff_5_1,
        "max_abs_diff_vs_5_2_any_checkpoint": max_abs_diff_5_2,
    }
    print(summary)
    with open(HERE / "_reconcile_summary.txt", "w", encoding="utf-8") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
