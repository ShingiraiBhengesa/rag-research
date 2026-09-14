"""Phase 4 of the final gap audit: RESEARCH_COMPLETE vs. NO_TOOL_DIRECT_EXIT
breakdown by checkpoint, over the SAME canonical eligible denominator used
for PCR@k. Does not change the canonical PCR definition -- reads
`canonical_run_level_completion.csv` (already computed, unmodified) and
recomputes checkpoint tables from it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL_CSV = HERE.parent / "canonical_run_level_completion.csv"
CONDITION_METADATA_CSV = HERE.parent / "condition_metadata.csv"

CHECKPOINTS = (1, 2, 3, 5, 10, 15, 20, 30)


def load_rows() -> list[dict]:
    with open(CANONICAL_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_primary_secondary_map() -> dict[str, str]:
    with open(CONDITION_METADATA_CSV, newline="", encoding="utf-8") as fh:
        return {r["condition_id"]: r["primary_or_secondary"] for r in csv.DictReader(fh)}


def checkpoint_table(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    """rows: list of canonical_run_level_completion.csv dict rows (already
    filtered to the desired grouping scope, e.g. all rows, or one model).
    Restricts BOTH numerator and denominator to pcr_eligible == True, per
    the task's explicit instruction never to divide route counts over all
    runs by n_eligible.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys) if group_keys else ("ALL",)
        grouped[key].append(r)

    out = []
    for key, grp in sorted(grouped.items()):
        eligible = [r for r in grp if r["pcr_eligible"] == "True"]
        n_eligible = len(eligible)
        max_budget_exit = sum(1 for r in grp if r["terminal_route"] == "MAX_BUDGET_EXIT")
        excluded_no_anchor = sum(1 for r in grp if r["pcr_eligible"] == "False" and r["anchor_found"] == "False")
        excluded_other = sum(1 for r in grp if r["pcr_eligible"] == "False") - excluded_no_anchor

        rc_steps = sorted(int(r["research_complete_step"]) for r in eligible if r["research_complete_step"])
        nt_steps = sorted(int(r["no_tool_direct_exit_step"]) for r in eligible if r["no_tool_direct_exit_step"])

        for k in CHECKPOINTS:
            rc_by_k = sum(1 for s in rc_steps if s <= k)
            nt_by_k = sum(1 for s in nt_steps if s <= k)
            combined_by_k = rc_by_k + nt_by_k
            rc_rate = (100.0 * rc_by_k / n_eligible) if n_eligible else None
            nt_rate = (100.0 * nt_by_k / n_eligible) if n_eligible else None
            pcr_at_k = (100.0 * combined_by_k / n_eligible) if n_eligible else None
            rc_share = (100.0 * rc_by_k / combined_by_k) if combined_by_k else None
            nt_share = (100.0 * nt_by_k / combined_by_k) if combined_by_k else None

            row = {}
            for i, gk in enumerate(group_keys):
                row[gk] = key[i]
            row.update(
                {
                    "k": k,
                    "N_eligible": n_eligible,
                    "ResearchComplete_by_k": rc_by_k,
                    "ResearchComplete_rate_by_k": rc_rate,
                    "NoToolDirectExit_by_k": nt_by_k,
                    "NoToolDirectExit_rate_by_k": nt_rate,
                    "Combined_completion_by_k": combined_by_k,
                    "PCR_at_k": pcr_at_k,
                    "ResearchComplete_share_among_completions": rc_share,
                    "DirectExit_share_among_completions": nt_share,
                    "share_sum_check": (rc_share + nt_share) if (rc_share is not None and nt_share is not None) else None,
                    "identity_check_pcr_eq_rc_plus_nt": (
                        abs(pcr_at_k - (rc_rate + nt_rate)) < 1e-6 if (pcr_at_k is not None and rc_rate is not None and nt_rate is not None) else None
                    ),
                    "MaxBudgetExit_n": max_budget_exit,
                    "excluded_no_search_anchor_n": excluded_no_anchor,
                    "excluded_other_ineligible_n": excluded_other,
                }
            )
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    rows = load_rows()
    ps_map = load_primary_secondary_map()

    write_csv(HERE / "route_breakdown_overall.csv", checkpoint_table(rows, ()))
    write_csv(HERE / "route_breakdown_by_model.csv", checkpoint_table(rows, ("model",)))
    write_csv(HERE / "route_breakdown_by_dataset.csv", checkpoint_table(rows, ("dataset",)))
    write_csv(HERE / "route_breakdown_by_condition.csv", checkpoint_table(rows, ("condition",)))

    # Full model x dataset x condition breakdown (the master checkpoint table)
    write_csv(HERE / "route_breakdown_by_checkpoint.csv", checkpoint_table(rows, ("model", "dataset", "condition")))
    write_csv(HERE / "route_breakdown_by_model_dataset_condition.csv", checkpoint_table(rows, ("model", "dataset", "condition")))
    write_csv(HERE / "route_breakdown_by_model_dataset.csv", checkpoint_table(rows, ("model", "dataset")))

    primary_rows = [r for r in rows if ps_map.get(r["condition"]) == "Primary"]
    secondary_rows = [r for r in rows if ps_map.get(r["condition"]) == "Secondary ablation"]
    write_csv(HERE / "route_breakdown_primary_conditions_only.csv", checkpoint_table(primary_rows, ("model", "dataset", "condition")))
    write_csv(HERE / "route_breakdown_secondary_ablations_only.csv", checkpoint_table(secondary_rows, ("model", "dataset", "condition")))

    # Identity verification pass (small tolerance)
    all_checks = checkpoint_table(rows, ("model", "dataset", "condition"))
    bad = [r for r in all_checks if r["identity_check_pcr_eq_rc_plus_nt"] is False]
    share_bad = [
        r for r in all_checks
        if r["share_sum_check"] is not None and abs(r["share_sum_check"] - 100.0) > 1e-6
    ]
    print(f"Total (model,dataset,condition,k) rows: {len(all_checks)}")
    print(f"PCR@k == RC_rate+NT_rate identity failures: {len(bad)}")
    print(f"RC_share+NT_share == 100% identity failures: {len(share_bad)}")
    if bad:
        print("First failure:", bad[0])
    if share_bad:
        print("First share failure:", share_bad[0])


if __name__ == "__main__":
    main()
