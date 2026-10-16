"""Phase 5: missing-required-evidence-unit distribution, overall and
stratified by required_dependency_count (2/3/4) where the stratum has
enough rows to be meaningful."""
from __future__ import annotations

import statistics
from collections import defaultdict

from .source_loader import CoverageRow

MISSING_ZERO = "zero missing"
MISSING_ONE = "one missing"
MISSING_TWO = "two missing"
MISSING_THREE_PLUS = "three or more missing"
MISSING_CATEGORIES = (MISSING_ZERO, MISSING_ONE, MISSING_TWO, MISSING_THREE_PLUS)

# Minimum denominator for a hop-count stratum to be reported at all --
# below this, a per-stratum percentage is too noisy to show on a slide.
# Documented, not hidden: every row below states its own denominator
# regardless, so a reader can judge for themselves.
MIN_STRATUM_N = 30


def _missing_category(missing: int) -> str:
    if missing <= 0:
        return MISSING_ZERO
    if missing == 1:
        return MISSING_ONE
    if missing == 2:
        return MISSING_TWO
    return MISSING_THREE_PLUS


def summarize_missing(rows: list[CoverageRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "eligible_N": 0,
            **{f"n_{c}": 0 for c in MISSING_CATEGORIES},
            **{f"pct_{c}": None for c in MISSING_CATEGORIES},
            "mean_missing": None, "median_missing": None,
        }
    missing_vals = [r.missing for r in rows]
    cat_counts = {c: 0 for c in MISSING_CATEGORIES}
    for r in rows:
        cat_counts[_missing_category(r.missing)] += 1
    return {
        "eligible_N": n,
        **{f"n_{c}": cat_counts[c] for c in MISSING_CATEGORIES},
        **{f"pct_{c}": round(cat_counts[c] / n * 100, 2) for c in MISSING_CATEGORIES},
        "mean_missing": round(statistics.fmean(missing_vals), 4),
        "median_missing": round(statistics.median(missing_vals), 4),
    }


def build_missing_evidence_table(rows: list[CoverageRow], model_datasets: dict[str, tuple[str, ...]],
                                  all_conditions: tuple[str, ...]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[CoverageRow]] = defaultdict(list)
    for r in rows:
        by_cell[(r.model, r.dataset, r.condition)].append(r)

    out = []
    for model, datasets in model_datasets.items():
        for dataset in datasets:
            for cond in all_conditions:
                cell_rows = by_cell.get((model, dataset, cond), [])
                stats = summarize_missing(cell_rows)
                out.append({"model": model, "dataset": dataset, "condition": cond, "stratum": "all", **stats})

                by_required: dict[int, list[CoverageRow]] = defaultdict(list)
                for r in cell_rows:
                    by_required[r.required].append(r)
                for required_n in (2, 3, 4):
                    stratum_rows = by_required.get(required_n, [])
                    if len(stratum_rows) < MIN_STRATUM_N:
                        continue  # denominator too small to report meaningfully -- see MIN_STRATUM_N
                    stats = summarize_missing(stratum_rows)
                    out.append({
                        "model": model, "dataset": dataset, "condition": cond,
                        "stratum": f"required={required_n}", **stats,
                    })
    return out
