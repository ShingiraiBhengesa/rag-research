"""Phase 2: mutually-exclusive coverage categories and per-cell summary
statistics."""
from __future__ import annotations

import statistics
from collections import defaultdict

from .source_loader import CoverageRow

EPS = 1e-9

CATEGORY_NO_EVIDENCE = "No required evidence"
CATEGORY_LOW = "Low coverage"
CATEGORY_HIGH_INCOMPLETE = "High but incomplete"
CATEGORY_FULL = "Full coverage"
CATEGORIES = (CATEGORY_NO_EVIDENCE, CATEGORY_LOW, CATEGORY_HIGH_INCOMPLETE, CATEGORY_FULL)

# Recommended, colorblind-safe (Okabe-Ito derived) palette; consistent with
# figures_slide.py / figures_ecdf.py / figures_missing.py.
CATEGORY_COLOR = {
    CATEGORY_NO_EVIDENCE: "#9B2226",         # dark red
    CATEGORY_LOW: "#CA6702",                  # orange
    CATEGORY_HIGH_INCOMPLETE: "#94D2BD",      # light blue/teal
    CATEGORY_FULL: "#005F3C",                 # dark green
}


def categorize(coverage: float) -> str:
    if coverage <= 0.0 + EPS:
        return CATEGORY_NO_EVIDENCE
    if coverage >= 1.0 - EPS:
        return CATEGORY_FULL
    if coverage < 0.5 - EPS:
        return CATEGORY_LOW
    return CATEGORY_HIGH_INCOMPLETE


def summarize_cell(rows: list[CoverageRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "eligible_N": 0,
            **{f"n_{c}": 0 for c in CATEGORIES},
            **{f"pct_{c}": None for c in CATEGORIES},
            "mean_coverage": None, "median_coverage": None, "stdev_coverage": None,
            "iqr_coverage": None, "mean_missing": None, "median_missing": None,
        }
    coverages = [r.coverage for r in rows]
    missing = [r.missing for r in rows]
    cat_counts = {c: 0 for c in CATEGORIES}
    for r in rows:
        cat_counts[categorize(r.coverage)] += 1

    sorted_cov = sorted(coverages)

    def _quantile(data: list[float], q: float) -> float:
        if len(data) == 1:
            return data[0]
        return statistics.quantiles(data, n=100, method="inclusive")[int(round(q * 100)) - 1]

    q1 = _quantile(sorted_cov, 0.25) if n > 1 else sorted_cov[0]
    q3 = _quantile(sorted_cov, 0.75) if n > 1 else sorted_cov[0]

    return {
        "eligible_N": n,
        **{f"n_{c}": cat_counts[c] for c in CATEGORIES},
        **{f"pct_{c}": round(cat_counts[c] / n * 100, 2) for c in CATEGORIES},
        "mean_coverage": round(statistics.fmean(coverages), 4),
        "median_coverage": round(statistics.median(coverages), 4),
        "stdev_coverage": round(statistics.stdev(coverages), 4) if n > 1 else 0.0,
        "iqr_coverage": round(q3 - q1, 4),
        "mean_missing": round(statistics.fmean(missing), 4),
        "median_missing": round(statistics.median(missing), 4),
    }


def build_coverage_distribution_table(rows: list[CoverageRow], model_datasets: dict[str, tuple[str, ...]],
                                       all_conditions: tuple[str, ...]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[CoverageRow]] = defaultdict(list)
    for r in rows:
        by_cell[(r.model, r.dataset, r.condition)].append(r)

    out = []
    for model, datasets in model_datasets.items():
        for dataset in datasets:
            for cond in all_conditions:
                cell_rows = by_cell.get((model, dataset, cond), [])
                stats = summarize_cell(cell_rows)
                out.append({"model": model, "dataset": dataset, "condition": cond, **stats})
    return out
