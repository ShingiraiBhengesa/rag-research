"""Phase 6 (paired evidence-loss distribution) and Phase 7 (full-coverage
transitions, exact McNemar test). Every comparison is computed per
model x dataset -- never pooled across datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source_loader import CoverageRow

N_BOOT = 10_000
BOOTSTRAP_SEED = 0

# (comparison_name, control_condition, cue_condition)
COMPARISONS = (
    ("Epistemic Confirmation vs Clean", "NaturalClean", "Natural_epistemic_confirmation"),
    ("Search Match vs Clean", "NaturalClean", "Natural_search_match_cue"),
    ("Fixed Sufficiency vs Clean", "NaturalClean", "Natural_PreRegisteredSufficiency"),
    ("Fixed Sufficiency Saturation vs Clean", "NaturalClean", "Natural_PreRegisteredSufficiencySaturation"),
    ("Source Coverage vs Natural Neutral", "NaturalNeutral", "Natural_source_coverage"),
    ("Epistemic Confirmation + Source Coverage vs Epistemic Confirmation", "Natural_epistemic_confirmation", "Natural_epistemic_confirmation_source"),
    ("Search Match + Source Coverage vs Search Match", "Natural_search_match_cue", "Natural_search_match_source"),
    ("Oracle Complete--Saturation vs Oracle Complete--Neutral", "OracleComplete_Neutral", "OracleComplete_PreRegisteredSufficiencySaturation"),
)
CALIBRATION_COMPARISONS = {"Oracle Complete--Saturation vs Oracle Complete--Neutral"}


def pair_by_example_id(control_rows: list[CoverageRow], cue_rows: list[CoverageRow]) -> list[tuple[CoverageRow, CoverageRow]]:
    """Intersects on example_id only -- never pads or imputes a missing partner."""
    control_by_id = {r.example_id: r for r in control_rows}
    cue_by_id = {r.example_id: r for r in cue_rows}
    shared_ids = sorted(set(control_by_id) & set(cue_by_id))
    return [(control_by_id[i], cue_by_id[i]) for i in shared_ids]


def paired_bootstrap_mean_diff(control_vals: list[float], cue_vals: list[float], n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED):
    """control - cue, matched-pair resampling (same index drawn for both
    arrays every resample -- never resampled independently). Returns
    (point, ci_lower, ci_upper)."""
    a = np.asarray(control_vals, dtype=float)
    b = np.asarray(cue_vals, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return point, float(lo), float(hi)


def paired_loss_row(model: str, dataset: str, comparison_name: str, control_cond: str, cue_cond: str,
                     rows: list[CoverageRow]) -> dict:
    control_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == control_cond]
    cue_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == cue_cond]
    pairs = pair_by_example_id(control_rows, cue_rows)
    n = len(pairs)
    base = {
        "model": model, "dataset": dataset, "comparison_name": comparison_name,
        "control_condition": control_cond, "cue_condition": cue_cond,
        "is_calibration": comparison_name in CALIBRATION_COMPARISONS,
        "paired_N": n,
    }
    if n == 0:
        return {**base, "pct_evidence_loss": None, "pct_no_change": None, "pct_evidence_gain": None,
                "mean_paired_loss": None, "median_paired_loss": None, "ci_lower": None, "ci_upper": None,
                "mean_missing_increase": None, "pct_losing_at_least_one_unit": None}

    control_cov = [p[0].coverage for p in pairs]
    cue_cov = [p[1].coverage for p in pairs]
    # positive = control retrieved MORE (cue lost evidence); per spec:
    # "paired evidence loss = control coverage - cue coverage"
    diffs = [c - q for c, q in zip(control_cov, cue_cov)]
    n_loss = sum(1 for d in diffs if d > 1e-9)
    n_gain = sum(1 for d in diffs if d < -1e-9)
    n_same = n - n_loss - n_gain
    missing_increase = [p[1].missing - p[0].missing for p in pairs]  # positive = cue missing MORE units
    n_losing_unit = sum(1 for d in missing_increase if d > 0)

    import statistics
    point, lo, hi = paired_bootstrap_mean_diff(control_cov, cue_cov)
    return {
        **base,
        "pct_evidence_loss": round(n_loss / n * 100, 2),
        "pct_no_change": round(n_same / n * 100, 2),
        "pct_evidence_gain": round(n_gain / n * 100, 2),
        "mean_paired_loss": round(point, 4),
        "median_paired_loss": round(statistics.median(diffs), 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "mean_missing_increase": round(statistics.fmean(missing_increase), 4),
        "pct_losing_at_least_one_unit": round(n_losing_unit / n * 100, 2),
    }


def build_paired_loss_table(rows: list[CoverageRow], model_datasets: dict[str, tuple[str, ...]]) -> list[dict]:
    out = []
    for model, datasets in model_datasets.items():
        for dataset in datasets:
            for comparison_name, control_cond, cue_cond in COMPARISONS:
                out.append(paired_loss_row(model, dataset, comparison_name, control_cond, cue_cond, rows))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs (b, c), implemented
    directly with the stdlib (no scipy/statsmodels dependency -- this
    machine's shared load made scipy's import take 3+ minutes of wall time
    during development, 200s of which was pure I/O wait, not CPU; the
    binomial(n, 0.5) two-sided p-value below is the standard exact-McNemar
    formulation and was cross-checked against scipy.stats.binomtest(3, 15,
    0.5, alternative='two-sided') -> 0.03515625, exact match both ways).

    p = min(1, 2 * P(X <= min(b,c))) under X ~ Binomial(b+c, 0.5).
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def full_coverage_transition_row(model: str, dataset: str, comparison_name: str, control_cond: str, cue_cond: str,
                                  rows: list[CoverageRow]) -> dict:
    control_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == control_cond]
    cue_rows = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == cue_cond]
    pairs = pair_by_example_id(control_rows, cue_rows)
    n = len(pairs)
    base = {
        "model": model, "dataset": dataset, "comparison_name": comparison_name,
        "control_condition": control_cond, "cue_condition": cue_cond,
        "is_calibration": comparison_name in CALIBRATION_COMPARISONS,
        "paired_N": n,
    }
    if n == 0:
        return {**base, "n_full_both": None, "n_full_control_only": None, "n_full_cue_only": None,
                "n_full_neither": None, "control_to_cue_full_coverage_loss_rate": None,
                "cue_associated_full_coverage_gain_rate": None, "net_paired_difference": None,
                "mcnemar_p_value": None}

    full_both = sum(1 for c, q in pairs if c.full_coverage and q.full_coverage)
    full_control_only = sum(1 for c, q in pairs if c.full_coverage and not q.full_coverage)
    full_cue_only = sum(1 for c, q in pairs if not c.full_coverage and q.full_coverage)
    full_neither = n - full_both - full_control_only - full_cue_only

    p_value = mcnemar_exact(full_control_only, full_cue_only)
    return {
        **base,
        "n_full_both": full_both, "n_full_control_only": full_control_only,
        "n_full_cue_only": full_cue_only, "n_full_neither": full_neither,
        "control_to_cue_full_coverage_loss_rate": round(full_control_only / n * 100, 2),
        "cue_associated_full_coverage_gain_rate": round(full_cue_only / n * 100, 2),
        "net_paired_difference": round((full_control_only - full_cue_only) / n * 100, 2),
        "mcnemar_p_value": round(p_value, 6),
    }


def build_full_coverage_transitions_table(rows: list[CoverageRow], model_datasets: dict[str, tuple[str, ...]]) -> list[dict]:
    out = []
    for model, datasets in model_datasets.items():
        for dataset in datasets:
            for comparison_name, control_cond, cue_cond in COMPARISONS:
                out.append(full_coverage_transition_row(model, dataset, comparison_name, control_cond, cue_cond, rows))
    return out
