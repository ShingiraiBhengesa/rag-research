"""Section 14: paired statistics at the prespecified PCR checkpoints.

Reuses the existing, validated `attack.metrics` helpers
(`wilson_ci`, `mcnemar_test`, `paired_bootstrap_ci`) rather than
re-implementing them (imported read-only; that module is never modified).

Every effect is reported separately by model and dataset (never averaged),
using ONLY the strict shared-prefix pairs from `pairing.py` -- an unpaired
row must never contribute to a paired McNemar/bootstrap result.

"Completed by step k" is defined as `terminal_post_branch_index is not None
and <= k`, but ONLY for rows where `branch_reached` is True on BOTH sides of
the pair -- a row where episode 0 never executed a search at all is not
"did not complete by k", it is "the cue was never administered" (the
suffix is only appended to the search tool's returned message), so
including it as a False outcome would silently invert the measured effect.
Rows that DID branch but never observed a terminal route within the
recorded episode (censored, or a logging gap) are still treated as `False`
at every k -- they certainly did not complete AT OR BEFORE k, whatever
happens later or off-trace.

Multiple-comparison correction (Benjamini-Hochberg) is applied within the
prespecified family: {PCR@1, PCR@3, PCR@5} x {each attack condition vs its
two baselines} x {model, dataset} -- the full per-step CDF remains
descriptive only, per the task's explicit instruction not to run an
uncorrected test at every step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from attack.metrics import mcnemar_test, paired_bootstrap_ci, wilson_ci
from analysis import config
from analysis.pcr import PCRRecord

ATTACK_CONDITIONS: tuple[str, ...] = (
    config.CUE_ONLY_ATTACK_CONDITIONS + config.SOURCE_ATTACHED_ATTACK_CONDITIONS + config.PREREGISTERED_SUFFICIENCY_CONDITIONS
)

# Mirrors Section 7's "Comparisons" list exactly, each pair listed once (a
# McNemar/bootstrap test on (a, b) is symmetric information to (b, a), so
# listing both directions would double-count the same test in the BH family).
COMPARISON_FAMILY: tuple[tuple[str, str], ...] = (
    ("NaturalClean", "NaturalNeutral"),  # suffix/format effect
    *(("NaturalNeutral", cond) for cond in ATTACK_CONDITIONS),  # semantic effect of the cue
    *(("NaturalClean", cond) for cond in ATTACK_CONDITIONS),  # total practical effect
    ("NaturalNeutral", "NaturalContinueSearch"),  # directional-control check
    ("OracleComplete_Neutral", "OracleComplete_PreRegisteredSufficiencySaturation"),  # evidence-complete calibration
)


@dataclass
class PairedCheckpointResult:
    model: str
    dataset: str
    condition_a: str
    condition_b: str
    checkpoint: int
    n_pairs: int
    a_successes: int
    b_successes: int
    pct_point_a: float
    pct_point_b: float
    pct_point_diff: float
    bootstrap_ci_lo: float
    bootstrap_ci_hi: float
    bootstrap_p_value: float
    mcnemar_stat: float
    mcnemar_p_value: float
    mcnemar_n_discordant: int
    bh_adjusted_p_value: Optional[float] = None


def _completed_by_k(record: PCRRecord, k: int) -> bool:
    return record.terminal_post_branch_index is not None and record.terminal_post_branch_index <= k


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Standard BH step-up procedure. Returns adjusted p-values in the input order."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(indexed), start=1):
        i = n - rank + 1
        value = min(prev, p_values[idx] * n / i)
        adjusted[idx] = value
        prev = value
    return adjusted


def compute_paired_checkpoint_stats(
    records_by_key: dict[tuple[str, str, str, str], PCRRecord],
    strictly_paired_example_ids: dict[tuple[str, str, str, str], set[str]],
    models: tuple[str, ...] = config.INCLUDED_MODELS,
    datasets: tuple[str, ...] = config.DATASETS,
    checkpoints: tuple[int, ...] = config.PCR_CHECKPOINTS,
    seed: int = config.DETERMINISTIC_SEED,
) -> list[PairedCheckpointResult]:
    """`records_by_key`: (model, dataset, condition, example_id) -> PCRRecord.
    `strictly_paired_example_ids`: (model, dataset, condition_a, condition_b) -> set of
    example_ids that passed the strict pairing audit for that comparison.
    """
    results: list[PairedCheckpointResult] = []
    for model in models:
        for dataset in datasets:
            for baseline, cond in COMPARISON_FAMILY:
                paired_ids = strictly_paired_example_ids.get((model, dataset, baseline, cond), set())
                if not paired_ids:
                    continue
                for k in checkpoints:
                    a_vals = []
                    b_vals = []
                    for eid in sorted(paired_ids):
                        rec_a = records_by_key.get((model, dataset, baseline, eid))
                        rec_b = records_by_key.get((model, dataset, cond, eid))
                        if rec_a is None or rec_b is None:
                            continue
                        # A branch never reached (no search executed in episode 0) means
                        # the cue's suffix -- only appended to the search tool's returned
                        # message -- was never actually administered for a non-Clean
                        # condition; counting that as "did not complete by k" would
                        # silently invert the measured effect (verified empirically: this
                        # exact bug produced a spurious large NEGATIVE effect for
                        # llama8b/multihoprag/PreRegisteredSufficiency, where 32% of rows
                        # never searched at all). Both sides of the pair must have reached
                        # their own branch point to be comparable.
                        if not (rec_a.branch_reached and rec_b.branch_reached):
                            continue
                        a_vals.append(_completed_by_k(rec_a, k))
                        b_vals.append(_completed_by_k(rec_b, k))
                    n = len(a_vals)
                    if n == 0:
                        continue
                    a_succ = sum(a_vals)
                    b_succ = sum(b_vals)
                    point_a, _, _ = wilson_ci(a_succ, n)
                    point_b, _, _ = wilson_ci(b_succ, n)
                    boot_point, boot_lo, boot_hi, boot_p = paired_bootstrap_ci(
                        [float(v) for v in b_vals], [float(v) for v in a_vals], seed=seed
                    )
                    mc_stat, mc_p, mc_n_disc = mcnemar_test(a_vals, b_vals)
                    results.append(
                        PairedCheckpointResult(
                            model=model, dataset=dataset, condition_a=baseline, condition_b=cond, checkpoint=k,
                            n_pairs=n, a_successes=a_succ, b_successes=b_succ,
                            pct_point_a=100.0 * point_a, pct_point_b=100.0 * point_b,
                            pct_point_diff=100.0 * (point_b - point_a),
                            bootstrap_ci_lo=100.0 * boot_lo, bootstrap_ci_hi=100.0 * boot_hi,
                            bootstrap_p_value=boot_p,
                            mcnemar_stat=mc_stat, mcnemar_p_value=mc_p, mcnemar_n_discordant=mc_n_disc,
                        )
                    )

    p_values = [r.mcnemar_p_value for r in results]
    adjusted = benjamini_hochberg(p_values)
    for r, adj in zip(results, adjusted):
        r.bh_adjusted_p_value = adj
    return results


def write_paired_checkpoint_stats(results: list[PairedCheckpointResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "dataset", "condition_a", "condition_b", "checkpoint", "n_pairs",
        "a_successes", "b_successes", "pct_point_a", "pct_point_b", "pct_point_diff",
        "bootstrap_ci_lo", "bootstrap_ci_hi", "bootstrap_p_value",
        "mcnemar_stat", "mcnemar_p_value", "mcnemar_n_discordant", "bh_adjusted_p_value",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({f: getattr(r, f) for f in fieldnames})
