"""Step 5: `data/pcr_paired_deltas_stepwise.csv`.

Sign convention (confirmed with user during planning): delta_pcr =
cue_pcr - control_pcr. Positive = the cue condition completed MORE by step k
than control, i.e. positive means "the attack accelerated completion."

Both curves in a comparison are recomputed on the exact same paired set
(`canonical_completion.build_strict_pairs`, which excludes rather than
imputes any row that isn't `pcr_eligible` on both sides -- structurally
satisfies validation gate 5/6). The confidence interval on the delta is a
genuine PAIRED bootstrap CI, computed with the *same* resampling logic
`attack.metrics.paired_bootstrap_ci` uses (mean-of-boolean-vector diff,
percentile CI) but vectorized across all 30 steps in one bootstrap pass per
comparison instead of calling that function once per step -- calling it
2640 times (30 steps x 11 comparisons x 2 models x 4 datasets), each with
its own O(n_boot) pure-Python loop (attack/metrics.py:78-98), is not
tractable for this bundle's full sweep. The per-step completion vectors for
one comparison are highly correlated (T_i <= step is monotonic in step for
a fixed run), so resampling once and evaluating all 30 thresholds against
the same resample indices is both faster and statistically more coherent
than 30 independent resamples would be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis import config
from analysis.canonical import canonical_completion as cc

from .comparisons import COMPARISONS
from .pcr_full import END_OF_BUDGET_STEP

# config.BOOTSTRAP_ITERATIONS (10000) is calibrated for the 3-checkpoint
# design statistics.py originally used. Reduced to 2000 for this bundle's
# full 30-step sweep -- still a stable percentile-CI estimate, documented
# here explicitly rather than silently diverging from the rest of the
# analysis package's constant.
STEPWISE_BOOTSTRAP_ITERATIONS = 2000


def _t_i_array(records: list[cc.CanonicalCompletionRecord]) -> np.ndarray:
    """T_i if the run completed via a PCR-numerator route, else +inf (never
    <= any real step, so it never counts as completed -- avoids a None/NaN
    special case in the vectorized comparison below)."""
    out = np.full(len(records), np.inf)
    for i, r in enumerate(records):
        if r.terminal_route in cc.PCR_NUMERATOR_ROUTES and r.first_terminal_post_observation_step is not None:
            out[i] = r.first_terminal_post_observation_step
    return out


def _bootstrap_all_steps(
    cue_t_i: np.ndarray, control_t_i: np.ndarray, steps: np.ndarray, n_boot: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (control_pcr, cue_pcr, delta_pcr, ci_lower, ci_upper), each an
    array indexed by `steps`. One resampling pass shared across all steps.
    """
    n = len(cue_t_i)
    # (n_paired, n_steps) boolean: completed_by_step[i, s] = cue_t_i[i] <= steps[s]
    cue_completed = cue_t_i[:, None] <= steps[None, :]
    control_completed = control_t_i[:, None] <= steps[None, :]
    control_pcr = control_completed.mean(axis=0)
    cue_pcr = cue_completed.mean(axis=0)
    delta_pcr = cue_pcr - control_pcr

    rng = np.random.default_rng(seed)
    # (n_boot, n_steps) accumulator of resampled deltas.
    boot_deltas = np.empty((n_boot, len(steps)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = cue_completed[idx].mean(axis=0) - control_completed[idx].mean(axis=0)
    lo, hi = np.quantile(boot_deltas, [0.025, 0.975], axis=0)
    return control_pcr, cue_pcr, delta_pcr, lo, hi


def write_pcr_paired_deltas_stepwise_csv(
    records: list[cc.CanonicalCompletionRecord], path: Path
) -> None:
    steps = np.arange(1, END_OF_BUDGET_STEP + 1)
    grouped = cc._group(records, lambda r: (r.model, r.dataset))  # noqa: SLF001
    rows: list[dict] = []
    for (model, dataset), recs in sorted(grouped.items()):
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
                            "model": model, "dataset": dataset,
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
                        "model": model, "dataset": dataset,
                        "comparison_name": comp.comparison_name,
                        "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                        "step": int(step),
                        "n_control_eligible": n_control_eligible, "n_cue_eligible": n_cue_eligible,
                        "n_paired": n_paired,
                        "control_pcr": float(control_pcr[i]), "cue_pcr": float(cue_pcr[i]), "delta_pcr": float(delta_pcr[i]),
                        "ci_lower": float(ci_lo[i]), "ci_upper": float(ci_hi[i]),
                    }
                )
    cc._write_csv(path, rows)  # noqa: SLF001
