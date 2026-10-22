"""E1 sensitivity checks: identical-d1 PCR contrasts and technical failures by condition.

1. Identical-d1 sensitivity (the strict observation subset promised in the
   methods): paired ΔPCR@k for every comparison in retrieval_equivalence.csv,
   once on all strictly paired runs (reproduces data/pcr_paired_deltas_stepwise.csv)
   and once restricted to pairs whose pre-cue first search observation is
   byte-identical (`identical_base_tool_message_hash`). Same T_i definition,
   pairing rule, bootstrap (2000 resamples, one pass across steps) and seed as
   results_bundle/paired_deltas.py.

2. Technical failures by condition: n_failures / n_completed per model x
   dataset x condition from data/run_inventory.csv, with a chi-square test of
   failure against condition per model x dataset (and pooled per model).

Inputs are the published bundles; outputs go to
results/whitebox/e1_fixed_cue_matrix/sensitivity/.

    python -m prematurestop.analysis.fixed_cue_matrix.sensitivity_checks
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from prematurestop.analysis.fixed_cue_matrix.results_bundle.paired_deltas import (
    STEPWISE_BOOTSTRAP_ITERATIONS,
    _bootstrap_all_steps,
)
from prematurestop.evaluation import completion as cc
from prematurestop.evaluation import config
from prematurestop.paths import RESULTS_E1

BUNDLES = ("qwen_llama", "gemma_gptoss")
OUT = RESULTS_E1 / "sensitivity"
STEPS = np.arange(1, 31)
CHAR = {"qwen14b": 1, "llama8b": 3, "gemma27b": 1, "gpt_oss_20b": 2}  # characteristic steps
SUMMARY_COMPARISONS = (("Natural_PreRegisteredSufficiency", "NaturalNeutral", "Fixed Sufficiency"),
                       ("Natural_PreRegisteredSufficiencySaturation", "NaturalNeutral", "Suff. + Saturation"))


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def t_i(r: dict) -> float:
    """T_i if the run completed via a PCR-numerator route, else +inf (as in paired_deltas._t_i_array)."""
    step = r["first_terminal_post_observation_step"]
    if r["terminal_route"] in cc.PCR_NUMERATOR_ROUTES and step not in ("", "None"):
        return float(step)
    return np.inf


# --------------------------------------------------------------------------- 1. identical-d1 sensitivity
def identical_d1_rows() -> list[dict]:
    runs, ident = {}, {}
    for b in BUNDLES:
        for r in read(RESULTS_E1 / b / "data" / "pcr_run_level.csv"):
            if r["pcr_eligible"] == "True":
                runs[(r["model"], r["dataset"], r["condition"], r["example_id"])] = t_i(r)
        for r in read(RESULTS_E1 / b / "data" / "retrieval_equivalence.csv"):
            ident[(r["model"], r["dataset"], r["condition"], r["baseline"], r["example_id"])] = (
                r["identical_base_tool_message_hash"] == "True")
    published = {}
    for b in BUNDLES:
        for r in read(RESULTS_E1 / b / "data" / "pcr_paired_deltas_stepwise.csv"):
            published[(r["model"], r["dataset"], r["cue_condition"], r["control_condition"], int(r["step"]))] = r

    cells = sorted({(m, d, c, base) for m, d, c, base, _ in ident})
    rows = []
    for m, d, cue, ctl in cells:
        qids = sorted(q for (mm, dd, cc_, bb, q) in ident if (mm, dd, cc_, bb) == (m, d, cue, ctl))
        paired = [q for q in qids if (m, d, cue, q) in runs and (m, d, ctl, q) in runs]
        n_pairs_total = len(qids)
        for subset, keep in (("all_paired", paired), ("identical_d1", [q for q in paired if ident[(m, d, cue, ctl, q)]])):
            row0 = {"model": m, "dataset": d, "cue_condition": cue, "control_condition": ctl, "subset": subset,
                    "n_pairs_in_retrieval_audit": n_pairs_total,
                    "n_identical_d1": sum(ident[(m, d, cue, ctl, q)] for q in qids)}
            if not keep:
                rows += [{**row0, "step": int(s), "n_paired": 0, "control_pcr": None, "cue_pcr": None,
                          "delta_pcr": None, "ci_lower": None, "ci_upper": None} for s in STEPS]
                continue
            cue_t = np.array([runs[(m, d, cue, q)] for q in keep])
            ctl_t = np.array([runs[(m, d, ctl, q)] for q in keep])
            cpcr, qpcr, delta, lo, hi = _bootstrap_all_steps(cue_t, ctl_t, STEPS, STEPWISE_BOOTSTRAP_ITERATIONS,
                                                             config.DETERMINISTIC_SEED)
            for i, s in enumerate(STEPS):
                if subset == "all_paired":  # must reproduce the published paired deltas exactly
                    pub = published.get((m, d, cue, ctl, int(s)))
                    if pub is not None:
                        assert int(pub["n_paired"]) == len(keep), (m, d, cue, ctl, s, pub["n_paired"], len(keep))
                        assert abs(float(pub["delta_pcr"]) - delta[i]) < 1e-9, (m, d, cue, ctl, s)
                rows.append({**row0, "step": int(s), "n_paired": len(keep), "control_pcr": float(cpcr[i]),
                             "cue_pcr": float(qpcr[i]), "delta_pcr": float(delta[i]),
                             "ci_lower": float(lo[i]), "ci_upper": float(hi[i])})
    return rows


def summary_md(rows: list[dict]) -> str:
    idx = {(r["model"], r["dataset"], r["cue_condition"], r["control_condition"], r["subset"], r["step"]): r
           for r in rows}
    fmt = lambda r: ("n/a (n=0)" if not r["n_paired"] else
                     f"{100 * r['delta_pcr']:+.1f} [{100 * r['ci_lower']:+.1f}, {100 * r['ci_upper']:+.1f}] "
                     f"(n={r['n_paired']})")
    excl = lambda r: r["n_paired"] and (r["ci_lower"] > 0 or r["ci_upper"] < 0)
    out = ["# E1 identical-d1 sensitivity: summary at each model's characteristic step", "",
           "ΔPCR in percentage points, cue − Neutral, 95% paired bootstrap CI. "
           "`identical_d1` = pairs whose pre-cue first search observation is byte-identical. "
           "Full curves for all 14 comparisons and steps 1–30: `identical_d1_pcr_deltas.csv`.", ""]
    agree = total = 0
    for cue, ctl, label in SUMMARY_COMPARISONS:
        out += [f"## {label} vs Neutral", "",
                "| Model | Dataset | Step | All paired | Identical d1 | Same sign | Identical-d1 CI excludes 0 |",
                "|---|---|---|---|---|---|---|"]
        for m in CHAR:
            for d in ("hotpot", "2wiki", "musique", "multihoprag"):
                a = idx.get((m, d, cue, ctl, "all_paired", CHAR[m]))
                s = idx.get((m, d, cue, ctl, "identical_d1", CHAR[m]))
                if a is None or s is None:
                    continue
                same = bool(s["n_paired"]) and np.sign(a["delta_pcr"]) == np.sign(s["delta_pcr"])
                total += 1
                agree += same
                out.append(f"| {m} | {d} | {CHAR[m]} | {fmt(a)} | {fmt(s)} | {'yes' if same else 'no'} | "
                           f"{'yes' if excl(s) else 'no'} |")
        out.append("")
    out.insert(4, f"**Same sign as the all-paired estimate in {agree}/{total} model × dataset × cue cells "
                  f"(both comparisons).**\n")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- 2. failures by condition
def failure_rows() -> tuple[list[dict], list[dict]]:
    inv = [r for b in BUNDLES for r in read(RESULTS_E1 / b / "data" / "run_inventory.csv")]
    by_cond = [{"model": r["model"], "dataset": r["dataset"], "condition": r["condition"],
                "n_completed": int(r["n_completed"]), "n_failures": int(r["n_failures"]),
                "failure_rate": int(r["n_failures"]) / int(r["n_completed"]) if int(r["n_completed"]) else None}
               for r in inv]
    tests = []
    groups = defaultdict(list)
    for r in by_cond:
        groups[(r["model"], r["dataset"])].append(r)
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (m, d), rs in sorted(groups.items()):
        for r in rs:
            pooled[m][r["condition"]][0] += r["n_failures"]
            pooled[m][r["condition"]][1] += r["n_completed"]
        tests.append(_chi_row(m, d, [(r["n_failures"], r["n_completed"]) for r in rs]))
    for m, conds in sorted(pooled.items()):
        tests.append(_chi_row(m, "pooled", list(conds.values())))
    return by_cond, tests


def _chi_row(model, dataset, counts):
    fails = sum(f for f, _ in counts)
    rates = [f / n for f, n in counts if n]
    row = {"model": model, "dataset": dataset, "n_conditions": len(counts), "n_failures": fails,
           "n_completed": sum(n for _, n in counts), "min_rate": min(rates), "max_rate": max(rates),
           "chi2": None, "dof": None, "p_value": None, "min_expected_count": None, "chi2_valid": None}
    if fails:
        chi, p, dof, expected = chi2_contingency([[f, n - f] for f, n in counts])
        # Cochran's rule: all expected counts >= 1 and at least 80% >= 5; otherwise p is unreliable.
        valid = expected.min() >= 1 and (expected >= 5).mean() >= 0.8
        row.update(chi2=float(chi), dof=int(dof), p_value=float(p), min_expected_count=float(expected.min()),
                   chi2_valid=bool(valid))
    return row


def main() -> int:
    rows = identical_d1_rows()
    write(OUT / "identical_d1_pcr_deltas.csv", rows)
    (OUT / "identical_d1_summary.md").write_text(summary_md(rows), encoding="utf-8")
    print(f"wrote {OUT / 'identical_d1_summary.md'}")
    by_cond, tests = failure_rows()
    write(OUT / "technical_failures_by_condition.csv", by_cond)
    write(OUT / "technical_failure_condition_tests.csv", tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
