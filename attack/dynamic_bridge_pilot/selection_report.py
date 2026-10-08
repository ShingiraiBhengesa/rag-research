"""Cue-selection report for the dynamic bridge-cue pilot (pilot spec section 5).

Reads the frozen qwen14b/llama8b canonical results bundle
(qwen_llama_results_bundle/data/) read-only and ranks the seven Natural attack
conditions by robustness of their premature-completion effect. No new runs.

Every candidate is compared against NaturalNeutral on the strict paired set
(both runs pcr_eligible), using the canonical bundle's own bootstrap
(analysis/canonical/results_bundle/paired_deltas._bootstrap_all_steps, same
seed and iteration count). Doing this uniformly matters because the bundle's
comparison manifest pairs the two source-attached ablations against their
parent cue, not against Neutral. The script cross-checks its own deltas against
the bundle's published pcr_paired_deltas_stepwise.csv wherever the bundle has a
Neutral-baseline comparison, and fails if they disagree.

Ranking (robustness first, magnitude second), per spec section 5:
  1. n_sig_cells: (model, dataset) cells, over all 4 datasets x 2 models, whose
     delta vs Neutral at the model's peak checkpoint has a 95% CI excluding 0
     on the positive side. Peak checkpoints are the ones documented in
     CUE_EFFECT_ANALYSIS.md section 6.1: qwen14b@1, llama8b@5.
  2. min_cell_delta_peak: the worst cell at the peak checkpoint (consistency).
  3. mean_delta_peak: average over the 8 cells (magnitude).
The Oracle conditions are excluded (different evidence regime; cannot be
instantiated over natural retrieval), as are controls and ContinueSearch.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import config as analysis_config
from analysis.canonical.results_bundle.paired_deltas import (
    STEPWISE_BOOTSTRAP_ITERATIONS,
    _bootstrap_all_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DATA = REPO_ROOT / "qwen_llama_results_bundle" / "data"
OUT_DIR = REPO_ROOT / "experiments" / "dynamic_bridge_cue_pilot" / "selection"

MODELS = ("qwen14b", "llama8b")
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")
PILOT_DATASETS = ("hotpot", "2wiki")
BASELINE = "NaturalNeutral"
CANDIDATES = (
    "Natural_epistemic_confirmation",
    "Natural_search_match_cue",
    "Natural_source_coverage",
    "Natural_epistemic_confirmation_source",
    "Natural_search_match_source",
    "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)
PEAK_STEP = {"qwen14b": 1, "llama8b": 5}
REPORT_STEPS = (1, 2, 3, 5, 10, 30)
NUMERATOR_ROUTES = ("RESEARCH_COMPLETE", "NO_TOOL_DIRECT_EXIT")
CROSSCHECK_TOL = 1e-9


def _load_run_level(path: Path) -> dict:
    """{(model, dataset, condition): {example_id: t_i or inf, None if ineligible}}"""
    out: dict = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["pcr_eligible"] != "True":
                t = None
            elif r["terminal_route"] in NUMERATOR_ROUTES and r["first_terminal_post_observation_step"]:
                t = float(r["first_terminal_post_observation_step"])
            else:
                t = np.inf
            out[(r["model"], r["dataset"], r["condition"])][r["example_id"]] = t
    return out


def _paired(runs: dict, model: str, dataset: str, cue: str, control: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = runs[(model, dataset, cue)], runs[(model, dataset, control)]
    ids = sorted(e for e in a.keys() & b.keys() if a[e] is not None and b[e] is not None)
    return np.array([a[e] for e in ids]), np.array([b[e] for e in ids])


def _abs_pcr(runs: dict, model: str, dataset: str, condition: str, steps: np.ndarray) -> np.ndarray:
    t = np.array([v for v in runs[(model, dataset, condition)].values() if v is not None])
    return (t[:, None] <= steps[None, :]).mean(axis=0)


def _keyed(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {tuple(r[k] for k in keys): r for r in rows}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(bundle: Path = BUNDLE_DATA) -> tuple[list[dict], list[dict], list[str]]:
    runs = _load_run_level(bundle / "pcr_run_level.csv")
    published = _keyed(_read_csv(bundle / "pcr_paired_deltas_stepwise.csv"),
                       ("model", "dataset", "control_condition", "cue_condition", "step"))
    evidence = _keyed(_read_csv(bundle / "evidence_summary.csv"),
                      ("model", "dataset", "control_condition", "cue_condition"))
    answers = _keyed(_read_csv(bundle / "answer_paired_flips.csv"),
                     ("model", "dataset", "condition_a", "condition_b"))

    all_steps = np.arange(1, 31)
    cell_rows: list[dict] = []
    crosschecks: list[str] = []
    for cond in CANDIDATES:
        for model in MODELS:
            for ds in DATASETS:
                cue_t, ctl_t = _paired(runs, model, ds, cond, BASELINE)
                ctl, cue, delta, lo, hi = _bootstrap_all_steps(
                    cue_t, ctl_t, all_steps, STEPWISE_BOOTSTRAP_ITERATIONS, analysis_config.DETERMINISTIC_SEED
                )
                abs_cue = _abs_pcr(runs, model, ds, cond, all_steps)
                pub = published.get((model, ds, BASELINE, cond, "1"))
                if pub is not None:
                    for s in REPORT_STEPS:
                        p = published[(model, ds, BASELINE, cond, str(s))]
                        if abs(float(p["delta_pcr"]) - delta[s - 1]) > CROSSCHECK_TOL or int(p["n_paired"]) != len(cue_t):
                            raise AssertionError(f"delta mismatch vs published bundle: {model}/{ds}/{cond}@{s}")
                    crosschecks.append(f"{model}/{ds}/{cond}")
                ev = evidence.get((model, ds, BASELINE, cond))
                ans = answers.get((model, ds, BASELINE, cond))
                row = {
                    "condition": cond, "model": model, "dataset": ds,
                    "pilot_dataset": ds in PILOT_DATASETS, "n_paired_vs_neutral": len(cue_t),
                    "peak_step": PEAK_STEP[model],
                }
                for s in REPORT_STEPS:
                    i = s - 1
                    row[f"pcr@{s}"] = round(100 * abs_cue[i], 2)
                    row[f"neutral_pcr@{s}"] = round(100 * ctl[i], 2)
                    row[f"delta@{s}"] = round(100 * delta[i], 2)
                    row[f"ci@{s}"] = f"[{100 * lo[i]:.1f}, {100 * hi[i]:.1f}]"
                    row[f"sig@{s}"] = bool(lo[i] > 0)
                row["cumulative_pcr@30"] = row["pcr@30"]
                pk = PEAK_STEP[model] - 1
                row["delta_peak"] = round(100 * delta[pk], 2)
                row["sig_peak"] = bool(lo[pk] > 0)
                row["evidence_loss_pp"] = round(100 * float(ev["paired_evidence_loss"]), 2) if ev else None
                row["answer_net_harm_pp"] = round(float(ans["net_harm_pct_points"]), 2) if ans else None
                cell_rows.append(row)

    summary: list[dict] = []
    for cond in CANDIDATES:
        cells = [r for r in cell_rows if r["condition"] == cond]
        pilot_cells = [r for r in cells if r["pilot_dataset"]]
        summary.append({
            "condition": cond,
            "display_name": analysis_config.CONDITION_DISPLAY_NAMES.get(cond, cond),
            "n_sig_cells_peak_all8": sum(r["sig_peak"] for r in cells),
            "n_sig_cells_peak_pilot4": sum(r["sig_peak"] for r in pilot_cells),
            "min_cell_delta_peak": min(r["delta_peak"] for r in cells),
            "mean_delta_peak": round(float(np.mean([r["delta_peak"] for r in cells])), 2),
            "qwen14b_mean_delta@1": round(float(np.mean([r["delta@1"] for r in cells if r["model"] == "qwen14b"])), 2),
            "qwen14b_mean_delta@2": round(float(np.mean([r["delta@2"] for r in cells if r["model"] == "qwen14b"])), 2),
            "llama8b_mean_delta@5": round(float(np.mean([r["delta@5"] for r in cells if r["model"] == "llama8b"])), 2),
            "llama8b_mean_delta@2": round(float(np.mean([r["delta@2"] for r in cells if r["model"] == "llama8b"])), 2),
            "mean_evidence_loss_pp": _mean_or_none(r["evidence_loss_pp"] for r in cells),
            "mean_answer_net_harm_pp": _mean_or_none(r["answer_net_harm_pp"] for r in cells),
        })
    summary.sort(key=lambda s: (-s["n_sig_cells_peak_all8"], -s["min_cell_delta_peak"], -s["mean_delta_peak"]))
    for rank, s in enumerate(summary, start=1):
        s["rank"] = rank
    return cell_rows, summary, crosschecks


def _mean_or_none(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _markdown(cell_rows: list[dict], summary: list[dict], crosschecks: list[str]) -> str:
    lines = [
        "# Cue-selection report (generated)",
        "",
        "Generated by `python -m attack.dynamic_bridge_pilot.selection_report` from the frozen",
        "`qwen_llama_results_bundle/data/` (read-only). All deltas: candidate minus NaturalNeutral,",
        "strict paired set, canonical bootstrap (2000 iters, seed `analysis.config.DETERMINISTIC_SEED`).",
        f"Self-check: {len(crosschecks)} (model, dataset, condition) cells reproduced the bundle's published",
        "deltas exactly; the remaining cells (source-attached ablations) have no published Neutral baseline.",
        "",
        "## Ranking (robustness first)",
        "",
        "| rank | condition | sig cells @peak (8) | sig cells @peak (pilot 4) | min cell Δ@peak | mean Δ@peak | qwen Δ@1 | llama Δ@5 | evid. loss | ans. harm |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['rank']} | `{s['condition']}` | {s['n_sig_cells_peak_all8']}/8 | {s['n_sig_cells_peak_pilot4']}/4 | "
            f"{s['min_cell_delta_peak']:+.1f} | {s['mean_delta_peak']:+.1f} | {s['qwen14b_mean_delta@1']:+.1f} | "
            f"{s['llama8b_mean_delta@5']:+.1f} | {s['mean_evidence_loss_pp']} | {s['mean_answer_net_harm_pp']} |"
        )
    lines += ["", "## Pilot cells (hotpot, 2wiki): absolute PCR and Δ vs Neutral", "",
              "| condition | model | dataset | PCR@1 | PCR@2 | PCR@5 | PCR@30 | Δ@1 [CI] | Δ@2 [CI] | Δ@5 [CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cell_rows:
        if not r["pilot_dataset"]:
            continue
        lines.append(
            f"| `{r['condition']}` | {r['model']} | {r['dataset']} | {r['pcr@1']} | {r['pcr@2']} | {r['pcr@5']} | "
            f"{r['pcr@30']} | {r['delta@1']:+.1f} {r['ci@1']} | {r['delta@2']:+.1f} {r['ci@2']} | {r['delta@5']:+.1f} {r['ci@5']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DATA)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    cell_rows, summary, crosschecks = build(args.bundle)
    _write_csv(args.out_dir / "selection_cells.csv", cell_rows)
    _write_csv(args.out_dir / "selection_ranking.csv", summary)
    (args.out_dir / "SELECTION_REPORT.md").write_text(_markdown(cell_rows, summary, crosschecks), encoding="utf-8")
    print(json.dumps({"top2": [s["condition"] for s in summary[:2]], "crosschecked_cells": len(crosschecks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
