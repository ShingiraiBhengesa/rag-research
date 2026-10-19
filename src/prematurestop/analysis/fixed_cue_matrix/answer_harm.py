"""Answer-harm analysis v2: does a cue make final answers worse, and where?

Specification (written before this code ran):
`results/whitebox/e1_fixed_cue_matrix/answer_harm/SPEC.md`.

Pure table joins, no trace re-parsing. For every pre-registered comparison and
every model x dataset it pairs the control and cue runs of the same question
and counts b (control correct, cue wrong) and c (control wrong, cue correct).

* Net harm = (b - c) / n, with an exact McNemar test and a paired bootstrap CI.
* Pooled per model and overall, stratified by model x dataset.
* Mechanism: net harm within classes of the paired evidence transition at the
  final-report input (both complete / evidence lost / evidence gained / both
  incomplete), and secondarily at the Researcher's local stop.
* Strata: evidence already complete in d1 (control run), and hop count.
* Evidence shift (post-hoc): lost vs gained evidence pairs, exact sign test.
* Sensitivity: drop adjudication-flagged pairs; drop the three gpt-oss
  datasets served without the lenient Harmony parser.
* Benjamini-Hochberg q-values within each comparison family.

    python -m prematurestop.analysis.fixed_cue_matrix.answer_harm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import COMPARISONS
from prematurestop.evaluation import config
from prematurestop.paths import LOCAL_DATA, RESULTS_E1

OUT_DIR = RESULTS_E1 / "answer_harm"
EVIDENCE_FILES = (
    RESULTS_E1 / "evidence_hop_coverage" / "run_level_evidence_hops.csv",
    RESULTS_E1 / "evidence_hop_coverage_gemma_gptoss" / "run_level_evidence_hops.csv",
)
QWEN_LLAMA_ANSWERS = (LOCAL_DATA / "evaluation" / "v3_final_completed_qwen_llama_20260910" / "answers"
                      / "final_answer_predictions.jsonl")
GEMMA_GPTOSS_ANSWERS = RESULTS_E1 / "gemma_gptoss" / "data" / "answer_run_level.csv"

SEED = 20260910
N_BOOT = 2000
MODELS = ("qwen14b", "llama8b", "gemma27b", "gpt_oss_20b")
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")

PRIMARY = {"Natural_PreRegisteredSufficiency", "Natural_PreRegisteredSufficiencySaturation"}
NEGATIVE_CONTROLS = {"NaturalNeutral", "OracleComplete_PreRegisteredSufficiencySaturation", "NaturalContinueSearch"}
GPT_OSS_WITHOUT_LENIENT_PARSER = {("gpt_oss_20b", d) for d in ("hotpot", "musique", "multihoprag")}


def family(comp) -> str:
    if comp.control_condition == "NaturalNeutral" and comp.cue_condition in PRIMARY:
        return "primary_attack"
    if comp.cue_condition in NEGATIVE_CONTROLS:
        return "control"
    return "secondary_attack"


# --------------------------------------------------------------------------- loading
@dataclass(frozen=True)
class Run:
    correct: bool
    flagged: bool              # needs_adjudication
    final_full: bool | None    # full evidence at final-report input
    local_full: bool | None    # full evidence at the Researcher's stop
    anchor_full: bool | None   # full evidence already in d1
    hop: str


def _b(v: str) -> bool | None:
    return None if v in ("", "None") else v == "True"


def load_runs() -> dict[tuple[str, str, str, str], Run]:
    """(model, dataset, condition, example_id) -> Run, for runs with both an answer and evidence."""
    answers: dict[tuple, tuple[bool, bool]] = {}
    with QWEN_LLAMA_ANSWERS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            answers[(r["model"], r["dataset"], r["condition"], str(r["example_id"]))] = (
                r["semantic_label"] == "correct", r["semantic_label"] == "needs_adjudication")
    with GEMMA_GPTOSS_ANSWERS.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            answers[(r["model"], r["dataset"], r["condition"], r["example_id"])] = (
                r["automatic_semantic_label"] == "correct", r["needs_adjudication_flag"] == "True")
    runs = {}
    for path in EVIDENCE_FILES:
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["model"], r["dataset"], r["condition"], r["example_id"])
                if key not in answers or r["valid_run"] != "True":
                    continue
                correct, flagged = answers[key]
                runs[key] = Run(correct, flagged, _b(r["final_full_evidence"]), _b(r["local_full_evidence"]),
                                _b(r["anchor_full_evidence"]), r["dataset_stratum_value"])
    return runs


# --------------------------------------------------------------------------- statistics
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of b vs c at p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # log-space binomial tail for large n
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2 * tail)


def net_harm(pairs: list[tuple[bool, bool]]) -> tuple[int, int, int, float | None]:
    """pairs of (control_correct, cue_correct) -> n, b, c, net harm (fraction)."""
    n = len(pairs)
    b = sum(1 for ctl, cue in pairs if ctl and not cue)
    c = sum(1 for ctl, cue in pairs if cue and not ctl)
    return n, b, c, ((b - c) / n if n else None)


def bootstrap_ci(strata: list[list[tuple[bool, bool]]], seed: int = SEED, n_boot: int = N_BOOT):
    """95% CI of pooled net harm, resampling pairs within each stratum.

    Each pair is harm (+1), help (-1) or no change (0), so resampling a stratum
    of n pairs is exactly a multinomial draw of n over those three outcomes.
    """
    import numpy as np

    strata = [s for s in strata if s]
    total = sum(len(s) for s in strata)
    if not total:
        return None, None
    rng = np.random.default_rng(seed)
    acc = np.zeros(n_boot)
    for s in strata:
        n = len(s)
        harm = sum(1 for ctl, cue in s if ctl and not cue)
        help_ = sum(1 for ctl, cue in s if cue and not ctl)
        draws = rng.multinomial(n, [harm / n, help_ / n, (n - harm - help_) / n], size=n_boot)
        acc += draws[:, 0] - draws[:, 1]
    stats = np.sort(acc / total)
    return float(stats[int(0.025 * n_boot)]), float(stats[int(0.975 * n_boot) - 1])


def bh_qvalues(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


# --------------------------------------------------------------------------- pairing
def index_runs(runs) -> dict[tuple[str, str, str], dict[str, Run]]:
    """(model, dataset, condition) -> {example_id: Run}."""
    idx: dict[tuple[str, str, str], dict[str, Run]] = defaultdict(dict)
    for (m, d, cond, ex), r in runs.items():
        idx[(m, d, cond)][ex] = r
    return idx


def make_pairs(idx, comp, model, dataset):
    """Pairs (control Run, cue Run) for the same question, sorted by example_id."""
    ctl = idx.get((model, dataset, comp.control_condition), {})
    cue = idx.get((model, dataset, comp.cue_condition), {})
    return [(ctl[ex], cue[ex]) for ex in sorted(ctl.keys() & cue.keys())]


def transition(ctl_full: bool | None, cue_full: bool | None) -> str | None:
    if ctl_full is None or cue_full is None:
        return None
    return {(True, True): "both_complete", (True, False): "evidence_lost",
            (False, True): "evidence_gained", (False, False): "both_incomplete"}[(ctl_full, cue_full)]


def _row(scope: dict, strata: list[list[tuple[bool, bool]]], ci: bool = True) -> dict:
    flat = [p for s in strata for p in s]
    n, b, c, nh = net_harm(flat)
    lo, hi = bootstrap_ci(strata) if ci else (None, None)
    pct = lambda x: None if x is None else round(100 * x, 3)
    return {**scope, "n_pairs": n, "ctl_correct_cue_wrong": b, "ctl_wrong_cue_correct": c,
            "net_harm_pp": pct(nh), "ci_low_pp": pct(lo), "ci_high_pp": pct(hi),
            "mcnemar_p": round(mcnemar_exact(b, c), 6)}


# --------------------------------------------------------------------------- main analysis
def analyse(runs) -> dict[str, list[dict]]:
    by_runs = lambda ps: [(ctl.correct, cue.correct) for ctl, cue in ps]
    idx = index_runs(runs)
    cells, pooled, mech, strata_rows, sens, shift = [], [], [], [], [], []

    for comp in COMPARISONS:
        fam = family(comp)
        base = {"family": fam, "comparison": comp.comparison_name,
                "control": comp.control_condition, "cue": comp.cue_condition}
        per_cell = {(m, d): make_pairs(idx, comp, m, d) for m in MODELS for d in DATASETS}

        for (m, d), ps in per_cell.items():
            cells.append(_row({**base, "model": m, "dataset": d}, [by_runs(ps)]))
        for scope_model in MODELS + ("all",):
            keys = [k for k in per_cell if scope_model in ("all", k[0])]
            pooled.append(_row({**base, "model": scope_model}, [by_runs(per_cell[k]) for k in keys]))

        if fam == "control" and comp.cue_condition == "NaturalContinueSearch":
            continue
        # mechanism: paired evidence transition, final (primary) and local (secondary) boundary
        for boundary, attr in (("final_report_input", "final_full"), ("local_stop", "local_full")):
            for scope_model in MODELS + ("all",):
                keys = [k for k in per_cell if scope_model in ("all", k[0])]
                for cls in ("both_complete", "evidence_lost", "evidence_gained", "both_incomplete"):
                    strata = [by_runs([p for p in per_cell[k]
                                       if transition(getattr(p[0], attr), getattr(p[1], attr)) == cls]) for k in keys]
                    mech.append(_row({**base, "boundary": boundary, "model": scope_model, "transition": cls},
                                     strata, ci=fam == "primary_attack"))
        # Post-hoc (added after the first run, see SPEC.md): does the cue shift the
        # balance of evidence lost vs gained beyond run-to-run noise? Exact sign test.
        for boundary, attr in (("final_report_input", "final_full"), ("local_stop", "local_full")):
            for scope_model in MODELS + ("all",):
                keys = [k for k in per_cell if scope_model in ("all", k[0])]
                tr = [transition(getattr(c, attr), getattr(q, attr)) for k in keys for c, q in per_cell[k]]
                lost, gained = tr.count("evidence_lost"), tr.count("evidence_gained")
                shift.append({**base, "boundary": boundary, "model": scope_model, "n_pairs": len(tr),
                              "n_evidence_lost": lost, "n_evidence_gained": gained,
                              "net_lost": lost - gained,
                              "net_lost_pct_of_pairs": round(100 * (lost - gained) / len(tr), 3) if tr else None,
                              "lost_share": round(lost / (lost + gained), 4) if lost + gained else None,
                              "sign_test_p": round(mcnemar_exact(lost, gained), 8)})
        # strata: d1 already complete (control run) and hop count, pooled over all models
        for name, fn in (("d1_already_complete", lambda ctl: str(ctl.anchor_full)),
                         ("hop_count", lambda ctl: ctl.hop)):
            values = sorted({fn(p[0]) for ps in per_cell.values() for p in ps})
            for v in values:
                strata = [by_runs([p for p in ps if fn(p[0]) == v]) for ps in per_cell.values()]
                strata_rows.append(_row({**base, "stratum": name, "value": v, "model": "all"}, strata,
                                        ci=fam == "primary_attack"))

        if fam == "primary_attack":
            for label, keep_pair, drop_cells in (
                    ("S1_drop_adjudication_flagged", lambda p: not (p[0].flagged or p[1].flagged), set()),
                    ("S2_drop_gptoss_without_lenient_parser", lambda p: True, GPT_OSS_WITHOUT_LENIENT_PARSER)):
                for scope_model in MODELS + ("all",):
                    keys = [k for k in per_cell if scope_model in ("all", k[0]) and k not in drop_cells]
                    strata = [by_runs([p for p in per_cell[k] if keep_pair(p)]) for k in keys]
                    sens.append(_row({**base, "sensitivity": label, "model": scope_model}, strata))

    # Benjamini-Hochberg within each family over the model x dataset cells
    for fam in {r["family"] for r in cells}:
        rows = [r for r in cells if r["family"] == fam and r["n_pairs"]]
        for r, q in zip(rows, bh_qvalues([r["mcnemar_p"] for r in rows])):
            r["bh_q"] = round(q, 6)
    return {"cells": cells, "pooled": pooled, "mechanism": mech, "strata": strata_rows, "sensitivity": sens,
            "evidence_shift": shift}


def write_csv(rows: list[dict], path: Path) -> None:
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- figures
SERIES = {"Natural_PreRegisteredSufficiency": ("Fixed Sufficiency", "#2a78d6"),
          "Natural_PreRegisteredSufficiencySaturation": ("Sufficiency + Saturation", "#eb6834")}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.axvline(0, color=INK_2, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def write_figures(tables: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    f = lambda x: float(x)

    # 1. Net harm per model (pooled over datasets), primary cues, with 95% CI
    rows = [r for r in tables["pooled"] if r["family"] == "primary_attack"]
    labels = [config.DISPLAY_MODEL_NAMES.get(m, m) for m in MODELS] + ["All models"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    _style(ax)
    for j, (cue, (name, color)) in enumerate(SERIES.items()):
        for i, m in enumerate(MODELS + ("all",)):
            r = next(x for x in rows if x["cue"] == cue and x["model"] == m)
            y = i + (j - 0.5) * 0.28
            ax.plot([f(r["ci_low_pp"]), f(r["ci_high_pp"])], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(f(r["net_harm_pp"]), y, "o", ms=8, color=color, mec=SURFACE, mew=2,
                    label=name if i == 0 else None)
    ax.set_yticks(range(len(labels)), labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Net answer harm vs Neutral (pp), 95% CI", color=INK_2, fontsize=9)
    ax.set_title("Average answer harm is indistinguishable from zero", color=INK, fontsize=11, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"net_harm_by_model.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # 2. Net harm by paired evidence transition at the final-report input, all models
    classes = ("both_complete", "evidence_lost", "evidence_gained", "both_incomplete")
    names = {"both_complete": "Both complete", "evidence_lost": "Evidence lost by cue",
             "evidence_gained": "Evidence gained by cue", "both_incomplete": "Both incomplete"}
    mech = [r for r in tables["mechanism"] if r["family"] == "primary_attack"
            and r["boundary"] == "final_report_input" and r["model"] == "all"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax)
    for j, (cue, (name, color)) in enumerate(SERIES.items()):
        for i, cls in enumerate(classes):
            r = next(x for x in mech if x["cue"] == cue and x["transition"] == cls)
            y = i + (j - 0.5) * 0.28
            ax.plot([f(r["ci_low_pp"]), f(r["ci_high_pp"])], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(f(r["net_harm_pp"]), y, "o", ms=8, color=color, mec=SURFACE, mew=2,
                    label=name if i == 0 else None)
    n_by_class = {cls: next(x for x in mech if x["cue"] == "Natural_PreRegisteredSufficiencySaturation"
                            and x["transition"] == cls)["n_pairs"] for cls in classes}
    ax.set_yticks(range(len(classes)), [f"{names[c]}  (n={n_by_class[c]})" for c in classes], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Net answer harm vs Neutral (pp), 95% CI, all models pooled", color=INK_2, fontsize=9)
    ax.set_title("Answers follow evidence: losing it hurts, gaining it helps", color=INK, fontsize=11, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"net_harm_by_evidence_transition.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print(f"{len(runs)} runs with an answer and evidence")
    tables = analyse(runs)
    for name, rows in tables.items():
        write_csv(rows, args.out_dir / f"{name}.csv")
        print(f"wrote {name}.csv ({len(rows)} rows)")
    write_figures(tables, args.out_dir)
    print("wrote figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
