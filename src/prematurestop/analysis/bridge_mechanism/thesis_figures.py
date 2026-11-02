"""Print-ready E2 thesis figures: PCR curves by arm, EICR by arm, contrast forest plot.

Every plotted value is read from the E2 v2b full-run outputs; nothing is typed in
by hand. Comparisons between arms use matched questions only (questions where
every plotted arm ran and was PCR-eligible).

    python -m prematurestop.analysis.bridge_mechanism.thesis_figures [--run-dir DIR]

Data sources (under the run directory, default local_data/e2_bridge_mechanism/run_results/full):
  e2_pcr_curves        trajectories/run_labels.csv              (t_i per run, matched questions)
  e2_eicr              trajectories/run_labels.csv
                       + diagnostics/evidence_coverage.csv     (coverage at the Researcher's stop)
  e2_contrast_forest   pcr/pcr_paired_contrasts.csv            (all_paired question set)

Output: results/figures/e2_thesis_figures/e2_*.{pdf,png} (gitignored scratch). These are earlier
iterations; the E2 figures printed in the thesis (5.3, 5.4) come from scripts/thesis_figures/.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from prematurestop.analysis.fixed_cue_matrix import thesis_figures as e1  # shared style helpers
from prematurestop.paths import LOCAL_DATA, RESULTS

RUN_DIR = LOCAL_DATA / "e2_bridge_mechanism" / "run_results" / "full"
OUT = RESULTS / "figures" / "e2_thesis_figures"  # gitignored scratch; final copies live in results/thesis_figures/
MODELS = (("qwen14b", "Qwen"), ("llama8b", "Llama"))
DATASETS = (("hotpot", "HotpotQA"), ("2wiki", "2Wiki"))
CHAR = {"qwen14b": 1, "llama8b": 3}  # E1 characteristic steps
INK, INK_2, INK_3, GRID, REF = e1.INK, e1.INK_2, e1.INK_3, e1.GRID, e1.REF
NEUTRAL_C, CORAL, TEAL, AMBER, PURPLE = "#2c2c2a", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
ARMS = (("neutral", "Neutral", NEUTRAL_C, "-"), ("fixed_sufficiency", "Fixed Sufficiency", CORAL, "-"),
        ("bridge_sufficiency_k3", "Bridge K3", TEAL, "-"), ("entity_sufficiency", "Entity-only", AMBER, (0, (6, 3))),
        ("unsupported_sufficiency", "Unsupported", PURPLE, (0, (6, 3))))
CONTRASTS = (("structure", "Structure\n(Bridge K3 vs Entity)"), ("truth", "Truth\n(Bridge K3 vs Unsupported)"),
             ("selection", "Selection\n(Bridge K3 vs K1)"),
             ("bridge_k3_vs_fixed_sufficiency", "Bridge K3 vs\nFixed Sufficiency"),
             ("bridge_sat_k3_vs_fixed_saturation", "Bridge Sat. vs\nFixed Sat."))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / stem}.png")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_runs(run_dir: Path) -> dict:
    """(victim, dataset, example_id, arm) -> {eligible, t_i, local} for runs that ran."""
    cov = {(r["victim"], r["dataset"], r["example_id"], r["arm"]): r
           for r in e1.read(run_dir / "diagnostics" / "evidence_coverage.csv")}
    num = lambda x: None if x in ("", "None", None) else float(x)
    runs = {}
    for r in e1.read(run_dir / "trajectories" / "run_labels.csv"):
        if r["status"] != "ran":
            continue
        k = (r["victim"], r["dataset"], r["example_id"], r["arm"])
        runs[k] = {"eligible": r["pcr_eligible"] == "True", "t_i": num(r["t_i"]),
                   "local": num(cov.get(k, {}).get("episode0_terminal_coverage"))}
    return runs


def matched(runs, v, d, arms, need):
    """Questions where every arm in `arms` ran and satisfies `need`."""
    by_q = defaultdict(dict)
    for (vv, dd, ex, a), x in runs.items():
        if vv == v and dd == d and a in arms and need(x):
            by_q[ex][a] = x
    return [q for q in by_q.values() if len(q) == len(arms)]


# --------------------------------------------------------------------------- E2.1 PCR curves
def fig_pcr(runs):
    arms = [a for a, *_ in ARMS]
    steps = list(range(1, 31))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ns = {}
    for i, (v, vname) in enumerate(MODELS):
        for j, (d, dname) in enumerate(DATASETS):
            ax = axes[i][j]
            qs = matched(runs, v, d, arms, need=lambda x: x["eligible"])
            n = ns[(v, d)] = len(qs)
            e1.grid(ax, "both")
            for a, _, color, ls in ARMS:
                ys = [100 * sum(1 for q in qs if q[a]["t_i"] is not None and q[a]["t_i"] <= k) / n for k in steps]
                fx, fy = e1.smooth(steps, ys, log=True)
                ax.plot(fx, fy, color=color, lw=2, ls=ls)
                ax.plot(steps[:5], ys[:5], "o", color=color, ms=3.5, zorder=3)
            ax.axvline(CHAR[v], color=REF, lw=1.2, ls=":")
            ax.text(CHAR[v] * 1.04, 3, f"step {CHAR[v]}", color=REF, fontsize=9)
            ax.set_title(f"{vname} — {dname}  (n={n})", fontsize=12, color=INK)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 3, 5, 10, 30], ["1", "2", "3", "5", "10", "30"])
            ax.minorticks_off()
            ax.set_xlim(0.92, 32)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.tick_params(labelsize=10, colors=INK_3)
            ax.set_xlabel("Researcher decision after the cue (log scale)", fontsize=11, color=INK_2)
            ax.set_ylabel("PCR@k", fontsize=11, color=INK_2)
    handles = [Line2D([0], [0], color=c, lw=3, ls=ls) for _, _, c, ls in ARMS]
    fig.legend(handles, [lab for _, lab, *_ in ARMS], loc="lower center", ncol=5, frameon=False, fontsize=12,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.045), handlelength=2.6)
    fig.text(0.5, 0.0, "Matched questions only (all five arms ran and were PCR-eligible; the Unsupported cue exists "
             "for fewer questions, so n is smaller than in the paired contrasts).\nDotted line: characteristic step "
             "from E1 (Qwen 1, Llama 3).",
             ha="center", fontsize=10, color=INK_2, style="italic")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "e2_pcr_curves")
    return ns


# --------------------------------------------------------------------------- E2.2 EICR
def fig_eicr(runs):
    arms = (("neutral", "Neutral", NEUTRAL_C), ("fixed_sufficiency", "Fixed Sufficiency", CORAL),
            ("bridge_sufficiency_k3", "Bridge K3", TEAL))
    groups = []
    for v, vname in MODELS:
        for d, dname in DATASETS:
            qs = matched(runs, v, d, [a for a, *_ in arms],
                         need=lambda x: x["eligible"] and x["t_i"] is not None and x["local"] is not None)
            k = {a: sum(1 for q in qs if q[a]["local"] < 1 - 1e-9) for a, *_ in arms}
            groups.append((f"{vname} — {dname}\n(n={len(qs)})", k, len(qs)))
    fig, ax = plt.subplots(figsize=(8, 5))
    e1.grid(ax)
    w = 0.26
    top = 0.0
    for g, (_, k, n) in enumerate(groups):
        for s, (a, _, color) in enumerate(arms):
            p = 100 * k[a] / n
            lo, hi = wilson(k[a], n)
            top = max(top, 100 * hi)
            x = g + (s - 1) * (w + 0.02)
            ax.bar(x, p, width=w, color=color, edgecolor="white")
            ax.errorbar(x, p, yerr=[[p - 100 * lo], [100 * hi - p]], color=INK, lw=1.2, capsize=4)
            ax.text(x, 100 * hi + 1.2, f"{p:.0f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, max(80, math.ceil((top + 8) / 10) * 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_ylabel("Evidence-incomplete completion rate", fontsize=11, color=INK_2)
    ax.set_xticks(range(len(groups)), [g for g, *_ in groups], fontsize=11, color=INK)
    ax.tick_params(axis="y", labelsize=10, colors=INK_3)
    ax.legend([Patch(color=c) for *_, c in arms], [lab for _, lab, _ in arms], loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=12, labelcolor=INK_2)
    ax.text(0.5, -0.29, "Matched questions (all three arms completed, PCR-eligible). Evidence incomplete = "
            "required evidence missing at the Researcher's stop. Error bars: 95% Wilson CI.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK_2, style="italic", wrap=True)
    save(fig, "e2_eicr")
    return {g: {a: round(100 * k[a] / n, 1) for a, *_ in arms} for g, k, n in groups}


# --------------------------------------------------------------------------- E2.3 forest plot
def fig_forest(run_dir: Path):
    pc = {(r["victim"], r["dataset"], r["contrast"]): r for r in e1.read(run_dir / "pcr" / "pcr_paired_contrasts.csv")
          if r["question_set"] == "all_paired" and int(r["step"]) == CHAR[r["victim"]]}
    colors = {"qwen14b": TEAL, "llama8b": CORAL}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    xs, labels = [], []
    for ax, (d, dname) in zip(axes, DATASETS):
        e1.grid(ax, "x")
        for row, (c, _) in enumerate(CONTRASTS):
            for k, (v, _) in enumerate(MODELS):
                r = pc[(v, d, c)]
                est, lo, hi = (100 * float(r[f]) for f in ("delta_pcr", "ci_lower", "ci_upper"))
                xs += [lo, hi]
                sig = lo > 0 or hi < 0
                y = row + (k - 0.5) * 0.3
                alpha = 1.0 if sig else 0.35
                ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", ms=8, color=colors[v], alpha=alpha,
                            lw=1.5, capsize=5, mfc=colors[v] if sig else "white", mew=1.5)
                labels.append((ax, y, f"{est:+.1f}", alpha))
        ax.axvline(0, color=REF, lw=1.6, ls=(0, (5, 3)))
        ax.axhline(2.5, color=INK_3, lw=0.8)
        ax.set_title(dname, fontsize=13, color=INK)
        ax.set_xlabel("ΔPCR at characteristic step (pp)", fontsize=11, color=INK_2)
        ax.tick_params(labelsize=10, colors=INK_3)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    lim = max(abs(min(xs)), abs(max(xs))) + 5
    for ax in axes:
        ax.set_xlim(-lim, lim + 6)  # right margin holds the value column
    for ax, y, text, alpha in labels:
        ax.text(lim + 5.5, y, text, va="center", ha="right", fontsize=9, color=INK_2, alpha=max(alpha, 0.7))
    axes[0].set_yticks(range(len(CONTRASTS)), [lab for _, lab in CONTRASTS], fontsize=11, color=INK)
    axes[0].set_ylim(len(CONTRASTS) - 0.5, -0.5)  # Structure at top
    axes[0].text(-lim + 0.5, 2.42, "pre-specified ↑   secondary ↓", fontsize=8.5, color=INK_3, va="bottom")
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=colors[v], mfc=mfc, alpha=a, mew=1.5)
               for v, _ in MODELS for mfc, a in ((colors[v], 1.0), ("white", 0.35))]
    labels = ["Qwen @ step 1, CI excludes 0", "Qwen, CI includes 0",
              "Llama @ step 3, CI excludes 0", "Llama, CI includes 0"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.07, "Positive = first-named arm stops more by the characteristic step. Matched pairs; "
             "95% bootstrap CIs.", ha="center", fontsize=9.5, color=INK_2, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "e2_contrast_forest")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = ap.parse_args(argv)
    runs = load_runs(args.run_dir)
    print("PCR matched n:", fig_pcr(runs))
    print("EICR %:", fig_eicr(runs))
    fig_forest(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
