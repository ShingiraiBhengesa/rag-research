"""Figure 5.3 (bridge-anchored study): PCR curves per cell for Neutral, Fixed Sufficiency and Bridge K3.

Run from the repo root after thesis_restructure_e2.py:  PYTHONPATH=src python notes/reviewer_checks/fig_e2_pcr_cdf.py
Data: notes/reviewer_checks/e2/e2_pcr_curves.csv. PCR_c(k) = % of the arm's valid runs (anchor-matched, PCR-eligible)
completed by step k; every valid run of each arm is used (not matched pairs), so Bridge K3 covers only questions with
a usable bridge fact. n per arm is in e2_pcr_curves.csv (not drawn).
Output: results/figures/thesis_restructure_e2/fig_e2_pcr_cdf.{pdf,png} (300 dpi, PDF fonttype 42), plus one file per
panel, fig_e2_pcr_cdf_<model>_<dataset>.{pdf,png} (5 x 3.8 in, own legend), for use as separate subfigures.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_restructure_e2 as E  # noqa: E402  (shared style, pdf.fonttype 42)
from thesis_restructure_e2 import plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ARMS = [("neutral", "Neutral", "Neutral", "#2a78d6"), ("fixed_sufficiency", "Fixed Sufficiency", "Fixed", "#eb6834"),
        ("bridge_sufficiency_k3", "Bridge K3", "Bridge K3", "#6250d6")]


def panel(ax, pc, v, d, xlabel=True, ylabel=True):
    E.V.style(ax, "y")
    for a, _, short, c in ARMS:  # n per arm is in the data CSV, not drawn on the panel
        x = pc[(pc.model == v) & (pc.dataset == d) & (pc.arm == a)]
        ax.plot(x.k, 100 * x.pcr, color=c, lw=2, marker="o", ms=2.3)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 29], ["1", "2", "3", "5", "10", "20", "29"])
    ax.minorticks_off()
    ax.set_xlim(0.92, 31)
    ax.set_ylim(0, 100)
    ax.set_title(f"{E.VNAME[v]} — {'HotpotQA' if d == 'hotpot' else '2WikiMultiHopQA'}", fontsize=11)
    if xlabel:
        ax.set_xlabel("Decision step after the cue, k (log scale)")
    if ylabel:
        ax.set_ylabel("Cumulative completion rate (%)")


def legend(fig, ncol=3, fontsize=None):
    h = [Line2D([0], [0], color=c, lw=2.2) for *_, c in ARMS]
    fig.legend(h, [lab for _, lab, _, _ in ARMS], loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=3, fontsize=fontsize)


def main():
    pc = pd.read_csv(E.OUTC / "e2_pcr_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, v in enumerate(("qwen14b", "llama8b")):
        for j, d in enumerate(("hotpot", "2wiki")):
            panel(axes[i, j], pc, v, d, xlabel=i == 1, ylabel=j == 0)
    legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    E.save_fig(fig, "fig_e2_pcr_cdf")
    for v in ("qwen14b", "llama8b"):
        for d in ("hotpot", "2wiki"):
            fig, ax = plt.subplots(figsize=(5, 3.8))
            panel(ax, pc, v, d)
            legend(fig, ncol=3, fontsize=8.5)
            fig.tight_layout(rect=(0, 0.12, 1, 1))
            E.save_fig(fig, f"fig_e2_pcr_cdf_{v}_{d}")


if __name__ == "__main__":
    main()
