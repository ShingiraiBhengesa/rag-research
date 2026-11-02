"""Figure 5.2: box plot of the paired step reduction D_q = t_Neutral - t_cue, Fixed Sufficiency vs Neutral,
one box per model x dataset.

Run from the repo root:  PYTHONPATH=src python notes/reviewer_checks/fig5_2_step_boxplot.py
Data: notes/reviewer_checks/v2/step_pairs_all.csv (pairs: same question, both runs PCR-eligible; non-completion
coded t = 30). Boxes = IQR with median; whiskers = 1.5 x IQR; outliers as faint dots. Statistics use every pair; the
y-axis is only clipped to [-6, +8] for legibility (points outside are counted in the printed summary / data CSV).
Output: results/figures/thesis_restructure_v2/fig5_2_step_boxplot.{pdf,png} (300 dpi) + fig5_2_step_boxplot_data.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_restructure_v2 as V  # noqa: E402
from thesis_restructure_v2 import plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TITLES = {"qwen14b": "Qwen2.5-14B", "llama8b": "Llama-3.1-8B", "gemma27b": "Gemma-3-27B", "gpt_oss_20b": "gpt-oss-20b"}
DATASETS = [("hotpot", "HotpotQA", "#2a78d6"), ("2wiki", "2WikiMultiHopQA", "#eb6834"),
            ("musique", "MuSiQue", "#1baf7a"), ("multihoprag", "MultiHop-RAG", "#6250d6")]
YLIM = (-6, 8)


def main():
    sp = pd.read_csv(V.OUTC / "step_pairs_all.csv")
    sp = sp[sp.contrast == "Fixed Sufficiency vs Neutral"].assign(reduction=lambda x: x.t_ref - x.t_cue)
    fig, ax = plt.subplots(figsize=(10, 5))
    V.style(ax, "y")
    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    rows = []
    for g, (m, _) in enumerate(V.MODELS):
        for j, (d, _, c) in enumerate(DATASETS):
            r = sp[(sp.model == m) & (sp.dataset == d)].reduction.to_numpy()
            pos = g * 5 + j
            ax.boxplot(r, positions=[pos], widths=0.7, whis=1.5, patch_artist=True, manage_ticks=False,
                       boxprops={"facecolor": to_rgba(c, 0.4), "edgecolor": c, "linewidth": 1.2},
                       medianprops={"color": c, "linewidth": 1.8}, whiskerprops={"color": c, "linewidth": 1.2},
                       capprops={"color": c, "linewidth": 1.2},
                       flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": c, "markeredgecolor": "none",
                                   "alpha": 0.3})
            q1, med, q3 = np.percentile(r, [25, 50, 75])
            rows.append({"model": m, "dataset": d, "n_pairs": len(r), "mean": r.mean(), "median": med, "q1": q1,
                         "q3": q3, "n_below_ylim": int((r < YLIM[0]).sum()), "n_above_ylim": int((r > YLIM[1]).sum())})
        if g:
            ax.axvline(g * 5 - 1, color="#cccccc", lw=0.8, ls="--")
    ax.set_xticks([g * 5 + 1.5 for g in range(4)], [TITLES[m] for m, _ in V.MODELS], fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.8, 18.8)
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Steps earlier  (D$_q$ = T$_{neutral}$ − T$_{cue}$)")
    fig.legend([Patch(facecolor=to_rgba(c, 0.4), edgecolor=c) for *_, c in DATASETS], [n for _, n, _ in DATASETS],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    stem = "fig5_2_step_boxplot"
    V.save_fig(fig, stem)
    out = pd.DataFrame(rows)
    out.to_csv(V.OUTF / f"{stem}_data.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
