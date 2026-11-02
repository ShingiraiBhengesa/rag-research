"""Figure 5.1: PCR@k curves, HotpotQA only, one panel per model; Neutral, Fixed Sufficiency, Suff.+Saturation and
Continue Search.

Run from the repo root:  PYTHONPATH=src python notes/reviewer_checks/fig5_1_pcr_curves.py
Data: run_level_evidence_hops.csv via thesis_restructure_v2.load_e1_all (same T and PCR-eligibility as
step_pairs_all.csv, which has no Continue Search). PCR_c(k) = % of the condition's PCR-eligible HotpotQA runs completed
by step k, k = 1-30 (non-completion is coded T = 30 but excluded: a run counts only if it completed). Every eligible run
of each condition is used (not matched pairs); n per condition is in the data CSV.
Output: results/figures/thesis_restructure_v2/fig5_1_pcr_curves.{pdf,png} (300 dpi) + fig5_1_pcr_curves_data.csv
Earlier versions: fig5_1_pcr_curves_grid.py (4x4), fig5_1_pcr_curves_overlay.py, fig5_1_pcr_curves_arms.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_restructure_v2 as V  # noqa: E402
from thesis_restructure_v2 import FIX, NEU, SAT, plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TITLES = {"qwen14b": "Qwen-14B", "llama8b": "Llama-8B", "gemma27b": "Gemma-27B", "gpt_oss_20b": "gpt-oss-20b"}
ARMS = [(NEU, "Neutral", "#2a78d6", "-"), (FIX, "Fixed Sufficiency", "#eb6834", "-"),
        (SAT, "Suff.+Saturation", "#eb6834", "--"), ("NaturalContinueSearch", "Continue Search", "#2ecc71", "--")]
K = list(range(1, 31))


def main():
    ev = V.load_e1_all()
    ev = ev[(ev.dataset == "hotpot") & ev.pe]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, (m, _) in zip(axes.flat, V.MODELS):
        V.style(ax, "both")
        for a, lab, c, ls in ARMS:
            x = ev[(ev.model == m) & (ev.condition == a)]
            pcr = [100 * (~x.nc & (x.t <= k)).mean() for k in K]
            ax.plot(K, pcr, color=c, ls=ls, lw=2)
            rows += [{"model": m, "dataset": "hotpot", "condition": lab, "k": k, "n_runs": len(x), "pcr": v}
                     for k, v in zip(K, pcr)]
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 30], ["1", "2", "3", "5", "10", "20", "30"])
        ax.minorticks_off()
        ax.set_xlim(0.92, 32)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.set_title(TITLES[m], fontsize=12)
        if m in ("gemma27b", "gpt_oss_20b"):
            ax.set_xlabel("Researcher decision after the cue (log scale)")
        if m in ("qwen14b", "gemma27b"):
            ax.set_ylabel("PCR@k")
    fig.legend([Line2D([0], [0], color=c, ls=ls, lw=2.4) for _, _, c, ls in ARMS], [lab for _, lab, *_ in ARMS],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=3, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem = "fig5_1_pcr_curves"
    V.save_fig(fig, stem)
    out = pd.DataFrame(rows)
    out.to_csv(V.OUTF / f"{stem}_data.csv", index=False)
    print(out.groupby(["model", "condition"]).n_runs.first().unstack().to_string())
    print(f"wrote {V.OUTF / stem}_data.csv")


if __name__ == "__main__":
    main()
