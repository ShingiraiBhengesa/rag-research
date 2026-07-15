"""
plot_multistep_pcr.py — PCR@1–10 figures for Wrong Gap Assessment (reads live JSONL).

One figure per dataset (4 total), each with three side-by-side subplots, one per model.
Three lines per subplot: Clean baseline (dashed gray), Epistemic Confirmation (blue),
Search-Match Cue (red). End-of-line labels at step 10; step-1 attack labels when
conditions diverge from zero.

Palette (reference palette.md, light mode):
  Clean baseline : #898781  muted gray  — reference line, labeled directly
  Epistemic Conf : #2a78d6  blue slot 1 — CVD-validated categorical
  Search-Match   : #e34948  red  slot 6 — CVD-validated categorical

Reads from:
  ../whitebox_results/s1_attack2e_pcr10_{qwen7b,qwen14b,llama8b}.jsonl

Output (per dataset):
  ../whitebox_results/figures/<dataset>_multistep_pcr_by_model.png  (300 DPI)

Regenerate:
  cd whitebox_attack/
  conda run -n whitebox python plot_multistep_pcr.py
"""

import os
import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Publication-quality rcParams ───────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "xtick.labelsize":    10.5,
    "ytick.labelsize":    10.5,
    "legend.fontsize":    11,
    "axes.linewidth":     0.8,
    "xtick.major.size":   0,      # no x tick marks (step numbers are clear enough)
    "ytick.major.size":   3.5,
    "ytick.major.width":  0.8,
    "ytick.minor.size":   0,
    "axes.axisbelow":     True,
    "grid.linewidth":     0.7,
    "grid.color":         "#e1e0d9",   # reference palette hairline
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        300,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "whitebox_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_SOURCES = {
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen7b.jsonl"),
    "Qwen2.5-14B":  os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen14b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_llama8b.jsonl"),
    "GPT-OSS-120B": os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_gptoss120b.jsonl"),
    "Gemma-3-27B":  os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_gemma27b.jsonl"),
}

DATASET_MAP = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

MODELS   = list(JSONL_SOURCES.keys())
DATASETS = list(DATASET_MAP.values())
STEPS    = list(range(1, 11))

DATASET_FILES = {
    "HotpotQA":        "hotpotqa_multistep_pcr_by_model",
    "2WikiMultiHopQA": "2wikimultihopqa_multistep_pcr_by_model",
    "MuSiQue":         "musique_multistep_pcr_by_model",
    "MultiHopRAG":     "multihoprag_multistep_pcr_by_model",
}

# ── Palette ────────────────────────────────────────────────────────────────────
C_CLEAN  = "#898781"   # muted gray  — reference line (labeled directly for identity)
C_EPIST  = "#2a78d6"   # blue slot 1 — CVD adj ΔE 55.8 vs red, well above ≥12 target
C_SMATCH = "#e34948"   # red  slot 6
C_SURF   = "#ffffff"   # chart surface — used as marker ring to create visual gap

CONDITIONS = [
    # (label,              color,    linestyle, linewidth, markersize, zorder, alpha)
    ("Clean baseline",         C_CLEAN,  "--", 1.5, 6.5, 2, 0.80),
    ("Epistemic Confirmation", C_EPIST,  "-",  2.2, 8.0, 4, 1.00),
    ("Search-Match Cue",       C_SMATCH, "-",  2.2, 8.0, 3, 1.00),
]

COND_KEYS = {
    "clean":                  "Clean baseline",
    "epistemic_confirmation": "Epistemic Confirmation",
    "search_match_cue":       "Search-Match Cue",
}


def _pcr_field(k):
    return "wrong_gap_assessment_success" if k == 1 else f"delayed_pcr_at_{k}"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    """Return data[dataset][model][condition] = [pct@1 … @10]."""
    acc = collections.defaultdict(list)
    for model_label, path in JSONL_SOURCES.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — {model_label} will be zeros")
            continue
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not row.get("eligible"):
                    continue
                cond_key = row.get("condition")
                if cond_key not in COND_KEYS:
                    continue
                ds = DATASET_MAP.get(row.get("dataset", ""))
                if ds is None:
                    continue
                for k in STEPS:
                    val = row.get(_pcr_field(k))
                    if val is not None:
                        acc[(ds, model_label, cond_key, k)].append(bool(val))

    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for model in MODELS:
            data[ds][model] = {}
            for cond_key, cond_label in COND_KEYS.items():
                data[ds][model][cond_label] = [
                    100.0 * sum(acc[(ds, model, cond_key, k)]) / len(acc[(ds, model, cond_key, k)])
                    if acc[(ds, model, cond_key, k)] else 0.0
                    for k in STEPS
                ]
    return data


# ── Step-10 end-of-line labels ─────────────────────────────────────────────────
def _annotate_step10(ax, mdata):
    entries = [
        (mdata["Clean baseline"][9],         C_CLEAN),
        (mdata["Epistemic Confirmation"][9],  C_EPIST),
        (mdata["Search-Match Cue"][9],        C_SMATCH),
    ]
    ranked = sorted(range(3), key=lambda i: -entries[i][0])
    # Three vertical slots: highest gets +20pt, middle 0, lowest −20pt
    dy_slots = [20, 0, -20]
    for rank, i in enumerate(ranked):
        val, color = entries[i]
        ax.annotate(
            f"{val:.0f}%",
            xy=(10, val), xytext=(31, dy_slots[rank]),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9, fontweight="semibold", color=color,
        )


# ── Step-1 inline labels (attack conditions only, skip if at zero) ─────────────
def _annotate_step1(ax, mdata):
    THRESH = 1.0  # skip if < 1 % — avoids "0%" clutter for Llama and clean baseline
    ve = mdata["Epistemic Confirmation"][0]
    vs = mdata["Search-Match Cue"][0]

    to_label = [(v, c) for v, c in [(ve, C_EPIST), (vs, C_SMATCH)] if v >= THRESH]
    if not to_label:
        return

    if len(to_label) == 1:
        val, col = to_label[0]
        ax.annotate(
            f"{val:.0f}%", xy=(1, val), xytext=(-26, 5),
            textcoords="offset points", ha="right", va="center",
            fontsize=9, fontweight="semibold", color=col, clip_on=False,
        )
        return

    # Two non-zero attack labels — sort hi/lo and stagger
    (v_hi, c_hi), (v_lo, c_lo) = sorted(to_label, key=lambda x: -x[0])
    dy_hi, dy_lo = (14, -14) if abs(v_hi - v_lo) < 6.0 else (6, -6)
    ax.annotate(f"{v_hi:.0f}%", xy=(1, v_hi), xytext=(-26, dy_hi),
                textcoords="offset points", ha="right", va="center",
                fontsize=9, fontweight="semibold", color=c_hi, clip_on=False)
    ax.annotate(f"{v_lo:.0f}%", xy=(1, v_lo), xytext=(-26, dy_lo),
                textcoords="offset points", ha="right", va="center",
                fontsize=9, fontweight="semibold", color=c_lo, clip_on=False)


# ── Figure builder ─────────────────────────────────────────────────────────────
def plot_dataset(dataset_name, dataset_data, active_models=None):
    if active_models is None:
        active_models = MODELS
    n_models = len(active_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.2), sharey=True)
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor(C_SURF)

    legend_handles = []
    legend_labels  = []

    for col, model in enumerate(active_models):
        ax = axes[col]
        ax.set_facecolor(C_SURF)
        mdata = dataset_data[model]

        # Spines — keep left + bottom, style to match reference palette chrome
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")

        # Grid: horizontal only
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

        # Data range — extra right margin for step-10 labels
        ax.set_xlim(0.3, 12.8)
        ax.set_ylim(-3, 104)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.tick_params(axis="y", color="#c3c2b7")
        ax.tick_params(axis="x", length=0)

        # Plot lines
        for cond_label, color, ls, lw, ms, zorder, alpha in CONDITIONS:
            ys = mdata[cond_label]
            line, = ax.plot(
                STEPS, ys,
                color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, zorder=zorder,
                marker="o", markersize=ms,
                markerfacecolor=color,
                markeredgecolor=C_SURF,   # 2px surface ring per mark spec
                markeredgewidth=2.0,
            )
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(cond_label)

        # Annotations
        _annotate_step1(ax, mdata)
        _annotate_step10(ax, mdata)

        # Labels
        ax.set_xlabel("Decision step", labelpad=7)
        if col == 0:
            ax.set_ylabel("Premature Completion Rate (%)", labelpad=8)
        ax.set_title(model, pad=11, fontweight="semibold")

    # Figure-level legend below the subplots
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.8,
        columnspacing=3.0,
        fontsize=11,
    )

    fig.suptitle(dataset_name, fontsize=15, fontweight="bold", y=1.03)

    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.subplots_adjust(wspace=0.22)

    out = os.path.join(OUT_DIR, DATASET_FILES[dataset_name] + ".png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C_SURF)
    print(f"  saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading data from {RESULTS_DIR} …\n")
    DATA = load_data()
    # Only include models whose JSONL file exists (keeps figures clean until new results arrive)
    active_models = [m for m in MODELS if os.path.exists(JSONL_SOURCES[m])]
    print(f"Active models: {active_models}")
    print(f"Output → {OUT_DIR}\n")
    for ds_name in DATASETS:
        print(f"  {ds_name}")
        ds_data = {m: DATA[ds_name][m] for m in active_models}
        plot_dataset(ds_name, ds_data, active_models)
    print("\nDone.")
