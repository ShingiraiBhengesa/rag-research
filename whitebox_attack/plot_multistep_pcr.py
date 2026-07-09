"""
plot_multistep_pcr.py — PCR@1–10 figures for Wrong Gap Assessment (reads live JSONL).

One figure per dataset (4 total), each with three side-by-side subplots, one per model.
Three lines per subplot: Clean baseline, Epistemic Confirmation, Search-Match Cue.
Value labels at step 1 (inline, diagonal stagger) and step 10 (right-side end-of-line).

Reads from:
  ../whitebox_results/s1_attack2e_pcr10_{qwen7b,qwen14b,llama8b}.jsonl

Output:
  ../whitebox_results/figures/<dataset>_multistep_pcr_by_model.png  (300 DPI)
"""

import os
import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "whitebox_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_SOURCES = {
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen7b.jsonl"),
    "Qwen2.5-14B":  os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_qwen14b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "s1_attack2e_pcr10_llama8b.jsonl"),
}

# Dataset key in JSONL → display name
DATASET_MAP = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

MODELS   = list(JSONL_SOURCES.keys())
DATASETS = list(DATASET_MAP.values())
STEPS    = list(range(1, 11))   # [1 … 10]

DATASET_FILES = {
    "HotpotQA":        "hotpotqa_multistep_pcr_by_model.png",
    "2WikiMultiHopQA": "2wikimultihopqa_multistep_pcr_by_model.png",
    "MuSiQue":         "musique_multistep_pcr_by_model.png",
    "MultiHopRAG":     "multihoprag_multistep_pcr_by_model.png",
}

# ── Palette ───────────────────────────────────────────────────────────────────
C_CLEAN  = "#555555"
C_EPIST  = "#0066cc"
C_SMATCH = "#d62728"

CONDITIONS = [
    ("Clean baseline",         C_CLEAN,  "--"),
    ("Epistemic Confirmation", C_EPIST,  "-"),
    ("Search-Match Cue",       C_SMATCH, "-"),
]

COND_KEYS = {
    "clean":                  "Clean baseline",
    "epistemic_confirmation": "Epistemic Confirmation",
    "search_match_cue":       "Search-Match Cue",
}


def _pcr_field(k):
    return "wrong_gap_assessment_success" if k == 1 else f"delayed_pcr_at_{k}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Aggregate JSONL into nested dict: data[dataset][model][condition] = [pct@1..@10]"""
    acc = collections.defaultdict(list)   # (ds, model, cond, k) -> [bool]

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
                ds_raw = row.get("dataset", "")
                ds = DATASET_MAP.get(ds_raw)
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
                series = []
                for k in STEPS:
                    lst = acc[(ds, model, cond_key, k)]
                    series.append(100.0 * sum(lst) / len(lst) if lst else 0.0)
                data[ds][model][cond_label] = series
    return data


# ── Step-1 label stagger (same logic as before) ───────────────────────────────
_TIGHT = 7.0
_CLOSE = 14.0


def _offsets(vc, ve, vs_val):
    vals = [vc, ve, vs_val]
    hi, md, lo = sorted(range(3), key=lambda i: -vals[i])
    d_hi_md = vals[hi] - vals[md]
    d_md_lo = vals[md] - vals[lo]
    d_hi_lo = vals[hi] - vals[lo]
    off = [None, None, None]
    if d_hi_lo < _TIGHT:
        off[hi] = (-28, 18); off[md] = (0, 28);  off[lo] = (28, 18)
    elif d_hi_md < _TIGHT:
        off[hi] = (-24, 17); off[md] = (26, -17); off[lo] = (0, -20)
    elif d_hi_md < _CLOSE:
        off[hi] = (-22, 15); off[md] = (24, -14); off[lo] = (0, -20)
    elif d_md_lo < _TIGHT:
        off[hi] = (0, 18);   off[md] = (-24, -17); off[lo] = (26, -17)
    elif d_md_lo < _CLOSE:
        off[hi] = (0, 18);   off[md] = (-22, -14); off[lo] = (24, -14)
    else:
        off[hi] = (0, 17);   off[md] = (22, 7);    off[lo] = (0, -20)
    return off[0], off[1], off[2]


def _ha(dx):
    if dx < -5: return "right"
    if dx >  5: return "left"
    return "center"


# ── Figure builder ────────────────────────────────────────────────────────────

def plot_dataset(dataset_name, dataset_data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    fig.patch.set_facecolor("white")
    legend_handles = []

    for col, model in enumerate(MODELS):
        ax = axes[col]
        ax.set_facecolor("white")
        mdata = dataset_data[model]

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="#d5d5d5", alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        lines_dict = {}
        for cond_label, color, ls in CONDITIONS:
            ys = mdata[cond_label]
            lw = 1.8 if ls == "--" else 2.4
            alpha = 0.70 if ls == "--" else 1.0
            line, = ax.plot(
                STEPS, ys,
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                marker="o", markersize=5.5,
                markerfacecolor=color, markeredgewidth=0,
                zorder=3, label=cond_label,
            )
            lines_dict[cond_label] = (line, ys)
            if col == 0:
                legend_handles.append(line)

        # Step-1 inline labels (diagonal stagger)
        vc1 = mdata["Clean baseline"][0]
        ve1 = mdata["Epistemic Confirmation"][0]
        vs1 = mdata["Search-Match Cue"][0]
        (dxc, dyc), (dxe, dye), (dxs, dys) = _offsets(vc1, ve1, vs1)
        for val, color, dx, dy in [(vc1, C_CLEAN, dxc, dyc),
                                    (ve1, C_EPIST, dxe, dye),
                                    (vs1, C_SMATCH, dxs, dys)]:
            if val < 5.0 and dy < 0:
                dx, dy = -28, 14
            ax.annotate(f"{val:.1f}%", xy=(1, val), xytext=(dx, dy),
                        textcoords="offset points",
                        ha=_ha(dx), va="center",
                        fontsize=8.5, fontweight="bold", color=color)

        # Step-10 end-of-line labels, stacked vertically to avoid pile-up
        step10_vals = [
            (mdata["Clean baseline"][9],         C_CLEAN),
            (mdata["Epistemic Confirmation"][9],  C_EPIST),
            (mdata["Search-Match Cue"][9],        C_SMATCH),
        ]
        # Sort by value to assign dy offsets cleanly (highest gets top spot)
        step10_sorted = sorted(enumerate(step10_vals), key=lambda x: -x[1][0])
        dy_slots = [18, 0, -18]   # upper / middle / lower
        for rank, (orig_i, (val, color)) in enumerate(step10_sorted):
            dy = dy_slots[rank]
            ax.annotate(f"{val:.1f}%", xy=(10, val), xytext=(32, dy),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=8.5, fontweight="bold", color=color)

        ax.set_xlim(0.3, 12.2)
        ax.set_ylim(-4, 106)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(s) for s in STEPS], fontsize=11)
        ax.set_xlabel("Decision step", fontsize=12, labelpad=8)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.tick_params(axis="both", labelsize=11, which="both", length=3)
        ax.tick_params(axis="y", which="minor", length=0)

        if col == 0:
            ax.set_ylabel("Premature Completion Rate (%)", fontsize=12, labelpad=10)

        ax.set_title(model, fontsize=14, fontweight="bold", pad=14)

    fig.legend(
        handles=legend_handles,
        labels=[c[0] for c in CONDITIONS],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.975),
        fontsize=12,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.8,
        columnspacing=3.0,
    )

    fig.suptitle(dataset_name, fontsize=16, fontweight="bold", y=1.04)
    fig.text(
        0.5, -0.01,
        "Ten-step controlled rollout — Clean baseline vs. Epistemic Confirmation vs. Search-Match Cue",
        ha="center", va="bottom", fontsize=10, color="#777777",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    plt.subplots_adjust(wspace=0.28)

    out = os.path.join(OUT_DIR, DATASET_FILES[dataset_name])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")
    return out


if __name__ == "__main__":
    print(f"Loading data from {RESULTS_DIR} ...\n")
    DATA = load_data()
    print(f"Output → {OUT_DIR}\n")
    for ds_name in DATASETS:
        plot_dataset(ds_name, DATA[ds_name])
    print("\nDone.")
