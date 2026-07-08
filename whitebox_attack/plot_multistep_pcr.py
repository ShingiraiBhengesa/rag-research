"""
plot_multistep_pcr.py — Publication-quality PCR@1/2/3 figures for Wrong Gap Assessment.

One figure per dataset (4 total), each with three side-by-side subplots, one per model.
Three lines per subplot: Clean baseline, Epistemic Confirmation, Search-Match Cue.
Value labels on every point with diagonal stagger so no two labels overlap.

Output: ../whitebox_results/figures/<dataset>_multistep_pcr_by_model.png  (300 DPI)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Output directory ──────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "whitebox_results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_CLEAN  = "#555555"
C_EPIST  = "#0066cc"
C_SMATCH = "#d62728"

STEPS       = [1, 2, 3]
STEP_LABELS = ["Step 1", "Step 2", "Step 3"]
MODELS      = ["Qwen2.5-7B", "Qwen2.5-14B", "Llama-3.1-8B"]

# ── Hardcoded PCR@1 / PCR@2 / PCR@3 values (%) ───────────────────────────────
DATA = {
    "HotpotQA": {
        "Qwen2.5-7B": {
            "Clean baseline":         [ 0.00, 17.65, 22.25],
            "Epistemic Confirmation": [18.13, 73.37, 75.07],
            "Search-Match Cue":       [24.65, 69.69, 70.25],
        },
        "Qwen2.5-14B": {
            "Clean baseline":         [ 0.00,  7.46, 13.18],
            "Epistemic Confirmation": [33.33, 52.50, 55.28],
            "Search-Match Cue":       [41.39, 57.50, 60.28],
        },
        "Llama-3.1-8B": {
            "Clean baseline":         [ 0.00, 24.04, 33.71],
            "Epistemic Confirmation": [ 0.00, 71.50, 88.50],
            "Search-Match Cue":       [ 0.00, 70.25, 86.50],
        },
    },
    "2WikiMultiHopQA": {
        "Qwen2.5-7B": {
            "Clean baseline":         [ 0.00,  5.13, 17.41],
            "Epistemic Confirmation": [ 9.46, 63.17, 65.22],
            "Search-Match Cue":       [11.25, 62.15, 65.98],
        },
        "Qwen2.5-14B": {
            "Clean baseline":         [ 0.00,  0.89,  6.03],
            "Epistemic Confirmation": [32.99, 52.43, 59.34],
            "Search-Match Cue":       [36.83, 53.20, 55.75],
        },
        "Llama-3.1-8B": {
            "Clean baseline":         [ 0.00,  4.05,  8.11],
            "Epistemic Confirmation": [ 0.00, 68.22, 87.86],
            "Search-Match Cue":       [ 0.00, 58.91, 78.81],
        },
    },
    "MuSiQue": {
        "Qwen2.5-7B": {
            "Clean baseline":         [ 0.00,  9.86, 11.47],
            "Epistemic Confirmation": [11.06, 42.46, 44.22],
            "Search-Match Cue":       [12.56, 41.46, 42.21],
        },
        "Qwen2.5-14B": {
            "Clean baseline":         [ 0.00,  2.74,  5.94],
            "Epistemic Confirmation": [16.79, 24.06, 30.83],
            "Search-Match Cue":       [17.04, 24.81, 33.33],
        },
        "Llama-3.1-8B": {
            "Clean baseline":         [ 0.00, 12.44, 17.11],
            "Epistemic Confirmation": [ 0.00, 59.61, 75.91],
            "Search-Match Cue":       [ 0.00, 52.80, 68.61],
        },
    },
    "MultiHopRAG": {
        "Qwen2.5-7B": {
            "Clean baseline":         [ 0.00, 11.74, 22.80],
            "Epistemic Confirmation": [ 9.95, 57.87, 63.19],
            "Search-Match Cue":       [ 6.71, 53.70, 58.80],
        },
        "Qwen2.5-14B": {
            "Clean baseline":         [ 0.00,  2.23,  6.70],
            "Epistemic Confirmation": [12.13, 20.14, 26.77],
            "Search-Match Cue":       [ 8.01, 18.54, 24.03],
        },
        "Llama-3.1-8B": {
            "Clean baseline":         [ 0.00,  1.57,  6.73],
            "Epistemic Confirmation": [ 0.00, 19.77, 37.47],
            "Search-Match Cue":       [ 0.00,  9.20, 24.37],
        },
    },
}

CONDITIONS = [
    ("Clean baseline",         C_CLEAN,  "--"),
    ("Epistemic Confirmation", C_EPIST,  "-"),
    ("Search-Match Cue",       C_SMATCH, "-"),
]

DATASET_FILES = {
    "HotpotQA":        "hotpotqa_multistep_pcr_by_model.png",
    "2WikiMultiHopQA": "2wikimultihopqa_multistep_pcr_by_model.png",
    "MuSiQue":         "musique_multistep_pcr_by_model.png",
    "MultiHopRAG":     "multihoprag_multistep_pcr_by_model.png",
}

# ── Label placement ───────────────────────────────────────────────────────────
# Thresholds in percentage-point units. Wider than before to catch close pairs
# earlier and push them onto a full diagonal, giving more breathing room.
_TIGHT = 7.0
_CLOSE = 14.0


def _offsets(v_clean, v_epist, v_smatch):
    """Return (off_clean, off_epist, off_smatch) as (dx, dy) offset points.

    Ranks the three values high→low, then assigns offsets so no two label
    boxes overlap. Close pairs always get a full diagonal split (upper-left /
    lower-right) so the combined separation exceeds any single label box.
    """
    vals = [v_clean, v_epist, v_smatch]
    hi, md, lo = sorted(range(3), key=lambda i: -vals[i])

    d_hi_md = vals[hi] - vals[md]
    d_md_lo = vals[md] - vals[lo]
    d_hi_lo = vals[hi] - vals[lo]

    off = [None, None, None]

    if d_hi_lo < _TIGHT:                      # all three bunched — fan
        off[hi] = (-28, 18)
        off[md] = (  0, 28)
        off[lo] = ( 28, 18)
    elif d_hi_md < _TIGHT:                    # top two very close — diagonal
        off[hi] = (-24, 17)
        off[md] = ( 26, -17)
        off[lo] = (  0, -20)
    elif d_hi_md < _CLOSE:                    # top two moderately close
        off[hi] = (-22, 15)
        off[md] = ( 24, -14)
        off[lo] = (  0, -20)
    elif d_md_lo < _TIGHT:                    # bottom two very close — diagonal
        off[hi] = (  0, 18)
        off[md] = (-24, -17)
        off[lo] = ( 26, -17)
    elif d_md_lo < _CLOSE:                    # bottom two moderately close
        off[hi] = (  0, 18)
        off[md] = (-22, -14)
        off[lo] = ( 24, -14)
    else:                                     # all well separated
        off[hi] = (  0, 17)
        off[md] = ( 22,  7)
        off[lo] = (  0, -20)

    return (off[0], off[1], off[2])


def _ha(dx):
    if dx < -5: return "right"
    if dx >  5: return "left"
    return "center"


# ── Figure builder ────────────────────────────────────────────────────────────

def plot_dataset(dataset_name, dataset_data):
    # Wider figure (20 in) gives each panel ~6.5 in — enough room for labels.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    fig.patch.set_facecolor("white")

    legend_handles = []

    for col, model in enumerate(MODELS):
        ax = axes[col]
        ax.set_facecolor("white")
        mdata = dataset_data[model]

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5,
                color="#d5d5d5", alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        for cond_name, color, ls in CONDITIONS:
            ys = mdata[cond_name]
            lw = 1.8 if ls == "--" else 2.4
            alpha = 0.70 if ls == "--" else 1.0
            line, = ax.plot(
                STEPS, ys,
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                marker="o", markersize=7.5,
                markerfacecolor=color, markeredgewidth=0,
                zorder=3, label=cond_name,
            )
            if col == 0:
                legend_handles.append(line)

        # Value labels — 1 decimal place, larger offsets for breathing room
        for si, step in enumerate(STEPS):
            vc = mdata["Clean baseline"][si]
            ve = mdata["Epistemic Confirmation"][si]
            vs = mdata["Search-Match Cue"][si]

            (dxc, dyc), (dxe, dye), (dxs, dys) = _offsets(vc, ve, vs)

            for val, color, dx, dy in [
                (vc, C_CLEAN,  dxc, dyc),
                (ve, C_EPIST,  dxe, dye),
                (vs, C_SMATCH, dxs, dys),
            ]:
                # Labels below the 0% axis collide with x-tick text — push to upper-left
                if val < 5.0 and dy < 0:
                    dx, dy = -28, 14
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(step, val),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=_ha(dx), va="center",
                    fontsize=8.5, fontweight="bold",
                    color=color,
                )

        # Wider x/y bounds give labels room at the edges
        ax.set_xlim(0.45, 3.65)
        ax.set_ylim(-4, 106)
        ax.set_xticks(STEPS)
        ax.set_xticklabels(STEP_LABELS, fontsize=12)
        ax.set_xlabel("Decision step", fontsize=12, labelpad=8)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.tick_params(axis="both", labelsize=11, which="both", length=3)
        ax.tick_params(axis="y", which="minor", length=0)

        if col == 0:
            ax.set_ylabel("Premature Completion Rate (%)", fontsize=12, labelpad=10)

        ax.set_title(model, fontsize=14, fontweight="bold", pad=14)

    # Legend with generous spacing so it doesn't crowd the panels
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
        "Three-step controlled rollout — Clean baseline vs. Epistemic Confirmation vs. Search-Match Cue",
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
    print(f"Output → {OUT_DIR}\n")
    for ds_name, ds_data in DATA.items():
        plot_dataset(ds_name, ds_data)
    print("\nDone.")
