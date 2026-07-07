"""
plot_multistep_pcr.py — Publication-quality Multi-Step Premature Completion Rate figures.

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
# Thresholds in percentage-point units.
_TIGHT = 5.0   # almost certain overlap at any single-direction offset
_CLOSE = 10.0  # risk of overlap; wider stagger needed


def _offsets(v_clean, v_epist, v_smatch):
    """
    Return (off_clean, off_epist, off_smatch) as (dx, dy) in offset points.

    Core strategy: rank conditions by value at this step, then assign
    diagonal offsets so no two label boxes can overlap regardless of
    how similar the values are.

    For close pairs the key invariant is: one label goes upper-left,
    the other goes lower-right.  The combined dx+dy separation is far
    larger than any single label box, so overlap is impossible.
    """
    vals = [v_clean, v_epist, v_smatch]
    # rank indices high → low
    hi, md, lo = sorted(range(3), key=lambda i: -vals[i])

    d_hi_md = vals[hi] - vals[md]   # gap between highest and middle
    d_md_lo = vals[md] - vals[lo]   # gap between middle and lowest
    d_hi_lo = vals[hi] - vals[lo]   # total range

    off = [None, None, None]

    # ── All three bunched together ────────────────────────────────────────────
    if d_hi_lo < _TIGHT:
        # Fan upward: left / center-high / right
        off[hi] = (-22, 15)
        off[md] = (  0, 24)
        off[lo] = ( 22, 15)

    # ── Highest two very close; lowest isolated below ─────────────────────────
    elif d_hi_md < _TIGHT:
        off[hi] = (-18, 13)   # upper-left
        off[md] = ( 20, -13)  # lower-right  ← diagonal guarantees separation
        off[lo] = (  0, -16)  # below

    # ── Highest two moderately close ─────────────────────────────────────────
    elif d_hi_md < _CLOSE:
        off[hi] = (-16, 12)
        off[md] = ( 18, -11)
        off[lo] = (  0, -16)

    # ── Lowest two very close; highest isolated above ─────────────────────────
    elif d_md_lo < _TIGHT:
        off[hi] = (  0, 13)
        off[md] = (-18, -13)  # lower-left
        off[lo] = ( 20, -13)  # lower-right ← diagonal

    # ── Lowest two moderately close ──────────────────────────────────────────
    elif d_md_lo < _CLOSE:
        off[hi] = (  0, 13)
        off[md] = (-16, -11)
        off[lo] = ( 18, -11)

    # ── All well separated ────────────────────────────────────────────────────
    else:
        off[hi] = (  0,  12)
        off[md] = ( 16,   6)
        off[lo] = (  0, -16)

    return (off[0], off[1], off[2])


def _ha(dx):
    """Horizontal alignment matched to label direction."""
    if dx < -4:
        return "right"
    if dx >  4:
        return "left"
    return "center"


# ── Figure builder ────────────────────────────────────────────────────────────

def plot_dataset(dataset_name, dataset_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor("white")

    legend_handles = []

    for col, model in enumerate(MODELS):
        ax = axes[col]
        ax.set_facecolor("white")
        mdata = dataset_data[model]

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6,
                color="#cccccc", alpha=0.75)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        # Draw lines
        for cond_name, color, ls in CONDITIONS:
            ys = mdata[cond_name]
            line, = ax.plot(
                STEPS, ys,
                color=color, linestyle=ls, linewidth=2.5,
                marker="o", markersize=6,
                markerfacecolor=color, markeredgewidth=0,
                zorder=3, label=cond_name,
            )
            if col == 0:
                legend_handles.append(line)

        # Value labels — one call per data point
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
                ax.annotate(
                    f"{val:.2f}%",
                    xy=(step, val),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=_ha(dx), va="center",
                    fontsize=7.5, fontweight="bold",
                    color=color,
                )

        # Axes
        ax.set_xlim(0.6, 3.4)
        ax.set_ylim(0, 100)
        ax.set_xticks(STEPS)
        ax.set_xticklabels(STEP_LABELS, fontsize=10)
        ax.set_xlabel("Decision step", fontsize=11, labelpad=6)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.tick_params(axis="both", labelsize=10)

        if col == 0:
            ax.set_ylabel("Premature Completion Rate (%)", fontsize=11, labelpad=8)

        ax.set_title(model, fontsize=13, fontweight="bold", pad=10)

    # Shared legend
    fig.legend(
        handles=legend_handles,
        labels=[c[0] for c in CONDITIONS],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=11,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=2.0,
    )

    # Dataset super-title and footer
    fig.suptitle(dataset_name, fontsize=14, fontweight="bold", y=1.04)
    fig.text(
        0.5, -0.03,
        "Three-step rollout: Clean baseline vs. two chosen attacks",
        ha="center", va="bottom", fontsize=9, color="#777777",
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])

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
