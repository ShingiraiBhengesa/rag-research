"""
plot_wrong_gap_assessment.py — PCR@k figures for the final Wrong Gap Assessment experiment.

Reads final_wga_{model}.jsonl files and produces:
  1. One per-dataset figure (4 total): 5 model subplots side-by-side, 5 attack lines + clean.
  2. One standalone per-model figure (5 total): 4 dataset subplots in a 2×2 grid.

The x-axis is dynamic: extends to the maximum observed completion step across all
models/conditions (typically 25–35), not capped at 10.

Conditions plotted (in order, with distinct colours)
-----------------------------------------------------
  clean                           gray dashed  — reference baseline
  epistemic_confirmation          blue
  search_match_cue                red
  source_coverage                 green
  epistemic_confirmation_source   purple
  search_match_source             orange

PCR@k derivation (from JSONL)
------------------------------
  For attack rows: pcr_at_k = completion_step_attack is not None and completion_step_attack <= k
  For clean rows:  pcr_at_k = completion_step_clean  is not None and completion_step_clean  <= k

Usage
-----
  cd whitebox_attack/
  conda run -n whitebox python plot_wrong_gap_assessment.py
"""

import os
import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Publication rcParams ───────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.linewidth":    0.8,
    "xtick.major.size":  0,
    "ytick.major.size":  3.5,
    "ytick.major.width": 0.8,
    "ytick.minor.size":  0,
    "axes.axisbelow":    True,
    "grid.linewidth":    0.7,
    "grid.color":        "#e1e0d9",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "whitebox_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_SOURCES = {
    "Qwen2.5-7B":   os.path.join(RESULTS_DIR, "final_wga_qwen7b.jsonl"),
    "Qwen2.5-14B":  os.path.join(RESULTS_DIR, "final_wga_qwen14b.jsonl"),
    "Llama-3.1-8B": os.path.join(RESULTS_DIR, "final_wga_llama8b.jsonl"),
    "Gemma-3-27B":  os.path.join(RESULTS_DIR, "final_wga_gemma27b.jsonl"),
    "GPT-OSS-20B":  os.path.join(RESULTS_DIR, "final_wga_gptoss20b.jsonl"),
}

DATASET_MAP = {
    "hotpot":      "HotpotQA",
    "2wiki":       "2WikiMultiHopQA",
    "musique":     "MuSiQue",
    "multihoprag": "MultiHopRAG",
}
MODELS   = list(JSONL_SOURCES.keys())
DATASETS = list(DATASET_MAP.values())

# ── Condition display config ───────────────────────────────────────────────────
# (key, display_label, color, linestyle, linewidth, markersize, alpha)
CONDITION_STYLES = [
    ("clean",                         "Clean baseline",              "#898781", "--", 1.5, 5.5, 0.80),
    ("epistemic_confirmation",        "Epistemic Confirmation",      "#2a78d6", "-",  2.0, 7.0, 1.00),
    ("search_match_cue",              "Search-Match Cue",            "#e34948", "-",  2.0, 7.0, 1.00),
    ("source_coverage",               "Source Coverage",             "#2ca02c", "-",  2.0, 7.0, 1.00),
    ("epistemic_confirmation_source", "Epistemic + Source",          "#9467bd", "-",  2.0, 7.0, 1.00),
    ("search_match_source",           "Search-Match + Source",       "#e87722", "-",  2.0, 7.0, 1.00),
]
COND_KEYS    = [c[0] for c in CONDITION_STYLES]
COND_LABEL   = {c[0]: c[1] for c in CONDITION_STYLES}
COND_COLOR   = {c[0]: c[2] for c in CONDITION_STYLES}
C_SURF       = "#ffffff"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    """Return (data, max_step) where data[dataset][model][condition] = list of
    completion_step values (int or None) for eligible rows, and max_step is the
    largest observed finite completion step across all files."""
    # acc[(dataset_label, model_label, condition)] = [completion_step, ...]
    acc = collections.defaultdict(list)
    max_step = 10  # floor

    for model_label, path in JSONL_SOURCES.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — {model_label} skipped")
            continue
        with open(path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not row.get("eligible"):
                    continue
                cond = row.get("condition")
                if cond not in COND_KEYS:
                    continue
                ds = DATASET_MAP.get(row.get("dataset", ""))
                if ds is None:
                    continue

                # completion step: use attack for attack rows, clean for clean row
                if cond == "clean":
                    cs = row.get("completion_step_clean")
                else:
                    cs = row.get("completion_step_attack")

                acc[(ds, model_label, cond)].append(cs)
                if cs is not None and cs > max_step:
                    max_step = cs

    # Reshape into nested dict
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for model in MODELS:
            data[ds][model] = {
                cond: acc[(ds, model, cond)] for cond in COND_KEYS
            }
    return data, max_step


def pcr_curve(comp_steps, max_k):
    """Return list of PCR@k values for k = 1..max_k."""
    n = len(comp_steps)
    if n == 0:
        return [0.0] * max_k
    return [
        100.0 * sum(1 for s in comp_steps if s is not None and s <= k) / n
        for k in range(1, max_k + 1)
    ]


# ── Annotation helpers ─────────────────────────────────────────────────────────

def _annotate_end(ax, steps, curves_by_cond, max_k):
    """Right-side end-of-line labels at step max_k."""
    entries = [
        (curves_by_cond[cond][-1], COND_COLOR[cond])
        for cond in COND_KEYS
        if curves_by_cond[cond]
    ]
    ranked = sorted(range(len(entries)), key=lambda i: -entries[i][0])
    dy_slots = [20, 10, 0, -10, -20, -30]
    for rank, i in enumerate(ranked):
        val, color = entries[i]
        dy = dy_slots[rank] if rank < len(dy_slots) else 0
        ax.annotate(f"{val:.0f}%",
                    xy=(max_k, val), xytext=(28, dy),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=8, fontweight="semibold", color=color)


def _annotate_step1(ax, curves_by_cond):
    """Left-side step-1 labels for attack conditions with PCR@1 > 1%."""
    labeled = [
        (curves_by_cond[cond][0], COND_COLOR[cond])
        for cond in COND_KEYS
        if cond != "clean" and curves_by_cond[cond] and curves_by_cond[cond][0] >= 1.0
    ]
    if not labeled:
        return
    labeled_sorted = sorted(labeled, key=lambda x: -x[0])
    dy_slots = [14, 5, -4, -13, -22]
    for i, (val, color) in enumerate(labeled_sorted):
        dy = dy_slots[i] if i < len(dy_slots) else 0
        ax.annotate(f"{val:.0f}%",
                    xy=(1, val), xytext=(-26, dy),
                    textcoords="offset points",
                    ha="right", va="center",
                    fontsize=8, fontweight="semibold", color=color, clip_on=False)


# ── Core subplot painter ───────────────────────────────────────────────────────

def paint_subplot(ax, mdata, max_k, is_leftmost=False):
    steps = list(range(1, max_k + 1))
    curves = {}
    for cond, label, color, ls, lw, ms, alpha in CONDITION_STYLES:
        comp_steps = mdata.get(cond, [])
        ys = pcr_curve(comp_steps, max_k) if comp_steps else [0.0] * max_k
        curves[cond] = ys
        ax.plot(steps, ys,
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                marker="o", markersize=ms,
                markerfacecolor=color, markeredgecolor=C_SURF,
                markeredgewidth=1.8, zorder=3 if cond == "clean" else 4)

    _annotate_step1(ax, curves)
    _annotate_end(ax, steps, curves, max_k)

    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_xlim(0.3, max_k + 3.5)
    ax.set_ylim(-3, 104)
    # Tick every 5 steps, label every 10
    ax.set_xticks(range(1, max_k + 1))
    major_ticks = [k for k in range(1, max_k + 1) if k == 1 or k % 5 == 0]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(k) for k in major_ticks])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="y", color="#c3c2b7")
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Decision step", labelpad=6)
    if is_leftmost:
        ax.set_ylabel("Premature Completion Rate (%)", labelpad=8)
    return curves


def make_legend_handles():
    handles, labels = [], []
    for cond, label, color, ls, lw, ms, alpha in CONDITION_STYLES:
        line, = plt.plot([], [], color=color, linestyle=ls, linewidth=lw,
                         marker="o", markersize=ms,
                         markerfacecolor=color, markeredgecolor=C_SURF,
                         markeredgewidth=1.8, alpha=alpha)
        handles.append(line)
        labels.append(label)
    plt.close()
    return handles, labels


# ── Per-dataset figure (1 row × 5 model subplots) ─────────────────────────────

def plot_by_dataset(ds_name, ds_data, active_models, max_k):
    n = len(active_models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.2), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(C_SURF)

    for col, model in enumerate(active_models):
        ax = axes[col]
        ax.set_facecolor(C_SURF)
        paint_subplot(ax, ds_data[model], max_k, is_leftmost=(col == 0))
        ax.set_title(model, pad=11, fontweight="semibold")

    handles, labels = make_legend_handles()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, frameon=False, handlelength=2.4, handletextpad=0.7,
               columnspacing=2.5, fontsize=10)
    fig.suptitle(ds_name, fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.subplots_adjust(wspace=0.20)

    slug = ds_name.lower().replace(" ", "").replace("multi", "multi").replace("hop", "hop")
    slug_map = {
        "HotpotQA": "hotpotqa", "2WikiMultiHopQA": "2wikimultihopqa",
        "MuSiQue": "musique", "MultiHopRAG": "multihoprag",
    }
    out = os.path.join(OUT_DIR, f"wga_{slug_map.get(ds_name, ds_name)}_by_model.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C_SURF)
    print(f"  saved: {out}")
    plt.close(fig)


# ── Per-model figure (2×2 dataset grid) ───────────────────────────────────────

def plot_by_model(model_name, model_data, max_k):
    ds_list = [("hotpot", "HotpotQA"), ("2wiki", "2WikiMultiHopQA"),
               ("musique", "MuSiQue"), ("multihoprag", "MultiHopRAG")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    fig.patch.set_facecolor(C_SURF)
    axes = axes.flatten()

    for idx, (ds_key, ds_label) in enumerate(ds_list):
        ax = axes[idx]
        ax.set_facecolor(C_SURF)
        mdata = model_data.get(ds_label, {})
        paint_subplot(ax, mdata, max_k, is_leftmost=(idx % 2 == 0))
        ax.set_title(ds_label, pad=11, fontweight="semibold")

    handles, labels = make_legend_handles()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=3, frameon=False, handlelength=2.4, handletextpad=0.7,
               columnspacing=2.5, fontsize=10)

    slug = model_name.lower().replace(".", "").replace("-", "").replace("/", "")
    fig.suptitle(f"{model_name} — Wrong Gap Assessment PCR@1–{max_k}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.subplots_adjust(hspace=0.35, wspace=0.12)

    out = os.path.join(OUT_DIR, f"wga_{slug}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C_SURF)
    print(f"  saved: {out}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading data from {RESULTS_DIR} …\n")
    DATA, MAX_K = load_data()
    active_models = [m for m in MODELS if os.path.exists(JSONL_SOURCES[m])]
    print(f"Active models : {active_models}")
    print(f"Max step found: {MAX_K}")
    print(f"Output → {OUT_DIR}\n")

    print("── Per-dataset figures (model comparison) ──")
    for ds_name in DATASETS:
        print(f"  {ds_name}")
        ds_data = {m: DATA[ds_name][m] for m in active_models}
        plot_by_dataset(ds_name, ds_data, active_models, MAX_K)

    print("\n── Per-model figures (dataset comparison) ──")
    for model in active_models:
        print(f"  {model}")
        model_data = {ds: DATA[ds][model] for ds in DATASETS}
        plot_by_model(model, model_data, MAX_K)

    print("\nDone.")
