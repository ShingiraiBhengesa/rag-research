"""Phase 5: missing-required-evidence-unit distribution figures. Per model,
one figure with dataset panels, 100% stacked bars (zero/one/two/three-plus
missing), one bar per natural-retrieval condition -- same visual language
as figures_slide.py, different category set."""
from __future__ import annotations

from .missing_evidence import MISSING_CATEGORIES
from .figures_common import SLIDE_DIR, APPENDIX_DIR, save_all_formats, NATURAL_CONDITION_ORDER, MODEL_DISPLAY, DATASET_DISPLAY, CONDITION_DISPLAY
from .source_loader import CoverageRow, MODEL_DATASETS, DATASETS

MISSING_CATEGORY_COLOR = {
    "zero missing": "#005F3C",
    "one missing": "#94D2BD",
    "two missing": "#CA6702",
    "three or more missing": "#9B2226",
}


def _cell_stats(rows: list[CoverageRow], model: str, dataset: str, condition: str) -> dict:
    from .missing_evidence import summarize_missing
    cell = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == condition]
    return summarize_missing(cell)


def _plot_panel(ax, model: str, dataset: str, rows: list[CoverageRow], conditions: tuple[str, ...], dataset_available: bool):
    if not dataset_available:
        ax.axis("off")
        ax.text(0.5, 0.5, "Dataset-level run data unavailable", ha="center", va="center",
                 fontsize=13, style="italic", color="#555555", transform=ax.transAxes)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=14, fontweight="bold")
        return

    y_positions = list(range(len(conditions)))
    labels = []
    for idx, cond in enumerate(conditions):
        stats = _cell_stats(rows, model, dataset, cond)
        n = stats["eligible_N"]
        left = 0.0
        for category in MISSING_CATEGORIES:
            pct = stats[f"pct_{category}"] or 0.0
            ax.barh(idx, pct, left=left, color=MISSING_CATEGORY_COLOR[category], edgecolor="white", linewidth=0.5, height=0.72)
            if pct >= 6.0:
                ax.text(left + pct / 2, idx, f"{pct:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if category in ("zero missing", "three or more missing") else "black", fontweight="bold")
            left += pct
        labels.append(f"{CONDITION_DISPLAY[cond]} (N={n})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of eligible runs", fontsize=10)
    ax.set_title(DATASET_DISPLAY[dataset], fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)


def plot_model_figure(rows: list[CoverageRow], model: str, out_name: str) -> None:
    import matplotlib.pyplot as plt
    available = set(MODEL_DATASETS[model])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for i, dataset in enumerate(DATASETS):
        _plot_panel(axes_flat[i], model, dataset, rows, NATURAL_CONDITION_ORDER, dataset in available)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MISSING_CATEGORY_COLOR[c]) for c in MISSING_CATEGORIES]
    fig.legend(handles, MISSING_CATEGORIES, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{MODEL_DISPLAY[model]} — missing required-evidence units at completion", fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_all_formats(fig, APPENDIX_DIR / out_name)
    plt.close(fig)


MODEL_FILE_PREFIX = {"qwen14b": "qwen", "llama8b": "llama", "gpt_oss_20b": "gpt_oss"}


def generate_all(rows: list[CoverageRow], missing_table: list[dict]) -> None:
    for model, prefix in MODEL_FILE_PREFIX.items():
        plot_model_figure(rows, model, f"{prefix}_missing_evidence_distribution")
    print("Phase 5 missing-evidence figures done")


if __name__ == "__main__":
    from .build_evidence_distribution_figures import run_stage_data
    data = run_stage_data()
    generate_all(data["rows"], data["missing_table"])
