"""Phase 3: primary slide figures -- one per model, 4 dataset panels,
horizontal 100% stacked bars (one per natural-retrieval condition), plus a
separate Oracle calibration figure per model (natural-retrieval conditions
never mixed with Oracle conditions in the same panel)."""
from __future__ import annotations

from collections import defaultdict

from . import categories as cat
from .figures_common import (
    SLIDE_DIR, save_all_formats, NATURAL_CONDITION_ORDER, ORACLE_CONDITION_ORDER,
    MODEL_DISPLAY, DATASET_DISPLAY, CONDITION_DISPLAY, DATASETS,
)
from .source_loader import CoverageRow, MODEL_DATASETS

MIN_LABEL_PCT = 6.0  # below this, a segment is "too small" to label directly


def _cell_stats(rows: list[CoverageRow], model: str, dataset: str, condition: str) -> dict:
    cell = [r for r in rows if r.model == model and r.dataset == dataset and r.condition == condition]
    return cat.summarize_cell(cell)


def _plot_panel(ax, model: str, dataset: str, rows: list[CoverageRow], conditions: tuple[str, ...], dataset_available: bool):
    if not dataset_available:
        ax.axis("off")
        ax.text(0.5, 0.5, "Dataset-level run data unavailable", ha="center", va="center",
                 fontsize=13, style="italic", color="#555555", transform=ax.transAxes)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=14, fontweight="bold")
        return

    y_positions = list(range(len(conditions)))
    ax.set_yticks(y_positions)
    labels = []
    for cond in conditions:
        stats = _cell_stats(rows, model, dataset, cond)
        n = stats["eligible_N"]
        left = 0.0
        for category in cat.CATEGORIES:
            pct = stats[f"pct_{category}"] or 0.0
            ax.barh(y_positions[conditions.index(cond)], pct, left=left, color=cat.CATEGORY_COLOR[category],
                    edgecolor="white", linewidth=0.5, height=0.72)
            if pct >= MIN_LABEL_PCT:
                ax.text(left + pct / 2, y_positions[conditions.index(cond)], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=9, color="white" if category in
                        (cat.CATEGORY_NO_EVIDENCE, cat.CATEGORY_FULL) else "black", fontweight="bold")
            left += pct
        labels.append(f"{CONDITION_DISPLAY[cond]} (N={n})")

    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of eligible runs", fontsize=10)
    ax.set_title(DATASET_DISPLAY[dataset], fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)


def plot_model_figure(rows: list[CoverageRow], model: str, conditions: tuple[str, ...], title_suffix: str,
                       out_name: str) -> None:
    import matplotlib.pyplot as plt
    datasets = DATASETS  # always 4 panel positions, even if one is unavailable
    available = set(MODEL_DATASETS[model])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for i, dataset in enumerate(datasets):
        _plot_panel(axes_flat[i], model, dataset, rows, conditions, dataset in available)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat.CATEGORY_COLOR[c]) for c in cat.CATEGORIES]
    fig.legend(handles, cat.CATEGORIES, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{MODEL_DISPLAY[model]} — {title_suffix}", fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_all_formats(fig, SLIDE_DIR / out_name)
    plt.close(fig)


MODEL_FILE_PREFIX = {"qwen14b": "qwen", "llama8b": "llama", "gpt_oss_20b": "gpt_oss"}


def generate_all(rows: list[CoverageRow]) -> None:
    for model, prefix in MODEL_FILE_PREFIX.items():
        plot_model_figure(
            rows, model, NATURAL_CONDITION_ORDER,
            "Required-evidence coverage at completion (natural retrieval)",
            f"{prefix}_evidence_coverage_distribution",
        )
        plot_model_figure(
            rows, model, ORACLE_CONDITION_ORDER,
            "Required-evidence coverage at completion — Oracle calibration (not a natural-retrieval attack effect)",
            f"{prefix}_oracle_calibration_distribution",
        )
    print("Phase 3 slide figures done")


if __name__ == "__main__":
    from .build_evidence_distribution_figures import run_stage_data
    data = run_stage_data()
    generate_all(data["rows"])
