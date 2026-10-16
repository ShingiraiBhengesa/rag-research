"""Phase 4: empirical CDF figures for the appendix. Step lines, markers at
observed values, no smoothing, no KDE. Kept per-dataset (never pooled).

Interpretation direction (stated here and in every caption): a cue curve
shifted toward the upper-left means more runs completed with LOWER
required-evidence coverage than the control."""
from __future__ import annotations

from .figures_common import (
    APPENDIX_DIR, save_all_formats, PRIMARY_ECDF_CONDITIONS, SOURCE_ABLATION_ECDF_CONDITIONS,
    CONDITION_LINE_COLOR, MODEL_DISPLAY, DATASET_DISPLAY, CONDITION_DISPLAY,
)
from .source_loader import CoverageRow, MODEL_DATASETS


def _ecdf_xy(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def plot_ecdf_panel(ax, rows: list[CoverageRow], model: str, dataset: str, conditions: tuple[str, ...]):
    for cond in conditions:
        cell = [r.coverage for r in rows if r.model == model and r.dataset == dataset and r.condition == cond]
        if not cell:
            continue
        xs, ys = _ecdf_xy(cell)
        # step plot, markers at observed values -- no smoothing
        ax.step([0.0] + xs, [0.0] + ys, where="post", color=CONDITION_LINE_COLOR.get(cond, "#333333"),
                linewidth=2, label=f"{CONDITION_DISPLAY[cond]} (N={len(cell)})")
        ax.scatter(xs, ys, color=CONDITION_LINE_COLOR.get(cond, "#333333"), s=10, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Evidence coverage at completion")
    ax.set_ylabel("Cumulative proportion of runs")
    ax.set_title(DATASET_DISPLAY[dataset], fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)


def plot_ecdf_figure(rows: list[CoverageRow], model: str, conditions: tuple[str, ...], title: str, out_name: str) -> None:
    import matplotlib.pyplot as plt
    datasets = MODEL_DATASETS[model]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.5), squeeze=False)
    for i, dataset in enumerate(datasets):
        plot_ecdf_panel(axes[0][i], rows, model, dataset, conditions)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(conditions), 3), fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        f"{MODEL_DISPLAY[model]} — {title}\n"
        "A cue curve shifted toward the upper-left = more runs completed with LOWER required-evidence coverage.",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    save_all_formats(fig, APPENDIX_DIR / out_name)
    plt.close(fig)


MODEL_FILE_PREFIX = {"qwen14b": "qwen", "llama8b": "llama", "gpt_oss_20b": "gpt_oss"}


def generate_all(rows: list[CoverageRow]) -> None:
    for model, prefix in MODEL_FILE_PREFIX.items():
        plot_ecdf_figure(rows, model, PRIMARY_ECDF_CONDITIONS,
                          "Primary comparison ECDF (Clean vs. sufficiency/confirmation cues)",
                          f"{prefix}_ecdf_primary_comparison")
        plot_ecdf_figure(rows, model, SOURCE_ABLATION_ECDF_CONDITIONS,
                          "Source-ablation ECDF (does adding Source Coverage shift the curve further?)",
                          f"{prefix}_ecdf_source_ablation")
    print("Phase 4 ECDF figures done")


if __name__ == "__main__":
    from .build_evidence_distribution_figures import run_stage_data
    data = run_stage_data()
    generate_all(data["rows"])
