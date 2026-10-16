"""Phase 6: diverging stacked-bar figure for paired evidence-loss
distribution (evidence loss / no change / evidence gain), one bar per
(model, dataset, comparison) row -- never pooled across datasets."""
from __future__ import annotations

from .figures_common import SLIDE_DIR, save_all_formats, MODEL_DISPLAY, DATASET_DISPLAY
from .source_loader import CoverageRow, MODEL_DATASETS
from .pairing import COMPARISONS, CALIBRATION_COMPARISONS

LOSS_COLOR = "#9B2226"
NO_CHANGE_COLOR = "#BBBBBB"
GAIN_COLOR = "#005F3C"


def plot_diverging_figure(paired_loss_table: list[dict], out_name: str) -> None:
    import matplotlib.pyplot as plt

    # One row per (model, dataset, comparison) with paired_N > 0, grouped by
    # comparison, model-dataset as sub-rows within each comparison block.
    comparisons = [c[0] for c in COMPARISONS]
    rows_by_comparison: dict[str, list[dict]] = {c: [] for c in comparisons}
    for r in paired_loss_table:
        if r["paired_N"]:
            rows_by_comparison[r["comparison_name"]].append(r)

    all_rows = []
    row_labels = []
    for comp in comparisons:
        for r in sorted(rows_by_comparison[comp], key=lambda x: (x["model"], x["dataset"])):
            all_rows.append(r)
            calib_tag = " [CALIBRATION]" if comp in CALIBRATION_COMPARISONS else ""
            row_labels.append(f"{comp}{calib_tag} — {MODEL_DISPLAY[r['model']]} / {DATASET_DISPLAY[r['dataset']]} (N={r['paired_N']})")

    n = len(all_rows)
    fig, ax = plt.subplots(figsize=(14, 0.35 * n + 2))
    y = list(range(n))
    loss = [r["pct_evidence_loss"] for r in all_rows]
    same = [r["pct_no_change"] for r in all_rows]
    gain = [r["pct_evidence_gain"] for r in all_rows]

    # Diverging layout: no-change centered at 0, loss to the right (positive),
    # gain to the left (negative) -- a standard diverging-stacked-bar
    # construction (loss and gain plotted as mirrored offsets from a
    # centered "no change" band).
    half_same = [s / 2 for s in same]
    ax.barh(y, same, left=[-h for h in half_same], color=NO_CHANGE_COLOR, height=0.7, label="No change")
    ax.barh(y, loss, left=half_same, color=LOSS_COLOR, height=0.7, label="Evidence loss")
    ax.barh(y, [-g for g in gain], left=[-h - g for h, g in zip(half_same, gain)], color=GAIN_COLOR, height=0.7, label="Evidence gain")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of paired examples (gain <- | -> loss; no-change centered)", fontsize=11)
    ax.set_title("Paired evidence-loss distribution — every model x dataset x comparison",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_all_formats(fig, SLIDE_DIR / out_name)
    plt.close(fig)


def generate_all(rows: list[CoverageRow], paired_loss_table: list[dict]) -> None:
    plot_diverging_figure(paired_loss_table, "paired_evidence_loss_distribution")
    print("Phase 6 paired diverging-bar figure done")


if __name__ == "__main__":
    from .build_evidence_distribution_figures import run_stage_data
    data = run_stage_data()
    generate_all(data["rows"], data["paired_loss_table"])
