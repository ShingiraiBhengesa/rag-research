"""Step 12 (figures): PCR curves, PCR deltas, route composition, evidence
progression.

Reuses `analysis/final_paper_evaluation/figures.py`'s visual conventions
(`CONDITION_COLORS`, `DPI`, the `_save()` dual png+pdf+caption helper)
directly by import. `plot_pcr_curve` there hardcodes legacy column names
(`k`, `denominator_n`) and can't be called unchanged against
`pcr_stepwise.csv`'s schema, so a new curve plotter is defined here reading
the new schema. `plot_evidence_progression` is reused **completely
unchanged** -- it consumes exactly `evidence.summarize_by_cell`'s output
keys verbatim.

Not yet implemented in this pass: the paired-effect forest plot and the
per-example mechanism/trace raster (spec 8's "1-2 paired trace rasters per
model") -- both need the `mechanism_ext.py` adapter porting
`analysis/final_paper_evaluation/mechanism.py`'s per-example step/coverage
delta logic to canonical field names, which is a larger, separate piece of
work. `data/process_metrics.csv` already has the per-example fields a
raster would need (`local_completion_step`, `local_completion_route`,
`post_observation_search_calls`); building the raster is a follow-up.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import config
from analysis.evidence import build_evidence_records, summarize_by_cell
from analysis.final_paper_evaluation.figures import CONDITION_COLORS, DPI, _save, plot_evidence_progression

from .scope import CANONICAL_BUNDLE_MODELS


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_pcr_curve_stepwise(rows: list[dict], model: str, dataset: str, conditions: tuple[str, ...], out_dir: Path, name: str) -> None:
    cell_rows = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["condition"] in conditions]
    if not cell_rows:
        return
    by_cond: dict[str, list[tuple[int, float]]] = defaultdict(list)
    n_eligible_by_cond: dict[str, str] = {}
    for r in cell_rows:
        pcr = r["pcr"]
        by_cond[r["condition"]].append((int(r["step"]), 100.0 * float(pcr) if pcr not in ("", "None") else float("nan")))
        n_eligible_by_cond[r["condition"]] = r["n_eligible"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cond in conditions:
        pts = sorted(by_cond.get(cond, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{config.CONDITION_DISPLAY_NAMES.get(cond, cond)} (n={n_eligible_by_cond.get(cond, '?')})"
        ax.step(xs, ys, where="post", label=label, color=CONDITION_COLORS.get(cond), linewidth=1.8)
    ax.set_xlabel("Post-observation researcher decision step (k)")
    ax.set_ylabel("PCR@k (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=1)
    ax.set_title(f"{config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    caption = (
        f"{name}: PCR@k, {config.DISPLAY_MODEL_NAMES.get(model, model)} on "
        f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. Denominator = pcr_eligible runs "
        f"(canonical_completion.py definition, see MASTER_RESULTS.md methodology section). "
        f"Source: data/pcr_stepwise.csv."
    )
    _save(fig, out_dir, name, caption)


def generate_pcr_figures(data_dir: Path, fig_root: Path) -> None:
    """Per-(model,dataset) appendix reference only -- all 12 conditions, one
    step-function panel each. The prespecified-5-condition "main" variant
    this used to also generate (pcr_curve_{model}_{dataset}_main.*) is
    superseded by plot_pcr_condition_grid's one-figure-per-dataset,
    both-models, all-12-condition, smoothed grid below; removed rather than
    kept alongside it to avoid two different "the" PCR curve figures."""
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    for model in CANONICAL_BUNDLE_MODELS:
        for dataset in config.DATASETS:
            plot_pcr_curve_stepwise(
                stepwise, model, dataset, config.CONDITIONS,
                fig_root / "pcr_curves" / "appendix", f"pcr_curve_{model}_{dataset}_all12",
            )


# Attack-panel figures: one image per (model, family), rows = that family's
# individual conditions, columns = datasets. Every condition is compared only
# against NaturalClean -- including Oracle Complete-Saturation, by explicit
# request, even though its evidence regime (all annotated gold evidence
# supplied) differs from Clean's real BM25 retrieval; this is a deliberate
# choice to keep one single baseline throughout, not an implicit default.
#
# 2026-09-28: extended from 4 families (8 attack conditions) to 5 (all 12
# conditions) by adding "Non-attack controls" -- NaturalNeutral,
# NaturalContinueSearch, and OracleComplete_Neutral were previously shown
# only in the dense all12 appendix overlay (generate_pcr_figures below), not
# in this one-condition-per-panel comparison style. They are not attacks (no
# manipulative content is injected), so they get their own family rather than
# being folded into one of the four attack families above.
PCR_GRID_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Cue-only attacks", "NaturalClean", config.CUE_ONLY_ATTACK_CONDITIONS),
    ("Source-attached ablations", "NaturalClean", config.SOURCE_ATTACHED_ATTACK_CONDITIONS),
    ("Preregistered sufficiency", "NaturalClean", config.PREREGISTERED_SUFFICIENCY_CONDITIONS),
    ("Oracle calibration", "NaturalClean", tuple(
        c for c in config.ORACLE_CONDITIONS if c != "OracleComplete_Neutral"
    )),
    ("Non-attack controls", "NaturalClean", (
        "NaturalNeutral", "NaturalContinueSearch", "OracleComplete_Neutral",
    )),
)

# CONDITION_COLORS["Natural_source_coverage"] (#8C8C8C) is a near-duplicate
# of this grid's #888888 gray-dashed baseline -- solid vs. dashed alone
# doesn't reliably distinguish two near-identical grays at figure scale, so
# Source Coverage effectively disappeared into the Clean baseline in the
# "Cue-only attacks" row. Local override, grid-only (not touched in the
# shared CONDITION_COLORS dict other figures still use unchanged).
_GRID_COLOR_OVERRIDES: dict[str, str] = {
    "Natural_source_coverage": "#C2185B",
}


def _condition_thesis_labels() -> dict[str, str]:
    """condition_id -> thesis_display_label, the authoritative thesis-facing
    condition names (analysis/canonical/condition_metadata.csv) -- distinct
    from and nicer than config.CONDITION_DISPLAY_NAMES's shorter internal
    labels (e.g. "Fixed Sufficiency" here vs "PreRegisteredSufficiency"
    there)."""
    metadata_path = Path(__file__).resolve().parents[1] / "condition_metadata.csv"
    return {r["condition_id"]: r["thesis_display_label"] for r in _read_csv(metadata_path)}


def _smooth_xy(xs: list[int], ys: list[float], n_dense: int = 300):
    """Monotone-preserving (PCHIP) interpolation for a visually smooth curve
    that never overshoots or dips below the true step-wise values -- PCR is
    monotonic nondecreasing by construction (canonical_completion.py), so a
    plain cubic spline could introduce a spurious visible decrease that
    misrepresents the metric. Falls back to the raw points untouched if
    there are too few to interpolate."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    if len(xs) < 3:
        return xs, ys
    xs_dense = np.linspace(min(xs), max(xs), n_dense)
    return xs_dense, PchipInterpolator(xs, ys)(xs_dense)


def _panel_xlim(by_cond: dict[str, list[tuple[int, float]]], conds: tuple[str, ...], floor: int = 10, ceiling: int = 30, threshold: float = 99.5) -> int:
    """Adaptive x-axis upper bound for one panel: the last step at which any
    plotted condition was still below `threshold`, plus a small margin --
    never below `floor`, never past the real observed `ceiling`. PCR curves
    saturate near 100% and then run flat for the remainder of the 30-step
    budget; a fixed 1-30 axis buries the actual divergence between baseline
    and attack in the first few steps against 15-20 steps of dead space.
    This keeps the y-axis honest (still fixed 0-100, see plot_pcr_attack_panels)
    while zooming the x-axis to where the effect is actually visible, computed
    per panel so a slower-saturating condition/dataset isn't clipped."""
    last_below = floor
    for cond in conds:
        for step, pcr in by_cond.get(cond, []):
            if pcr < threshold:
                last_below = max(last_below, step)
    return min(last_below + 2, ceiling)


def plot_pcr_attack_panels(rows: list[dict], model: str, group_label: str, baseline_cond: str, attack_conds: tuple[str, ...], out_dir: Path, name: str, datasets: tuple[str, ...] = config.DATASETS) -> None:
    """One image per (model, attack family): rows = the family's individual
    conditions (each gets its own panel -- never more than one attack line
    sharing a panel with the baseline), columns = `datasets` (defaults to
    all 4; pass an explicit subset for a model with incomplete dataset
    coverage, e.g. gpt_oss_20b missing 2wiki -- an empty column would
    otherwise render as a blank panel instead of simply not existing).
    Lets a reader follow one specific attack's effect across every
    available dataset at a glance, for one model at a time."""
    model_rows = [r for r in rows if r["model"] == model]
    if not model_rows:
        return

    thesis_labels = _condition_thesis_labels()
    n_rows, n_cols = len(attack_conds), len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), sharex=False, sharey=True, squeeze=False,
    )
    legend_handles: dict[str, plt.Line2D] = {}
    baseline_color = "#888888" if baseline_cond == "NaturalClean" else CONDITION_COLORS.get(baseline_cond)
    baseline_label = thesis_labels.get(baseline_cond, baseline_cond)

    for row_idx, attack_cond in enumerate(attack_conds):
        attack_color = _GRID_COLOR_OVERRIDES.get(attack_cond, CONDITION_COLORS.get(attack_cond))
        attack_label = thesis_labels.get(attack_cond, attack_cond)
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            by_cond: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in model_rows:
                if r["dataset"] != dataset:
                    continue
                pcr = r["pcr"]
                if pcr in ("", "None"):
                    continue
                by_cond[r["condition"]].append((int(r["step"]), 100.0 * float(pcr)))

            for cond, label, style in (
                (baseline_cond, baseline_label, {"linestyle": "--", "color": baseline_color, "linewidth": 2.2}),
                (attack_cond, attack_label, {"color": attack_color, "linewidth": 2.6}),
            ):
                pts = sorted(by_cond.get(cond, []))
                if not pts:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                xs_s, ys_s = _smooth_xy(xs, ys)
                (line,) = ax.plot(xs_s, ys_s, label=label, **style)
                legend_handles.setdefault(label, line)

            ax.set_ylim(0, 100)
            ax.set_xlim(1, _panel_xlim(by_cond, (baseline_cond, attack_cond)))
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=11)
            if row_idx == 0:
                ax.set_title(config.DISPLAY_DATASET_NAMES.get(dataset, dataset), fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{attack_label}\nPCR@k (%)", fontsize=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Decision step (k)", fontsize=12)

    fig.suptitle(
        f"{config.DISPLAY_MODEL_NAMES.get(model, model)} — {group_label}",
        fontsize=19, fontweight="bold",
    )
    fig.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="lower center", ncol=min(len(legend_handles), 4), fontsize=12, bbox_to_anchor=(0.5, -0.015 * (4 / max(n_rows, 1))),
    )
    fig.tight_layout(rect=(0, 0.06 / max(n_rows, 1) + 0.02, 1, 0.94))

    caption = (
        f"{name}: PCR@k, {config.DISPLAY_MODEL_NAMES.get(model, model)}, {group_label}. Each panel plots "
        f"exactly one condition ({attack_label if n_rows == 1 else 'one per row'}) against its baseline "
        f"({baseline_label}) for one dataset. Y-axis fixed 0-100% across every panel (not rescaled per panel, "
        f"so effect sizes stay comparable); x-axis upper bound is computed per panel as the last step any "
        f"plotted condition was still below 99.5% PCR (+2 steps margin), trimming the flat post-saturation "
        f"tail without altering the y-scale. Curves are monotone (PCHIP) interpolations of the true step-wise "
        f"pcr_stepwise.csv values -- no data beyond the observed step range (1-30) is implied or extrapolated. "
        f"Denominator = pcr_eligible runs (canonical_completion.py definition). Source: data/pcr_stepwise.csv."
    )
    _save(fig, out_dir, name, caption)


def _slugify(label: str) -> str:
    return label.lower().replace(" ", "-")


def generate_pcr_attack_panels(data_dir: Path, fig_root: Path) -> None:
    stepwise = _read_csv(data_dir / "pcr_stepwise.csv")
    for model in CANONICAL_BUNDLE_MODELS:
        for group_label, baseline_cond, attack_conds in PCR_GRID_ROWS:
            plot_pcr_attack_panels(
                stepwise, model, group_label, baseline_cond, attack_conds,
                fig_root / "pcr_curves", f"pcr_attack_{model}_{_slugify(group_label)}",
            )


def plot_pcr_delta_ribbon(rows: list[dict], model: str, dataset: str, comparison_name: str, out_dir: Path, name: str) -> None:
    pts = sorted(
        (int(r["step"]), r["delta_pcr"], r["ci_lower"], r["ci_upper"])
        for r in rows
        if r["model"] == model and r["dataset"] == dataset and r["comparison_name"] == comparison_name and r["delta_pcr"] not in ("", "None")
    )
    if not pts:
        return
    xs = [p[0] for p in pts]
    delta = [100.0 * float(p[1]) for p in pts]
    lo = [100.0 * float(p[2]) for p in pts]
    hi = [100.0 * float(p[3]) for p in pts]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, delta, color="#C44E52", linewidth=1.8, label="delta_pcr (cue - control)")
    ax.fill_between(xs, lo, hi, color="#C44E52", alpha=0.2, label="paired bootstrap 95% CI")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Post-observation researcher decision step (k)")
    ax.set_ylabel("Delta PCR@k, cue minus control (percentage points)")
    ax.set_title(f"{comparison_name}\n{config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    caption = (
        f"{comparison_name}, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
        f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. Positive = cue condition completed "
        f"MORE by step k than control (attack accelerated completion). Shaded band = paired bootstrap "
        f"95% CI on the exact same paired eligible set both curves use. Source: data/pcr_paired_deltas_stepwise.csv."
    )
    _save(fig, out_dir, name, caption)


def generate_pcr_delta_figures(data_dir: Path, fig_root: Path) -> None:
    rows = _read_csv(data_dir / "pcr_paired_deltas_stepwise.csv")
    seen = set()
    for r in rows:
        key = (r["model"], r["dataset"], r["comparison_name"])
        if key in seen:
            continue
        seen.add(key)
        safe_name = r["comparison_name"].replace(" ", "_").replace("/", "-")
        plot_pcr_delta_ribbon(rows, r["model"], r["dataset"], r["comparison_name"], fig_root / "pcr_deltas", f"delta_{r['model']}_{r['dataset']}_{safe_name}")


def _route_summary_rows_from_decomposition(data_dir: Path) -> list[dict]:
    """Adapts route_decomposition_stepwise.csv's final route tallies (7
    canonical routes) into the 4-category shape
    figures.plot_termination_route_composition expects
    (research_complete/no_tool_exit/other/technical_error), reused
    unchanged below."""
    rows = _read_csv(data_dir / "route_decomposition_stepwise.csv")
    last_per_cell: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        last_per_cell[(r["model"], r["dataset"], r["condition"])] = r  # last row per cell (rows are step-ordered)
    out = []
    for (model, dataset, condition), r in last_per_cell.items():
        n_total = sum(int(r[f"final_n_{route}"]) for route in ("RESEARCH_COMPLETE", "NO_TOOL_DIRECT_EXIT", "MAX_BUDGET_EXIT", "TOOL_OR_PARSER_FAILURE", "TRACE_LOGGING_FAILURE", "EXECUTION_FAILURE", "OTHER_VALID_ROUTE"))
        if n_total == 0:
            continue
        rc = int(r["final_n_RESEARCH_COMPLETE"])
        nt = int(r["final_n_NO_TOOL_DIRECT_EXIT"])
        other = int(r["final_n_MAX_BUDGET_EXIT"]) + int(r["final_n_TOOL_OR_PARSER_FAILURE"]) + int(r["final_n_OTHER_VALID_ROUTE"])
        err = int(r["final_n_TRACE_LOGGING_FAILURE"]) + int(r["final_n_EXECUTION_FAILURE"])
        out.append(
            {
                "model": model, "dataset": dataset, "condition": condition,
                "pct_research_complete": 100.0 * rc / n_total,
                "pct_no_tool_exit": 100.0 * nt / n_total,
                "pct_other_terminal_route": 100.0 * other / n_total,
                "pct_technical_error": 100.0 * err / n_total,
            }
        )
    return out


def generate_route_figures(data_dir: Path, fig_root: Path) -> None:
    from analysis.final_paper_evaluation.figures import plot_termination_route_composition

    route_summary = _route_summary_rows_from_decomposition(data_dir)
    plot_termination_route_composition(route_summary, fig_root / "route_decomposition")


def generate_evidence_figures(production_root: Path, fig_root: Path, max_workers: int = 8) -> None:
    """Reuses evidence.build_evidence_records + summarize_by_cell +
    figures.plot_evidence_progression entirely unchanged -- verbatim key
    match, no adapter needed."""
    from analysis.completion_inventory import _row_is_valid, build_inventory

    inventory = build_inventory(production_root, models=CANONICAL_BUNDLE_MODELS)
    valid_rows = [r for r in inventory.rows if _row_is_valid(r)]
    evidence_records = build_evidence_records(valid_rows, max_workers=max_workers)
    summary_rows = summarize_by_cell(evidence_records)
    plot_evidence_progression(summary_rows, fig_root / "evidence")


def plot_paired_answer_flips(rows: list[dict], out_dir: Path) -> None:
    """`figures/answers/`: paired answer-flip composition per model
    (spec section 8's "Paired answer-flip summary"). Self-contained --
    `answer_paired_flips.csv` (`answer_evaluation.compute_paired_answer_transitions`'s
    output) already has exactly the 4 flip categories + n_pairs + net_harm."""
    for model in CANONICAL_BUNDLE_MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        by_comparison: dict[str, list[dict]] = defaultdict(list)
        for r in model_rows:
            by_comparison[f"{r['condition_a']} vs {r['condition_b']}"].append(r)
        labels = sorted(by_comparison)
        cats = ("both_correct", "control_correct_attack_wrong", "control_wrong_attack_correct", "both_wrong")
        cat_labels = ("Both correct", "Control correct, cue wrong (harmed)", "Control wrong, cue correct (helped)", "Both wrong")
        colors = ("#4C72B0", "#C44E52", "#55A868", "#8C8C8C")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
        ys = range(len(labels))
        bottoms = [0.0] * len(labels)
        for cat, cat_label, color in zip(cats, cat_labels, colors):
            vals = []
            for label in labels:
                rs = by_comparison[label]
                n = sum(int(r["n_pairs"]) for r in rs) or 1
                vals.append(100.0 * sum(int(r[cat]) for r in rs) / n)
            ax.barh(list(ys), vals, left=bottoms, label=cat_label, color=color)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("% of strictly-paired examples (pooled across datasets)")
        ax.set_xlim(0, 100)
        ax.set_title(f"Paired answer-flip composition -- {config.DISPLAY_MODEL_NAMES.get(model, model)}")
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2)
        caption = (
            f"Paired answer-flip composition, {config.DISPLAY_MODEL_NAMES.get(model, model)}, pooled across "
            f"the 4 datasets (per-dataset counts in data/answer_paired_flips.csv). Pre-adjudication automatic "
            f"semantic labels -- see needs_adjudication_flag counts in answer_run_level.csv before citing as "
            f"final accuracy (spec 3.3)."
        )
        _save(fig, out_dir, f"answer_paired_flips_{model}", caption)


def generate_answer_figures(data_dir: Path, fig_root: Path) -> None:
    rows = _read_csv(data_dir / "answer_paired_flips.csv")
    plot_paired_answer_flips(rows, fig_root / "answers")


def plot_paired_trace_raster(mechanism_rows: list[dict], answer_rows: list[dict], model: str, dataset: str, comparison_name: str, out_dir: Path, max_examples: int = 60) -> None:
    """`figures/trace_rasters/`: one row per strictly-paired example, columns
    matching `figures.plot_mechanism_raster`'s layout (completion-step delta,
    search-count delta, episode-0 evidence-loss flag, supervisor-recovery
    flag, final-answer transition) -- built directly from this bundle's own
    `mechanism_run_level.csv` + `answer_run_level.csv` rather than calling
    `plot_mechanism_raster` unchanged, since that function expects
    `answer_harm.py`'s legacy column names (`compression_only_loss`,
    `control_correct`/`attack_correct`), which this bundle does not compute
    (no `answer_harm_ext.py` port exists yet)."""
    mech = [r for r in mechanism_rows if r["model"] == model and r["dataset"] == dataset and r["comparison_name"] == comparison_name]
    if not mech:
        return
    control_cond, cue_cond = mech[0]["condition_a"], mech[0]["condition_b"]
    ans_by_key = {(r["model"], r["dataset"], r["condition"], r["example_id"]): r for r in answer_rows}

    def step_delta(r):
        try:
            return int(r["attack_completion_step"]) - int(r["baseline_completion_step"])
        except (ValueError, TypeError):
            return 0

    mech.sort(key=step_delta)
    mech = mech[:max_examples]
    n = len(mech)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.15 * n)))
    col_labels = ["Completion\nstep Δ", "Search\ncount Δ", "Episode-0\nevidence loss", "Supervisor\nrecovered gap", "Final answer\ntransition"]
    ncols = len(col_labels)
    for y, r in enumerate(mech):
        step_d = step_delta(r)
        step_color = "#C44E52" if step_d < 0 else ("#55A868" if step_d > 0 else "#D9D9D9")
        ax.add_patch(plt.Rectangle((0, y), 1, 1, color=step_color))
        ax.text(0.5, y + 0.5, str(step_d), ha="center", va="center", fontsize=6)

        try:
            search_d = float(r["post_observation_search_calls_delta"])
        except (ValueError, TypeError):
            search_d = 0.0
        search_color = "#C44E52" if search_d < 0 else ("#55A868" if search_d > 0 else "#D9D9D9")
        ax.add_patch(plt.Rectangle((1, y), 1, 1, color=search_color))
        ax.text(1.5, y + 0.5, f"{search_d:g}", ha="center", va="center", fontsize=6)

        ep0_loss = r.get("attack_has_less_evidence_at_episode0_terminal") == "True"
        ax.add_patch(plt.Rectangle((2, y), 1, 1, color="#C44E52" if ep0_loss else "#D9D9D9"))

        recovered = r.get("supervisor_recovery_narrowed_gap") == "True"
        ax.add_patch(plt.Rectangle((3, y), 1, 1, color="#55A868" if recovered else "#D9D9D9"))

        eid = r["example_id"]
        control_rec = ans_by_key.get((model, dataset, control_cond, eid))
        cue_rec = ans_by_key.get((model, dataset, cue_cond, eid))
        control_ok = bool(control_rec) and control_rec["automatic_semantic_label"] == "correct"
        cue_ok = bool(cue_rec) and cue_rec["automatic_semantic_label"] == "correct"
        if control_ok and not cue_ok:
            trans_color, trans_label = "#C44E52", "harmed"
        elif not control_ok and cue_ok:
            trans_color, trans_label = "#55A868", "helped"
        elif control_ok and cue_ok:
            trans_color, trans_label = "#4C72B0", "both OK"
        else:
            trans_color, trans_label = "#8C8C8C", "both wrong"
        ax.add_patch(plt.Rectangle((4, y), 1, 1, color=trans_color))
        ax.text(4.5, y + 0.5, trans_label, ha="center", va="center", fontsize=5.5)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(ncols)])
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel(f"{n} strictly-paired examples (sorted by completion-step Δ)")
    ax.set_title(
        f"Paired trace raster: {config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)} -- {comparison_name}",
        fontsize=9,
    )
    caption = (
        f"Per-example paired trace raster, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
        f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}, {comparison_name}, strict pairs only "
        f"(n={n} shown, sorted by completion-step delta; capped at {max_examples} rows for legibility -- "
        f"full data in data/mechanism_run_level.csv joined with data/answer_run_level.csv). Negative "
        f"completion-step/search-count delta (red) means the cue condition completed earlier / searched "
        f"less than the paired control. Final-answer-transition colors: red=harmed (control correct, cue "
        f"wrong), green=helped (reverse), blue=both correct, grey=both wrong. Pre-adjudication automatic "
        f"semantic labels (spec 3.3)."
    )
    _save(fig, out_dir, f"trace_raster_{model}_{dataset}_{comparison_name.replace(' ', '_').replace('/', '-')}", caption)


def generate_trace_raster_figures(data_dir: Path, fig_root: Path) -> None:
    """One or two rasters per model (spec section 8), on the two main-paper
    primary comparisons: Epistemic Confirmation vs Natural Neutral, and
    Fixed Sufficiency Saturation vs Natural Neutral, on hotpot (the
    dataset with the largest example count)."""
    mechanism_rows = _read_csv(data_dir / "mechanism_run_level.csv")
    answer_rows = _read_csv(data_dir / "answer_run_level.csv")
    if not mechanism_rows or not answer_rows:
        return
    raster_comparisons = ("Epistemic Confirmation vs Natural Neutral", "Fixed Sufficiency Saturation vs Natural Neutral")
    for model in CANONICAL_BUNDLE_MODELS:
        for comparison_name in raster_comparisons:
            plot_paired_trace_raster(mechanism_rows, answer_rows, model, "hotpot", comparison_name, fig_root / "trace_rasters")


def generate_all_figures(bundle_root: Path, production_root: Path, max_workers: int = 8) -> None:
    data_dir = bundle_root / "data"
    fig_root = bundle_root / "figures"
    generate_pcr_figures(data_dir, fig_root)
    generate_pcr_attack_panels(data_dir, fig_root)
    generate_pcr_delta_figures(data_dir, fig_root)
    generate_route_figures(data_dir, fig_root)
    generate_evidence_figures(production_root, fig_root, max_workers=max_workers)
    generate_answer_figures(data_dir, fig_root)
    generate_trace_raster_figures(data_dir, fig_root)
