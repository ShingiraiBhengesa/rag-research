"""Section 13: publication figures.

Unsmoothed step curves for PCR (no interpolation), fixed honest axes
(0-100% where the data is a percentage), and every figure's caption states
its denominator explicitly -- written as a companion `<name>_caption.txt`
next to each figure, since a caption baked into the image itself is harder
to reuse in the paper text.

Main-paper figures use only the 5 prespecified conditions
(`config.MAIN_PAPER_CDF_CONDITIONS`); all-12-condition versions go in
`figures/appendix/`. Both are generated from the same underlying data so
they can never disagree.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prematurestop.evaluation import config

CONDITION_COLORS = {
    "NaturalClean": "#4C72B0",
    "NaturalNeutral": "#55A868",
    "NaturalContinueSearch": "#8172B2",
    "Natural_epistemic_confirmation": "#CCB974",
    "Natural_search_match_cue": "#64B5CD",
    "Natural_source_coverage": "#8C8C8C",
    "Natural_epistemic_confirmation_source": "#D65F5F",
    "Natural_search_match_source": "#B47CC7",
    "Natural_PreRegisteredSufficiency": "#DD8452",
    "Natural_PreRegisteredSufficiencySaturation": "#C44E52",
    "OracleComplete_Neutral": "#937860",
    "OracleComplete_PreRegisteredSufficiencySaturation": "#8B4513",
}

DPI = 300


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig, out_dir: Path, name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    (out_dir / f"{name}_caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def plot_pcr_curve(
    rows: list[dict], model: str, dataset: str, conditions: tuple[str, ...],
    pct_field: str, denom_label: str, out_dir: Path, name: str,
) -> None:
    cell_rows = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["condition"] in conditions]
    if not cell_rows:
        return
    by_cond: dict[str, list[tuple[int, float]]] = defaultdict(list)
    denom_by_cond: dict[str, str] = {}
    for r in cell_rows:
        by_cond[r["condition"]].append((int(r["k"]), float(r[pct_field]) if r[pct_field] not in ("", "NA") else float("nan")))
        denom_by_cond[r["condition"]] = r["denominator_n"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cond in conditions:
        pts = sorted(by_cond.get(cond, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{config.CONDITION_DISPLAY_NAMES.get(cond, cond)} (n={denom_by_cond.get(cond, '?')})"
        ax.step(xs, ys, where="post", label=label, color=CONDITION_COLORS.get(cond, None), linewidth=1.8)
    ax.set_xlabel("Post-injection researcher decision step (k)")
    ax.set_ylabel("Cumulative % completed by step k")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=1)
    ax.set_title(f"{config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    caption = (
        f"{name}: {denom_label}, {config.DISPLAY_MODEL_NAMES.get(model, model)} on "
        f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}. Unsmoothed empirical step "
        f"function, one integer step per post-injection researcher decision. Per-condition n "
        f"(the denominator) shown in the legend; see pcr_denominator_audit.csv for the exact "
        f"route/technical-failure breakdown behind each n."
    )
    _save(fig, out_dir, name, caption)


def generate_pcr_figures(pcr_output_root: Path, fig_root: Path) -> None:
    conditional_rows = _read_csv(pcr_output_root / "pcr" / "pcr_completion_cdf_long.csv")
    unconditional_rows = _read_csv(pcr_output_root / "pcr" / "pcr_unconditional_incidence_long.csv")

    for model in config.INCLUDED_MODELS:
        for dataset in config.DATASETS:
            plot_pcr_curve(
                conditional_rows, model, dataset, config.MAIN_PAPER_CDF_CONDITIONS, "combined_pct",
                "Conditional (completion-time) PCR CDF -- denominator is traces with an observed "
                "ResearchComplete or no-tool exit; reaches 100% by construction",
                fig_root / "main", f"pcr_conditional_{model}_{dataset}",
            )
            plot_pcr_curve(
                conditional_rows, model, dataset, config.CONDITIONS, "combined_pct",
                "Conditional (completion-time) PCR CDF, all 12 conditions -- denominator is traces "
                "with an observed ResearchComplete or no-tool exit; reaches 100% by construction",
                fig_root / "appendix", f"pcr_conditional_{model}_{dataset}_all12",
            )
            plot_pcr_curve(
                unconditional_rows, model, dataset, config.MAIN_PAPER_CDF_CONDITIONS, "cum_pct",
                "Unconditional cumulative local-completion incidence -- denominator is every valid "
                "completed row in the cell; may plateau below 100% (parser failures, uncompleted "
                "episodes, and other terminal outcomes remain visible mass, not hidden)",
                fig_root / "main", f"pcr_unconditional_{model}_{dataset}",
            )
            plot_pcr_curve(
                unconditional_rows, model, dataset, config.CONDITIONS, "cum_pct",
                "Unconditional cumulative local-completion incidence, all 12 conditions -- "
                "denominator is every valid completed row in the cell; may plateau below 100%",
                fig_root / "appendix", f"pcr_unconditional_{model}_{dataset}_all12",
            )


def plot_termination_route_composition(route_summary_rows: list[dict], out_dir: Path) -> None:
    """Figure 3: RC / no-tool / other / technical-error composition, one panel per model."""
    for model in config.INCLUDED_MODELS:
        rows = [r for r in route_summary_rows if r["model"] == model]
        datasets = config.DATASETS
        conditions = config.MAIN_PAPER_CDF_CONDITIONS
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.5), sharey=True)
        for ax, dataset in zip(axes, datasets):
            cell_rows = {r["condition"]: r for r in rows if r["dataset"] == dataset}
            xs = range(len(conditions))
            rc = [float(cell_rows[c]["pct_research_complete"]) if c in cell_rows else 0 for c in conditions]
            nt = [float(cell_rows[c]["pct_no_tool_exit"]) if c in cell_rows else 0 for c in conditions]
            other = [float(cell_rows[c]["pct_other_terminal_route"]) if c in cell_rows else 0 for c in conditions]
            err = [float(cell_rows[c]["pct_technical_error"]) if c in cell_rows else 0 for c in conditions]
            ax.bar(xs, rc, label="ResearchComplete", color="#4C72B0")
            ax.bar(xs, nt, bottom=rc, label="No-tool exit", color="#DD8452")
            bottom2 = [a + b for a, b in zip(rc, nt)]
            ax.bar(xs, other, bottom=bottom2, label="Other", color="#8C8C8C")
            bottom3 = [a + b for a, b in zip(bottom2, other)]
            ax.bar(xs, err, bottom=bottom3, label="Technical error", color="#C44E52")
            ax.set_xticks(list(xs))
            ax.set_xticklabels([config.CONDITION_DISPLAY_NAMES.get(c, c) for c in conditions], rotation=45, ha="right", fontsize=7)
            ax.set_title(config.DISPLAY_DATASET_NAMES.get(dataset, dataset), fontsize=9)
            ax.set_ylim(0, 100)
        axes[0].set_ylabel("% of valid completed rows")
        axes[-1].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.6, 1))
        fig.suptitle(f"Termination-route composition -- {config.DISPLAY_MODEL_NAMES.get(model, model)}")
        caption = (
            f"Termination-route composition, {config.DISPLAY_MODEL_NAMES.get(model, model)}, main-paper "
            f"conditions. Denominator per bar = all valid completed rows in that (dataset, condition) "
            f"cell (pcr_terminal_route_summary.csv). 'Other' and 'technical error' are always kept "
            f"separate, never combined."
        )
        _save(fig, out_dir, f"termination_routes_{model}", caption)


def plot_evidence_progression(evidence_cell_summary_rows: list[dict], out_dir: Path) -> None:
    """Figure 4: mean coverage at each of the 4 boundaries, main-paper conditions, per model-dataset."""
    boundaries = ["mean_coverage_at_injection", "mean_coverage_at_episode0_terminal", "mean_coverage_compressed_notes", "mean_coverage_at_final_input"]
    boundary_labels = ["Injection", "Episode-0\nterminal", "Compressed\nnotes", "Final-report\ninput"]
    for model in config.INCLUDED_MODELS:
        for dataset in config.DATASETS:
            cell_rows = {r["condition"]: r for r in evidence_cell_summary_rows if r["model"] == model and r["dataset"] == dataset}
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for cond in config.MAIN_PAPER_CDF_CONDITIONS:
                row = cell_rows.get(cond)
                if row is None:
                    continue
                ys = []
                for b in boundaries:
                    v = row.get(b, "")
                    ys.append(float(v) * 100 if v not in ("", "None") else float("nan"))
                ax.plot(boundary_labels, ys, marker="o", label=config.CONDITION_DISPLAY_NAMES.get(cond, cond),
                        color=CONDITION_COLORS.get(cond))
            ax.set_ylabel("Mean coverage fraction (%)")
            ax.set_ylim(0, 100)
            ax.set_title(f"{config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            caption = (
                f"Evidence progression across the four measurement boundaries, "
                f"{config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}, "
                f"main-paper conditions. Episode-0-terminal = union of retrieved paragraph IDs across "
                f"every search in the first attacked episode; final-report input = citation-based match "
                f"against what final_report_generation actually received (see REPORT.md Section 3 for the "
                f"full corrected methodology). Denominator = every valid row in the cell (evidence_coverage_cell_summary.csv)."
            )
            _save(fig, out_dir, f"evidence_progression_{model}_{dataset}", caption)


def plot_supervisor_recovery(recovery_summary_rows: list[dict], out_dir: Path) -> None:
    """Figure 5: supervisor-recovery category composition, restricted to redelegated rows, per model."""
    categories = ["n_redelegated_fully_restored", "n_redelegated_partially_restored", "n_redelegated_did_not_restore"]
    cat_labels = ["Fully restored", "Partially restored", "Did not restore"]
    colors = ["#55A868", "#DD8452", "#C44E52"]
    for model in config.INCLUDED_MODELS:
        rows = [r for r in recovery_summary_rows if r["model"] == model]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        datasets = config.DATASETS
        width = 0.6
        xs = range(len(datasets))
        totals_by_dataset = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            ds = r["dataset"]
            for i, cat in enumerate(categories):
                totals_by_dataset[ds][i] += int(r.get(cat, 0) or 0)
        bottoms = [0.0] * len(datasets)
        for i, (cat, label, color) in enumerate(zip(categories, cat_labels, colors)):
            vals = [totals_by_dataset[ds][i] for ds in datasets]
            totals = [sum(totals_by_dataset[ds]) for ds in datasets]
            pct_vals = [100.0 * v / t if t else 0 for v, t in zip(vals, totals)]
            ax.bar(xs, pct_vals, bottom=bottoms, label=label, color=color, width=width)
            bottoms = [b + p for b, p in zip(bottoms, pct_vals)]
        ax.set_xticks(list(xs))
        ax.set_xticklabels([config.DISPLAY_DATASET_NAMES.get(d, d) for d in datasets])
        ax.set_ylabel("% of redelegated rows (all conditions combined)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Supervisor recovery outcome, given redelegation occurred -- {config.DISPLAY_MODEL_NAMES.get(model, model)}")
        ax.legend(fontsize=8)
        caption = (
            f"Supervisor-recovery outcome, {config.DISPLAY_MODEL_NAMES.get(model, model)}, restricted to "
            f"rows where the supervisor actually redelegated (n varies by dataset -- see "
            f"supervisor_recovery_summary.csv for exact counts). Redelegation alone is never counted as "
            f"recovery; category is by whether final-input coverage measurably improved on episode-0-terminal coverage."
        )
        _save(fig, out_dir, f"supervisor_recovery_{model}", caption)


def plot_forest(correctness_transition_rows: list[dict], out_dir: Path) -> None:
    """Figure 6: paired accuracy-effect forest plot, Neutral-vs-attack, main-paper attack conditions."""
    attack_conds = (
        config.CUE_ONLY_ATTACK_CONDITIONS + config.SOURCE_ATTACHED_ATTACK_CONDITIONS + config.PREREGISTERED_SUFFICIENCY_CONDITIONS
    )
    for model in config.INCLUDED_MODELS:
        entries = []
        for dataset in config.DATASETS:
            for cond in attack_conds:
                match = [
                    r for r in correctness_transition_rows
                    if r["model"] == model and r["dataset"] == dataset and r["condition_a"] == "NaturalNeutral" and r["condition_b"] == cond
                ]
                if not match:
                    continue
                r = match[0]
                if r["n"] in ("0", "", None):
                    continue
                entries.append(
                    {
                        "label": f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)} / {config.CONDITION_DISPLAY_NAMES.get(cond, cond)}",
                        "diff": float(r["semantic_accuracy_diff_pct_points"]),
                        "lo": float(r["bootstrap_accuracy_diff_ci_lo_pct"]),
                        "hi": float(r["bootstrap_accuracy_diff_ci_hi_pct"]),
                        "n": int(r["n"]),
                    }
                )
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(entries) + 1.5))
        ys = range(len(entries))
        for y, e in zip(ys, entries):
            ax.plot([e["lo"], e["hi"]], [y, y], color="#4C72B0", linewidth=1.5)
            ax.plot(e["diff"], y, "o", color="#C44E52")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"{e['label']} (n={e['n']})" for e in entries], fontsize=7)
        ax.set_xlabel("Semantic-accuracy difference, attack minus Neutral (percentage points)")
        ax.set_title(f"Paired accuracy effect, strict pairs only -- {config.DISPLAY_MODEL_NAMES.get(model, model)}")
        ax.grid(alpha=0.3, axis="x")
        caption = (
            f"Paired accuracy-effect forest plot, {config.DISPLAY_MODEL_NAMES.get(model, model)}, "
            f"strictly-paired examples only (see audits/strict_pairing_audit.csv -- n is the number of "
            f"strict pairs for that cell, NOT the full per-condition sample). Point = observed "
            f"difference; whiskers = paired bootstrap 95% CI (10,000 resamples, seed={config.DETERMINISTIC_SEED}). "
            f"Most intervals cross zero given small strict-pair n -- see correctness_transitions.csv for "
            f"exact McNemar p-values; treat as descriptive of effect size, not a significance claim, at this n."
        )
        _save(fig, out_dir, f"forest_accuracy_effect_{model}", caption)


def plot_mechanism_raster(
    earlier_completion_rows: list[dict], answer_harm_rows: list[dict],
    model: str, dataset: str, condition_a: str, condition_b: str, out_dir: Path, max_examples: int = 60,
) -> None:
    """Figure 7: per-example mechanism raster for one (model, dataset, comparison).

    One row per strictly-paired example, columns: completion-step delta,
    post-injection search-count delta, episode-0 evidence-loss flag,
    compression-only-loss flag, supervisor-recovery narrowed gap, and the
    final correctness transition -- exactly the six per-example signals
    the task requires in one raster.
    """
    ec_by_example = {
        r["example_id"]: r for r in earlier_completion_rows
        if r["model"] == model and r["dataset"] == dataset and r["condition_a"] == condition_a and r["condition_b"] == condition_b
    }
    ah_by_example = {
        r["example_id"]: r for r in answer_harm_rows
        if r["model"] == model and r["dataset"] == dataset and r["condition_a"] == condition_a and r["condition_b"] == condition_b
    }
    example_ids = sorted(set(ec_by_example) & set(ah_by_example))
    if not example_ids:
        return

    def step_delta_val(r):
        try:
            return int(r["attack_completion_step"]) - int(r["baseline_completion_step"])
        except (ValueError, TypeError):
            return 0

    example_ids.sort(key=lambda eid: step_delta_val(ec_by_example[eid]))
    example_ids = example_ids[:max_examples]

    n = len(example_ids)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.15 * n)))
    col_labels = ["Completion\nstep Δ", "Search\ncount Δ", "Episode-0\nevidence loss", "Compression-\nonly loss", "Supervisor\nrecovered gap", "Final answer\ntransition"]
    ncols = len(col_labels)

    for y, eid in enumerate(example_ids):
        ec = ec_by_example[eid]
        ah = ah_by_example[eid]

        step_d = step_delta_val(ec)
        step_color = "#C44E52" if step_d < 0 else ("#55A868" if step_d > 0 else "#D9D9D9")
        ax.add_patch(plt.Rectangle((0, y), 1, 1, color=step_color))
        ax.text(0.5, y + 0.5, str(step_d), ha="center", va="center", fontsize=6)

        try:
            search_d = float(ec["post_injection_search_count_delta"])
        except (ValueError, TypeError):
            search_d = 0.0
        search_color = "#C44E52" if search_d < 0 else ("#55A868" if search_d > 0 else "#D9D9D9")
        ax.add_patch(plt.Rectangle((1, y), 1, 1, color=search_color))
        ax.text(1.5, y + 0.5, f"{search_d:g}", ha="center", va="center", fontsize=6)

        ep0_loss = ec.get("attack_has_less_evidence_at_episode0_terminal") == "True"
        ax.add_patch(plt.Rectangle((2, y), 1, 1, color="#C44E52" if ep0_loss else "#D9D9D9"))

        comp_loss = ah.get("compression_only_loss") == "True"
        ax.add_patch(plt.Rectangle((3, y), 1, 1, color="#DD8452" if comp_loss else "#D9D9D9"))

        recovered = ec.get("supervisor_recovery_narrowed_gap") == "True"
        ax.add_patch(plt.Rectangle((4, y), 1, 1, color="#55A868" if recovered else "#D9D9D9"))

        control_ok = ah.get("control_correct") == "True"
        attack_ok = ah.get("attack_correct") == "True"
        if control_ok and not attack_ok:
            trans_color, trans_label = "#C44E52", "harmed"
        elif not control_ok and attack_ok:
            trans_color, trans_label = "#55A868", "helped"
        elif control_ok and attack_ok:
            trans_color, trans_label = "#4C72B0", "both OK"
        else:
            trans_color, trans_label = "#8C8C8C", "both wrong"
        ax.add_patch(plt.Rectangle((5, y), 1, 1, color=trans_color))
        ax.text(5.5, y + 0.5, trans_label, ha="center", va="center", fontsize=5.5)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(ncols)])
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel(f"{n} strictly-paired examples (sorted by completion-step Δ)")
    ax.set_title(
        f"Mechanism raster: {config.DISPLAY_MODEL_NAMES.get(model, model)} / {config.DISPLAY_DATASET_NAMES.get(dataset, dataset)} -- "
        f"{config.CONDITION_DISPLAY_NAMES.get(condition_a, condition_a)} vs {config.CONDITION_DISPLAY_NAMES.get(condition_b, condition_b)}",
        fontsize=9,
    )
    caption = (
        f"Per-example mechanism raster, {config.DISPLAY_MODEL_NAMES.get(model, model)} / "
        f"{config.DISPLAY_DATASET_NAMES.get(dataset, dataset)}, {condition_a} vs {condition_b}, strict pairs only "
        f"(n={n} shown of the full strict-pair set, sorted by completion-step delta; capped at {max_examples} rows "
        f"for legibility -- full data in earlier_completion_to_evidence_loss.csv and evidence_loss_to_answer_harm.csv). "
        f"Negative completion-step/search-count delta (red) means the attack completed earlier / searched less than "
        f"the paired baseline. 'Final answer transition' colors: red=harmed (control correct, attack wrong), "
        f"green=helped (reverse), blue=both correct, grey=both wrong."
    )
    _save(fig, out_dir, f"mechanism_raster_{model}_{dataset}_{condition_a}_vs_{condition_b}", caption)


def generate_all_figures(
    old_pcr_output_root: Path,
    new_output_root: Path,
    evidence_cell_summary_rows: list[dict],
    recovery_summary_rows: list[dict],
    correctness_transition_rows: list[dict],
    earlier_completion_rows: list[dict],
    answer_harm_rows: list[dict],
) -> None:
    main_dir = new_output_root / "figures" / "main"
    appendix_dir = new_output_root / "figures" / "appendix"
    recovery_dir = new_output_root / "figures" / "main"  # supervisor recovery is a main-paper figure

    generate_pcr_figures(old_pcr_output_root, new_output_root / "figures")

    route_summary_rows = _read_csv(old_pcr_output_root / "pcr" / "pcr_terminal_route_summary.csv")
    plot_termination_route_composition(route_summary_rows, main_dir)

    plot_evidence_progression(evidence_cell_summary_rows, main_dir)
    plot_supervisor_recovery(recovery_summary_rows, recovery_dir)
    plot_forest(correctness_transition_rows, main_dir)

    # At least one mechanism raster per model, using the strongest attack condition on the
    # dataset with the most strict pairs for that comparison (most legible example, not cherry-picked
    # for effect size -- selection criterion is pair COUNT, stated here explicitly).
    for model in config.INCLUDED_MODELS:
        best = None
        for dataset in config.DATASETS:
            n_pairs = sum(
                1 for r in earlier_completion_rows
                if r["model"] == model and r["dataset"] == dataset
                and r["condition_a"] == "NaturalNeutral" and r["condition_b"] == "Natural_PreRegisteredSufficiencySaturation"
            )
            if best is None or n_pairs > best[1]:
                best = (dataset, n_pairs)
        if best and best[1] > 0:
            plot_mechanism_raster(
                earlier_completion_rows, answer_harm_rows, model, best[0],
                "NaturalNeutral", "Natural_PreRegisteredSufficiencySaturation",
                new_output_root / "figures" / "main",
            )
