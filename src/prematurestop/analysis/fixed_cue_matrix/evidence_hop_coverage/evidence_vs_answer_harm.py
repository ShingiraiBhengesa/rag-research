"""Smallest justified next analysis after evidence/hop coverage (see
`EVIDENCE_HOP_RESULTS.md` section 9): does an evidence-incomplete local
completion actually coincide with a paired answer-harm flip, or is the
evidence-loss association found in `paired_evidence_deltas.csv` a
process-level effect that doesn't touch the final answer?

Locked/reused, unmodified:
  - `run_level_evidence_hops.csv` (this package's own already-validated
    output -- `local_full_evidence`, read from disk, never recomputed)
  - `prematurestop.evaluation.answer_evaluation.build_answer_records` /
    `.AnswerRecord.semantic_correct` (the same answer-correctness source
    `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s answer tables use)
  - `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS` (the single
    reconciled comparison manifest -- no third comparisons list)
  - `prematurestop.evaluation.completion_inventory.build_inventory` / `_row_is_valid`

New here: for each comparison, splits the strict answer-pairs by whether
the CUE side was evidence-complete or evidence-incomplete at local
completion (`local_full_evidence`), and reports the paired
correct/incorrect transition counts within each split -- an association
check, not a causal or thesis-ready claim (see output file's own header).
"""

from __future__ import annotations

import csv
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.evaluation.answer_evaluation import AnswerRecord, TECHNICAL_FAILURE_REASONS, build_answer_records
from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import COMPARISONS
from prematurestop.analysis.fixed_cue_matrix.results_bundle.scope import CANONICAL_BUNDLE_MODELS
from prematurestop.evaluation.completion_inventory import _row_is_valid, build_inventory
from prematurestop.stats.metrics import mcnemar_test, wilson_ci
from prematurestop.paths import RESULTS_E1


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_evidence_lookup(run_level_path: Path) -> dict[tuple[str, str, str, str], dict]:
    """(model, dataset, condition, example_id) -> {local_full_evidence, local_coverage_fraction}."""
    out: dict[tuple[str, str, str, str], dict] = {}
    for row in _read_csv(run_level_path):
        key = (row["model"], row["dataset"], row["condition"], row["example_id"])
        lfe = row["local_full_evidence"]
        lcf = row["local_coverage_fraction"]
        out[key] = {
            "local_full_evidence": None if lfe in ("", "None") else (lfe == "True"),
            "local_coverage_fraction": None if lcf in ("", "None") else float(lcf),
        }
    return out


def build_strictly_paired_example_ids(records: list[AnswerRecord]) -> dict[tuple[str, str, str, str], set[str]]:
    """Existence-based pairing (present under both conditions), same rule
    `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/answer_ext.py` already uses for
    answer-flip pairing -- answer correctness pairing is independent of PCR
    eligibility (spec's "same model, dataset, benchmark example" unit)."""
    by_model_dataset_example: dict[tuple[str, str, str], set[str]] = {}
    for r in records:
        by_model_dataset_example.setdefault((r.model, r.dataset, r.example_id), set()).add(r.condition)

    out: dict[tuple[str, str, str, str], set[str]] = {}
    for (model, dataset, example_id), conditions_present in by_model_dataset_example.items():
        for comp in COMPARISONS:
            if comp.control_condition in conditions_present and comp.cue_condition in conditions_present:
                key = (model, dataset, comp.control_condition, comp.cue_condition)
                out.setdefault(key, set()).add(example_id)
    return out


def build_evidence_stratified_transitions(
    answer_records: list[AnswerRecord],
    strictly_paired: dict[tuple[str, str, str, str], set[str]],
    evidence_lookup: dict[tuple[str, str, str, str], dict],
) -> tuple[list[dict], dict[str, list[tuple[bool, bool]]]]:
    """Returns (per-comparison rows, pooled per-stratum (control_ok, cue_ok)
    pair lists) -- the pooled lists feed the top-line McNemar/Wilson stats
    in `write_notes` without needing a second pass over the answer records.
    """
    ans_by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in answer_records}
    rows: list[dict] = []
    pooled_pairs: dict[str, list[tuple[bool, bool]]] = {"cue_evidence_complete": [], "cue_evidence_incomplete": []}

    for comp in COMPARISONS:
        grouped: dict[tuple[str, str], set[str]] = {}
        for (model, dataset, control, cue), eids in strictly_paired.items():
            if control == comp.control_condition and cue == comp.cue_condition:
                grouped[(model, dataset)] = eids

        for (model, dataset), eids in grouped.items():
            pairs_by_stratum: dict[str, list[tuple[bool, bool]]] = {"cue_evidence_complete": [], "cue_evidence_incomplete": []}
            n_excluded_technical = 0
            n_excluded_no_evidence_data = 0
            for eid in eids:
                control_ans = ans_by_key.get((model, dataset, comp.control_condition, eid))
                cue_ans = ans_by_key.get((model, dataset, comp.cue_condition, eid))
                if control_ans is None or cue_ans is None:
                    continue
                if control_ans.extraction_reason in TECHNICAL_FAILURE_REASONS or cue_ans.extraction_reason in TECHNICAL_FAILURE_REASONS:
                    n_excluded_technical += 1
                    continue
                cue_evidence = evidence_lookup.get((model, dataset, comp.cue_condition, eid))
                if cue_evidence is None or cue_evidence["local_full_evidence"] is None:
                    n_excluded_no_evidence_data += 1
                    continue
                bucket_key = "cue_evidence_complete" if cue_evidence["local_full_evidence"] else "cue_evidence_incomplete"
                pair = (control_ans.semantic_correct, cue_ans.semantic_correct)
                pairs_by_stratum[bucket_key].append(pair)
                pooled_pairs[bucket_key].append(pair)

            for stratum, pairs in pairs_by_stratum.items():
                n = len(pairs)
                both_correct = sum(1 for a, b in pairs if a and b)
                control_correct_cue_wrong = sum(1 for a, b in pairs if a and not b)
                control_wrong_cue_correct = sum(1 for a, b in pairs if not a and b)
                both_wrong = n - both_correct - control_correct_cue_wrong - control_wrong_cue_correct
                net_harm = control_correct_cue_wrong - control_wrong_cue_correct
                if n > 0:
                    _stat, mcnemar_p, n_discordant = mcnemar_test([a for a, _ in pairs], [b for _, b in pairs])
                else:
                    mcnemar_p, n_discordant = None, 0
                rows.append(
                    {
                        "model": model, "dataset": dataset, "comparison_name": comp.comparison_name,
                        "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                        "cue_evidence_stratum": stratum, "n_pairs": n,
                        "both_correct": both_correct,
                        "control_correct_cue_wrong": control_correct_cue_wrong,
                        "control_wrong_cue_correct": control_wrong_cue_correct,
                        "both_wrong": both_wrong,
                        "net_harm": net_harm,
                        "net_harm_rate_pct": (100.0 * net_harm / n) if n else None,
                        "control_correct_cue_wrong_rate_pct": (100.0 * control_correct_cue_wrong / n) if n else None,
                        "mcnemar_n_discordant": n_discordant,
                        "mcnemar_p_value": mcnemar_p,
                        "n_excluded_technical_failure": n_excluded_technical,
                        "n_excluded_no_evidence_data": n_excluded_no_evidence_data,
                    }
                )
    return rows, pooled_pairs


def compute_pooled_stratum_stats(pooled_pairs: dict[str, list[tuple[bool, bool]]]) -> dict[str, dict]:
    """Pooled-level statistics per stratum: McNemar's test on the
    control-correct/cue-wrong vs control-wrong/cue-correct asymmetry WITHIN
    that stratum (is the cue's effect on answers significant, given this
    evidence-completeness state?), plus a Wilson CI on the "harmed rate"
    (control_correct_cue_wrong / n_pairs) for comparing the two strata by
    whether their CIs overlap -- the two strata are disjoint pair sets, so a
    paired test between them is not applicable; Wilson CI overlap is the
    defensible unpaired comparison here."""
    out: dict[str, dict] = {}
    for stratum, pairs in pooled_pairs.items():
        n = len(pairs)
        control_correct_cue_wrong = sum(1 for a, b in pairs if a and not b)
        control_wrong_cue_correct = sum(1 for a, b in pairs if not a and b)
        if n > 0:
            _stat, mcnemar_p, n_discordant = mcnemar_test([a for a, _ in pairs], [b for _, b in pairs])
            harmed_point, harmed_lo, harmed_hi = wilson_ci(control_correct_cue_wrong, n)
        else:
            mcnemar_p, n_discordant = None, 0
            harmed_point, harmed_lo, harmed_hi = 0.0, 0.0, 0.0
        out[stratum] = {
            "n_pairs": n,
            "control_correct_cue_wrong": control_correct_cue_wrong,
            "control_wrong_cue_correct": control_wrong_cue_correct,
            "mcnemar_n_discordant": n_discordant,
            "mcnemar_p_value": mcnemar_p,
            "harmed_rate": harmed_point,
            "harmed_rate_ci_lo": harmed_lo,
            "harmed_rate_ci_hi": harmed_hi,
        }
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_notes(path: Path, pooled_stats: dict[str, dict]) -> None:
    complete = pooled_stats["cue_evidence_complete"]
    incomplete = pooled_stats["cue_evidence_incomplete"]

    def fmt_pct(x):
        return f"{100.0 * x:.2f}%" if x is not None else "NA"

    def fmt_p(x):
        return f"{x:.4f}" if x is not None else "NA"

    lines = [
        "# Evidence-incompleteness vs paired answer harm -- notes",
        "",
        "Smallest justified next analysis after `EVIDENCE_HOP_RESULTS.md` "
        "(section 9). Tests whether pairs where the CUE condition was "
        "evidence-INCOMPLETE at local completion (`local_full_evidence == "
        "False` in `run_level_evidence_hops.csv`) show a higher "
        "control-correct/cue-wrong (\"harmed\") rate than pairs where the cue "
        "condition WAS evidence-complete, across every comparison in "
        "`prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS`, pooled "
        "over all datasets/models/comparisons.",
        "",
        "**This is an association check only, not a causal or thesis-ready "
        "claim** -- pairs are pooled across comparisons/datasets for this "
        "top-line number; see `evidence_incomplete_vs_answer_harm.csv` for "
        "the full per-comparison, per-model, per-dataset breakdown (each row "
        "also carries its own McNemar p-value) before citing anything.",
        "",
        "## Pooled top-line comparison",
        "",
        "Within-stratum McNemar's test asks: given this evidence-completeness "
        "state, is the cue's effect on answer correctness (control-correct/"
        "cue-wrong vs control-wrong/cue-correct) distinguishable from chance? "
        "The two strata are disjoint pair sets (not paired against each "
        "other), so comparing them directly uses non-overlap of the Wilson "
        "95% CI on the harmed rate, not a paired test.",
        "",
        "| Cue evidence at local completion | n_pairs | control_correct_cue_wrong | control_wrong_cue_correct | harmed rate (95% CI) | McNemar p (n_discordant) |",
        "|---|---|---|---|---|---|",
        f"| Complete | {complete['n_pairs']} | {complete['control_correct_cue_wrong']} | {complete['control_wrong_cue_correct']} | "
        f"{fmt_pct(complete['harmed_rate'])} ({fmt_pct(complete['harmed_rate_ci_lo'])}-{fmt_pct(complete['harmed_rate_ci_hi'])}) | "
        f"{fmt_p(complete['mcnemar_p_value'])} (n={complete['mcnemar_n_discordant']}) |",
        f"| Incomplete | {incomplete['n_pairs']} | {incomplete['control_correct_cue_wrong']} | {incomplete['control_wrong_cue_correct']} | "
        f"{fmt_pct(incomplete['harmed_rate'])} ({fmt_pct(incomplete['harmed_rate_ci_lo'])}-{fmt_pct(incomplete['harmed_rate_ci_hi'])}) | "
        f"{fmt_p(incomplete['mcnemar_p_value'])} (n={incomplete['mcnemar_n_discordant']}) |",
        "",
    ]

    ci_overlap = not (complete["harmed_rate_ci_hi"] < incomplete["harmed_rate_ci_lo"] or incomplete["harmed_rate_ci_hi"] < complete["harmed_rate_ci_lo"])
    direction = "higher than" if incomplete["harmed_rate"] > complete["harmed_rate"] else ("lower than" if incomplete["harmed_rate"] < complete["harmed_rate"] else "the same as")
    lines.append(
        f"The pooled harmed rate among evidence-incomplete cue completions is "
        f"**{direction}** the rate among evidence-complete cue completions "
        f"({fmt_pct(incomplete['harmed_rate'])} vs {fmt_pct(complete['harmed_rate'])}). "
        f"Their 95% Wilson CIs **{'overlap' if ci_overlap else 'do not overlap'}**"
        + (
            " -- this pooled difference is not distinguishable from noise at "
            "this level of aggregation."
            if ci_overlap
            else " -- the two strata's harmed rates are distinguishable at "
            "the pooled level, though this still pools across heterogeneous "
            "comparisons/datasets/models and is not a per-comparison claim."
        )
    )
    for stats, label in ((complete, "evidence-complete"), (incomplete, "evidence-incomplete")):
        if stats["mcnemar_p_value"] is not None and stats["mcnemar_p_value"] < 0.05:
            lines.append(
                f"\n\nWithin the {label} stratum, the cue's effect on paired "
                f"answer correctness is itself significant by McNemar's test "
                f"(p={fmt_p(stats['mcnemar_p_value'])}, n_discordant={stats['mcnemar_n_discordant']})."
            )
    lines += [
        "",
        "## Known limitations",
        "",
        "- Pairs are pooled across all comparisons/datasets/models for the "
        "top-line table above; the CSV has the disaggregated view, each with "
        "its own McNemar p-value (small per-cell n means most individual "
        "rows will not reach significance -- use the pooled numbers or "
        "aggregate deliberately, not a single row's p-value, for any claim).",
        "- Answer correctness is pre-adjudication (automatic semantic label), "
        "same caveat as `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s answer tables.",
        "- Technical-failure and no-evidence-data pairs are excluded, not "
        "imputed (counts in the CSV's `n_excluded_*` columns).",
        "- This does not establish that evidence loss CAUSES the answer flip "
        "-- only whether the two co-occur more than chance-pooled would "
        "suggest, and whether that co-occurrence itself looks like chance "
        "within each stratum.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = RESULTS_E1 / "evidence_hop_coverage"
    run_level_path = output_dir / "run_level_evidence_hops.csv"

    print("Loading evidence-hop lookup from run_level_evidence_hops.csv...")
    evidence_lookup = load_evidence_lookup(run_level_path)
    print(f"  {len(evidence_lookup)} rows")

    print("Building inventory + answer records (valid rows only)...")
    inventory = build_inventory(config.DEFAULT_PRODUCTION_ROOT, models=CANONICAL_BUNDLE_MODELS)
    valid_rows = [r for r in inventory.rows if _row_is_valid(r)]
    answer_records = build_answer_records(valid_rows, config.DEFAULT_PRODUCTION_ROOT, max_workers=8)
    print(f"  {len(answer_records)} answer records")

    print("Building strictly-paired example sets...")
    strictly_paired = build_strictly_paired_example_ids(answer_records)

    print("Building evidence-stratified paired transitions...")
    rows, pooled_pairs = build_evidence_stratified_transitions(answer_records, strictly_paired, evidence_lookup)
    write_csv(rows, output_dir / "evidence_incomplete_vs_answer_harm.csv")

    print("Computing pooled McNemar/Wilson statistics...")
    pooled_stats = compute_pooled_stratum_stats(pooled_pairs)

    print("Writing notes...")
    write_notes(output_dir / "EVIDENCE_VS_ANSWER_HARM_NOTES.md", pooled_stats)

    print(f"\nDone. {len(rows)} (comparison x model x dataset x stratum) rows written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
