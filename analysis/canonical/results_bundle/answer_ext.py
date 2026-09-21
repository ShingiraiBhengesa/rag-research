"""Steps 9-10: `data/answer_run_level.csv`, `data/answer_summary.csv`,
`data/answer_paired_flips.csv`.

Reuses `analysis.answer_evaluation.build_answer_records`/`.summarize_by_cell`/
`.compute_paired_answer_transitions` entirely unchanged -- the last one
already produces almost exactly `answer_paired_flips.csv`'s required shape.
Adds `needs_adjudication_flag` (spec 5.8's adjudication column, derived from
`semantic_label == "needs_adjudication"`) and re-fetches `gold_aliases` via
the existing manifest loader (`AnswerRecord` itself doesn't retain aliases).

Pairing for the answer-flip analysis uses simple (model, dataset, example_id)
presence in both conditions -- NOT gated on PCR eligibility, since answer
correctness is a downstream measure independent of the researcher-episode
completion-timing question (spec section 2's "Unit of pairing: same model,
dataset, and benchmark example" makes no PCR-eligibility requirement).
`compute_paired_answer_transitions`'s own technical-failure filtering
(`no_manifest_entry`/`no_final_report`) still excludes rows with no usable
answer.

Step 10 (completing the blinded adjudication review) is a human process --
this module marks the flag and reports counts/denominators; it cannot
"complete" adjudication.
"""

from __future__ import annotations

from pathlib import Path

from analysis.answer_evaluation import AnswerRecord, compute_paired_answer_transitions, load_manifest, summarize_by_cell
from analysis.canonical import canonical_completion as cc

from .comparisons import COMPARISONS


def _build_strictly_paired_example_ids(
    records: list[AnswerRecord],
) -> dict[tuple[str, str, str, str], set[str]]:
    """(model, dataset, control_condition, cue_condition) -> set of example_ids
    present under both conditions -- existence-based pairing (see module
    docstring for why this is NOT gated on PCR eligibility)."""
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


def write_answer_run_level_csv(records: list[AnswerRecord], production_root: Path, path: Path) -> None:
    rows = []
    for r in records:
        manifest = load_manifest(r.dataset, production_root)
        entry = manifest.get(r.example_id)
        rows.append(
            {
                "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                "gold_answer": r.gold_answer,
                "gold_aliases": ";".join(entry.answer_aliases) if entry else "",
                "raw_final_report": r.raw_final_report,
                "extracted_short_answer": r.extracted_short_answer,
                "extraction_reason": r.extraction_reason,
                "automatic_semantic_label": r.semantic_label,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
                "gold_containment_diagnostic_only": r.gold_containment,
                "needs_adjudication_flag": r.semantic_label == "needs_adjudication",
            }
        )
    cc._write_csv(path, rows)  # noqa: SLF001


def write_answer_summary_csv(records: list[AnswerRecord], path: Path) -> None:
    cc._write_csv(path, summarize_by_cell(records))  # noqa: SLF001


def write_answer_paired_flips_csv(records: list[AnswerRecord], path: Path) -> None:
    strictly_paired = _build_strictly_paired_example_ids(records)
    comparisons_as_tuples = [(c.control_condition, c.cue_condition) for c in COMPARISONS]
    rows = compute_paired_answer_transitions(records, strictly_paired, comparisons_as_tuples)
    cc._write_csv(path, rows)  # noqa: SLF001
