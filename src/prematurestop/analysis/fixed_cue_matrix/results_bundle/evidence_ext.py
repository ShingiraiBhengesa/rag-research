"""Step 8: `data/evidence_run_level.csv` and `data/evidence_summary.csv`.

Reuses `prematurestop.evaluation.evidence.build_evidence_records` unchanged. Adds the two
columns the spec asks for that `src/prematurestop/evaluation/evidence.py` doesn't compute:
evidence deficit (1 - final-input coverage) and paired evidence loss
(control coverage - cue coverage, spec 3.2's literal wording -- the
*opposite* sign direction from `pcr_paired_deltas_stepwise.csv`'s
cue-minus-control convention; see MASTER_RESULTS.md's methodology section
for why both conventions independently mean "positive = attack made it
worse"). Deliberately does NOT reuse
`src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation/mechanism.py`'s `ep0_delta`/`final_delta`
(attack minus baseline -- inverted sign vs. this spec).

Evidence units: this repo's `evidence.py` already normalizes all 4 datasets'
paragraph identifiers into one comparable per-example index space (see its
own docstring on `_paragraph_index`) before computing a single pooled
`coverage_fraction` -- there is no separate document/supporting-fact/hop
count preserved beyond `required_dependency_count` (the same "required
paragraph" unit for every dataset in this study; there is no separate
"supporting-fact" annotation layer available). `evidence_run_level.csv`'s
"gold-document recall"/"supporting-fact recall where available" columns are
therefore both populated from the same coverage-fraction computation --
documented explicitly as an equivalence, not silently conflated.
"""

from __future__ import annotations

from pathlib import Path

from prematurestop.evaluation import completion as cc
from prematurestop.evaluation.evidence import EvidenceRecord
from prematurestop.stats.metrics import paired_bootstrap_ci
from prematurestop.evaluation import config

from .comparisons import COMPARISONS


def write_evidence_run_level_csv(
    records: list[EvidenceRecord],
    recovery_by_key: dict[tuple[str, str, str, str], str],
    path: Path,
) -> None:
    rows = []
    for r in records:
        key = (r.model, r.dataset, r.condition, r.example_id)
        recovery_label = recovery_by_key.get(key, "unknown")
        deficit = (1.0 - r.final_input_coverage_fraction) if r.final_input_coverage_fraction is not None else None
        evidence_incomplete_completion = (
            (r.final_input_coverage_fraction is not None and r.final_input_coverage_fraction < 1.0)
        )
        rows.append(
            {
                "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                "required_dependency_count": r.required_dependency_count,
                "local_completion_coverage": r.episode0_terminal_coverage_fraction,
                "final_run_coverage": r.final_input_coverage_fraction,
                "evidence_deficit": deficit,
                "missing_hop_count": r.final_input_missing_count,
                "full_hop_indicator": r.full_hop_coverage_at_final_input,
                # See module docstring: both populated from the same
                # normalized-paragraph coverage computation -- this repo has
                # no separate document-vs-supporting-fact annotation layer.
                "gold_document_recall": r.final_input_coverage_fraction,
                "supporting_fact_recall_where_available": r.final_input_coverage_fraction,
                "evidence_incomplete_completion_conditioned_on_completion": evidence_incomplete_completion,
                "evidence_incomplete_unconditional_eligible_denominator": evidence_incomplete_completion,
                "recovery_after_local_completion": recovery_label,
            }
        )
    cc._write_csv(path, rows)  # noqa: SLF001


def write_evidence_summary_csv(
    canonical_records: list[cc.CanonicalCompletionRecord],
    evidence_records: list[EvidenceRecord],
    path: Path,
) -> None:
    ev_by_key: dict[tuple[str, str, str, str], EvidenceRecord] = {
        (r.model, r.dataset, r.condition, r.example_id): r for r in evidence_records
    }
    grouped = cc._group(canonical_records, lambda r: (r.model, r.dataset))  # noqa: SLF001
    rows: list[dict] = []
    for (model, dataset), recs in sorted(grouped.items()):
        for comp in COMPARISONS:
            pairs = cc.build_strict_pairs(recs, comp.cue_condition, comp.control_condition)
            control_vals, cue_vals = [], []
            for cue_rec, control_rec in pairs:
                cue_ev = ev_by_key.get((model, dataset, cue_rec.condition, cue_rec.example_id))
                control_ev = ev_by_key.get((model, dataset, control_rec.condition, control_rec.example_id))
                if cue_ev is None or control_ev is None:
                    continue
                if cue_ev.final_input_coverage_fraction is None or control_ev.final_input_coverage_fraction is None:
                    continue
                cue_vals.append(cue_ev.final_input_coverage_fraction)
                control_vals.append(control_ev.final_input_coverage_fraction)
            n = len(control_vals)
            if n == 0:
                rows.append(
                    {
                        "model": model, "dataset": dataset, "comparison_name": comp.comparison_name,
                        "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                        "n_paired": 0, "control_mean_coverage": None, "cue_mean_coverage": None,
                        "paired_evidence_loss": None, "ci_lower": None, "ci_upper": None,
                    }
                )
                continue
            # Spec 3.2 literal wording: control minus cue.
            point, lo, hi, _p = paired_bootstrap_ci(control_vals, cue_vals, seed=config.DETERMINISTIC_SEED)
            rows.append(
                {
                    "model": model, "dataset": dataset, "comparison_name": comp.comparison_name,
                    "control_condition": comp.control_condition, "cue_condition": comp.cue_condition,
                    "n_paired": n,
                    "control_mean_coverage": sum(control_vals) / n,
                    "cue_mean_coverage": sum(cue_vals) / n,
                    "paired_evidence_loss": point,
                    "ci_lower": lo, "ci_upper": hi,
                }
            )
    cc._write_csv(path, rows)  # noqa: SLF001
