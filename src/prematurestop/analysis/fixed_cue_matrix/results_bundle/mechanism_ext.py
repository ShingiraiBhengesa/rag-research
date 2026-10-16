"""Supports the paired trace-raster figures (spec section 8): earlier
completion -> evidence loss, ported from
`src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation/mechanism.py` to canonical field names.

`mechanism.py`'s own logic is a paired-pathway comparison keyed off
`PCRRecord.terminal_post_branch_index` (legacy) and
`EvidenceRecord.episode0_terminal_coverage_fraction`/`final_input_coverage_fraction`
(unchanged, canonical-compatible). The completion-step field name is the
only thing that differs: `CanonicalCompletionRecord.first_terminal_post_observation_step`
(T_i) plays the same role as `PCRRecord.terminal_post_branch_index`, and
`post_observation_search_calls` plays the same role as
`post_injection_search_count`. Everything else below -- stratum
definitions, delta/recovery math -- is copied unchanged from
`mechanism.py:63-114`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

from prematurestop.evaluation import completion as cc
from prematurestop.evaluation.evidence import EvidenceRecord

STRATUM_EARLIER = "attack_completed_earlier"
STRATUM_SAME = "same_completion_step"
STRATUM_LATER = "attack_completed_later"
STRATUM_INCOMPARABLE = "incomparable_or_missing_completion"


def _key(model, dataset, condition, example_id):
    return (model, dataset, condition, example_id)


def build_earlier_completion_to_evidence_loss(
    canonical_records: list[cc.CanonicalCompletionRecord],
    evidence_records: list[EvidenceRecord],
    comparisons,
) -> list[dict]:
    """`comparisons` is `results_bundle.comparisons.COMPARISONS` -- iterated
    directly instead of a separately-built strictly-paired-example-id dict,
    using `cc.build_strict_pairs` per comparison (same pairing rule used
    everywhere else in this bundle)."""
    cc_by_key = {_key(r.model, r.dataset, r.condition, r.example_id): r for r in canonical_records}
    ev_by_key = {_key(r.model, r.dataset, r.condition, r.example_id): r for r in evidence_records}

    grouped = cc._group(canonical_records, lambda r: (r.model, r.dataset))  # noqa: SLF001
    rows = []
    for (model, dataset), recs in grouped.items():
        for comp in comparisons:
            pairs = cc.build_strict_pairs(recs, comp.cue_condition, comp.control_condition)
            for att_rec, base_rec in pairs:
                eid = att_rec.example_id
                e_base = ev_by_key.get(_key(model, dataset, comp.control_condition, eid))
                e_att = ev_by_key.get(_key(model, dataset, comp.cue_condition, eid))
                if not (e_base and e_att):
                    continue

                base_step = base_rec.first_terminal_post_observation_step
                att_step = att_rec.first_terminal_post_observation_step
                if base_step is None or att_step is None:
                    stratum = STRATUM_INCOMPARABLE
                elif att_step < base_step:
                    stratum = STRATUM_EARLIER
                elif att_step == base_step:
                    stratum = STRATUM_SAME
                else:
                    stratum = STRATUM_LATER

                def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                    return (a - b) if (a is not None and b is not None) else None

                ep0_delta = delta(e_att.episode0_terminal_coverage_fraction, e_base.episode0_terminal_coverage_fraction)
                final_delta = delta(e_att.final_input_coverage_fraction, e_base.final_input_coverage_fraction)
                search_delta = att_rec.post_observation_search_calls - base_rec.post_observation_search_calls

                fh_base_ep0 = e_base.full_hop_coverage_at_episode0_terminal
                fh_att_ep0 = e_att.full_hop_coverage_at_episode0_terminal
                fh_base_final = e_base.full_hop_coverage_at_final_input
                fh_att_final = e_att.full_hop_coverage_at_final_input

                attack_less_evidence_ep0 = (
                    e_att.episode0_terminal_coverage_fraction < e_base.episode0_terminal_coverage_fraction
                    if (e_att.episode0_terminal_coverage_fraction is not None and e_base.episode0_terminal_coverage_fraction is not None)
                    else None
                )
                attack_more_evidence_ep0 = (
                    e_att.episode0_terminal_coverage_fraction > e_base.episode0_terminal_coverage_fraction
                    if (e_att.episode0_terminal_coverage_fraction is not None and e_base.episode0_terminal_coverage_fraction is not None)
                    else None
                )

                recovered = final_delta is not None and ep0_delta is not None and final_delta > ep0_delta + 1e-9

                rows.append(
                    {
                        "model": model, "dataset": dataset,
                        "comparison_name": comp.comparison_name,
                        "condition_a": comp.control_condition, "condition_b": comp.cue_condition,
                        "example_id": eid,
                        "stratum": stratum,
                        "baseline_completion_step": base_step, "attack_completion_step": att_step,
                        "post_observation_search_calls_delta": search_delta,
                        "episode0_terminal_coverage_delta": ep0_delta,
                        "final_input_coverage_delta": final_delta,
                        "full_hop_at_episode0_terminal_base": fh_base_ep0, "full_hop_at_episode0_terminal_attack": fh_att_ep0,
                        "full_hop_at_final_input_base": fh_base_final, "full_hop_at_final_input_attack": fh_att_final,
                        "attack_has_less_evidence_at_episode0_terminal": attack_less_evidence_ep0,
                        "attack_has_more_evidence_at_episode0_terminal": attack_more_evidence_ep0,
                        "supervisor_recovery_narrowed_gap": recovered,
                    }
                )
    return rows


def summarize_by_stratum(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dataset"], r["comparison_name"], r["stratum"])].append(r)

    summary = []
    for (model, dataset, comparison_name, stratum), rs in grouped.items():
        n = len(rs)
        n_less = sum(1 for r in rs if r["attack_has_less_evidence_at_episode0_terminal"] is True)
        n_more = sum(1 for r in rs if r["attack_has_more_evidence_at_episode0_terminal"] is True)
        n_equal = n - n_less - n_more
        n_recovered = sum(1 for r in rs if r["supervisor_recovery_narrowed_gap"] is True)
        summary.append(
            {
                "model": model, "dataset": dataset, "comparison_name": comparison_name, "stratum": stratum, "n": n,
                "n_attack_less_evidence_ep0": n_less, "pct_attack_less_evidence_ep0": 100.0 * n_less / n if n else None,
                "n_attack_more_evidence_ep0": n_more, "pct_attack_more_evidence_ep0": 100.0 * n_more / n if n else None,
                "n_equal_evidence_ep0": n_equal,
                "n_supervisor_recovery_narrowed_gap": n_recovered, "pct_supervisor_recovery_narrowed_gap": 100.0 * n_recovered / n if n else None,
            }
        )
    return summary


def write_mechanism_csvs(canonical_records, evidence_records, comparisons, run_level_path: Path, summary_path: Path) -> None:
    rows = build_earlier_completion_to_evidence_loss(canonical_records, evidence_records, comparisons)
    cc._write_csv(run_level_path, rows)  # noqa: SLF001
    cc._write_csv(summary_path, summarize_by_stratum(rows))  # noqa: SLF001
