"""Section 9: earlier completion -> evidence loss, on strict pairs only.

For every strictly-paired (baseline, attack) example (from `pairing.py`'s
audit -- never inferred, never run on unmatched rows), classifies the pair
by whether the attack's local-completion step was earlier, the same, later,
or not comparable to the baseline's, then reports within each stratum
whether the attack shows less/equal/more episode-terminal evidence, whether
full-hop coverage flipped, and whether the supervisor recovered a resulting
loss by the final-input boundary.

This is a paired PATHWAY analysis, not a causal-mediation claim -- reported
as an observed association per the task's explicit instruction.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

from analysis.evidence import EvidenceRecord
from analysis.pcr import PCRRecord

STRATUM_EARLIER = "attack_completed_earlier"
STRATUM_SAME = "same_completion_step"
STRATUM_LATER = "attack_completed_later"
STRATUM_INCOMPARABLE = "incomparable_or_missing_completion"


def _key(model, dataset, condition, example_id):
    return (model, dataset, condition, example_id)


def build_earlier_completion_to_evidence_loss(
    pcr_records: list[PCRRecord],
    evidence_records: list[EvidenceRecord],
    strictly_paired_example_ids: dict[tuple, set],
    comparisons: list[tuple[str, str]],
) -> list[dict]:
    pcr_by_key = {_key(r.model, r.dataset, r.condition, r.example_id): r for r in pcr_records}
    ev_by_key = {_key(r.model, r.dataset, r.condition, r.example_id): r for r in evidence_records}

    rows = []
    grouped_keys: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in strictly_paired_example_ids:
        model, dataset, baseline, cond = key
        grouped_keys[(model, dataset)].append((baseline, cond))

    for (model, dataset), pairs in grouped_keys.items():
        for baseline, cond in pairs:
            if (baseline, cond) not in comparisons:
                continue
            eids = strictly_paired_example_ids[(model, dataset, baseline, cond)]
            for eid in eids:
                p_base = pcr_by_key.get(_key(model, dataset, baseline, eid))
                p_att = pcr_by_key.get(_key(model, dataset, cond, eid))
                e_base = ev_by_key.get(_key(model, dataset, baseline, eid))
                e_att = ev_by_key.get(_key(model, dataset, cond, eid))
                if not (p_base and p_att and e_base and e_att):
                    continue

                base_step, att_step = p_base.terminal_post_branch_index, p_att.terminal_post_branch_index
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
                search_delta = p_att.post_injection_search_count - p_base.post_injection_search_count

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

                recovered = (
                    final_delta is not None and ep0_delta is not None and final_delta > ep0_delta + 1e-9
                )

                rows.append(
                    {
                        "model": model, "dataset": dataset, "condition_a": baseline, "condition_b": cond, "example_id": eid,
                        "stratum": stratum,
                        "baseline_completion_step": base_step, "attack_completion_step": att_step,
                        "post_injection_search_count_delta": search_delta,
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
        grouped[(r["model"], r["dataset"], r["condition_a"], r["condition_b"], r["stratum"])].append(r)

    summary = []
    for (model, dataset, a, b, stratum), rs in grouped.items():
        n = len(rs)
        n_less = sum(1 for r in rs if r["attack_has_less_evidence_at_episode0_terminal"] is True)
        n_more = sum(1 for r in rs if r["attack_has_more_evidence_at_episode0_terminal"] is True)
        n_equal = n - n_less - n_more
        n_recovered = sum(1 for r in rs if r["supervisor_recovery_narrowed_gap"] is True)
        summary.append(
            {
                "model": model, "dataset": dataset, "condition_a": a, "condition_b": b, "stratum": stratum, "n": n,
                "n_attack_less_evidence_ep0": n_less, "pct_attack_less_evidence_ep0": 100.0 * n_less / n if n else None,
                "n_attack_more_evidence_ep0": n_more, "pct_attack_more_evidence_ep0": 100.0 * n_more / n if n else None,
                "n_equal_evidence_ep0": n_equal,
                "n_supervisor_recovery_narrowed_gap": n_recovered, "pct_supervisor_recovery_narrowed_gap": 100.0 * n_recovered / n if n else None,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
