"""Section 11: evidence loss -> answer harm, and the final mechanism-chain summary.

Built from `master.MasterRow` (evidence boundaries and answer scores from
the SAME single-pass parse, so there is no cross-dataset join risk here),
restricted to strict pairs only (`pairing.py`'s audit), and reusing the
existing validated significance helpers (`attack.metrics`) rather
than re-implementing McNemar/bootstrap.

Evidence-loss strata per pair (based on the boundary that actually feeds
the answer, final-report input):

- no_evidence_loss: attack's final-input coverage >= baseline's
- episode_terminal_loss_then_recovered: attack's episode-0 coverage was
  lower than baseline's, but final-input coverage caught back up
- unrecovered_final_input_loss: attack's final-input coverage is still
  lower than baseline's

Two additional non-exclusive flags are reported alongside every row for
finer-grained slicing, not as separate primary strata (they can co-occur
with any of the three above): `full_hop_control_to_missing_hop_attack`
(baseline had full-hop coverage at final input, attack did not) and
`compression_only_loss` (episode-0 coverage was NOT lower for the attack,
but compressed-notes coverage was -- evidence found fine, lost specifically
during compression).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from attack.metrics import mcnemar_test, paired_bootstrap_ci, wilson_ci
from analysis import config
from analysis.final_paper_evaluation.master import MasterRow

STRATUM_NO_LOSS = "no_evidence_loss"
STRATUM_RECOVERED = "episode_terminal_loss_then_recovered"
STRATUM_UNRECOVERED = "unrecovered_final_input_loss"


def _key(row: MasterRow) -> tuple:
    return (row.model, row.dataset, row.condition, row.example_id)


def classify_pair_evidence_loss(baseline: MasterRow, attack: MasterRow) -> dict:
    ep0_b, ep0_a = baseline.episode0_terminal_coverage_fraction, attack.episode0_terminal_coverage_fraction
    fin_b, fin_a = baseline.final_input_coverage_fraction, attack.final_input_coverage_fraction
    comp_b, comp_a = baseline.compressed_notes_coverage_fraction, attack.compressed_notes_coverage_fraction

    if fin_b is None or fin_a is None or ep0_b is None or ep0_a is None:
        stratum = "unresolved_missing_evidence_state"
    else:
        early_loss = ep0_a < ep0_b - 1e-9
        persistent_loss = fin_a < fin_b - 1e-9
        if persistent_loss:
            stratum = STRATUM_UNRECOVERED
        elif early_loss:
            stratum = STRATUM_RECOVERED
        else:
            stratum = STRATUM_NO_LOSS

    full_hop_flip = None
    if fin_b is not None and fin_a is not None:
        full_hop_flip = (fin_b >= 1.0) and (fin_a < 1.0)

    compression_only_loss = None
    if ep0_b is not None and ep0_a is not None and comp_b is not None and comp_a is not None:
        compression_only_loss = (ep0_a >= ep0_b - 1e-9) and (comp_a < comp_b - 1e-9)

    return {
        "evidence_loss_stratum": stratum,
        "full_hop_control_to_missing_hop_attack": full_hop_flip,
        "compression_only_loss": compression_only_loss,
    }


def build_paired_answer_harm_rows(
    master_rows: list[MasterRow],
    strictly_paired_example_ids: dict[tuple, set],
    comparisons: list[tuple[str, str]],
) -> list[dict]:
    by_key = {_key(r): r for r in master_rows}
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
                b = by_key.get((model, dataset, baseline, eid))
                a = by_key.get((model, dataset, cond, eid))
                if b is None or a is None:
                    continue
                if b.extraction_reason in ("no_manifest_entry", "no_final_report") or a.extraction_reason in (
                    "no_manifest_entry", "no_final_report",
                ):
                    continue  # technical failure excluded from primary answer-harm accounting
                loss_info = classify_pair_evidence_loss(b, a)
                rows.append(
                    {
                        "model": model, "dataset": dataset, "condition_a": baseline, "condition_b": cond, "example_id": eid,
                        "control_correct": b.semantic_correct, "attack_correct": a.semantic_correct,
                        "control_em": b.exact_match, "attack_em": a.exact_match,
                        "control_f1": b.token_f1, "attack_f1": a.token_f1,
                        **loss_info,
                    }
                )
    return rows


def _paired_stats_block(rows: list[dict], seed: int = config.DETERMINISTIC_SEED) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    control_correct = [r["control_correct"] for r in rows]
    attack_correct = [r["attack_correct"] for r in rows]
    both_correct = sum(1 for c, a in zip(control_correct, attack_correct) if c and a)
    control_correct_attack_wrong = sum(1 for c, a in zip(control_correct, attack_correct) if c and not a)
    control_wrong_attack_correct = sum(1 for c, a in zip(control_correct, attack_correct) if not c and a)
    both_wrong = sum(1 for c, a in zip(control_correct, attack_correct) if not c and not a)
    net_harm = control_correct_attack_wrong - control_wrong_attack_correct

    mc_stat, mc_p, mc_n_disc = mcnemar_test(control_correct, attack_correct)
    acc_point, acc_lo, acc_hi, acc_p = paired_bootstrap_ci(
        [float(x) for x in attack_correct], [float(x) for x in control_correct], seed=seed
    )
    f1_point, f1_lo, f1_hi, f1_p = paired_bootstrap_ci(
        [r["attack_f1"] for r in rows], [r["control_f1"] for r in rows], seed=seed
    )

    control_acc, _, _ = wilson_ci(sum(control_correct), n)
    attack_acc, _, _ = wilson_ci(sum(attack_correct), n)

    return {
        "n": n,
        "both_correct": both_correct,
        "control_correct_attack_wrong": control_correct_attack_wrong,
        "control_wrong_attack_correct": control_wrong_attack_correct,
        "both_wrong": both_wrong,
        "net_harm": net_harm,
        "control_semantic_accuracy_pct": 100.0 * control_acc,
        "attack_semantic_accuracy_pct": 100.0 * attack_acc,
        "semantic_accuracy_diff_pct_points": 100.0 * (attack_acc - control_acc),
        "mcnemar_stat": mc_stat, "mcnemar_p_value": mc_p, "mcnemar_n_discordant": mc_n_disc,
        "bootstrap_accuracy_diff_ci_lo_pct": 100.0 * acc_lo, "bootstrap_accuracy_diff_ci_hi_pct": 100.0 * acc_hi,
        "bootstrap_accuracy_p_value": acc_p,
        "control_em_mean": sum(r["control_em"] for r in rows) / n, "attack_em_mean": sum(r["attack_em"] for r in rows) / n,
        "em_diff": sum(r["attack_em"] for r in rows) / n - sum(r["control_em"] for r in rows) / n,
        "control_f1_mean": sum(r["control_f1"] for r in rows) / n, "attack_f1_mean": sum(r["attack_f1"] for r in rows) / n,
        "f1_diff": f1_point, "f1_bootstrap_ci_lo": f1_lo, "f1_bootstrap_ci_hi": f1_hi, "f1_bootstrap_p_value": f1_p,
    }


def build_correctness_transitions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dataset"], r["condition_a"], r["condition_b"])].append(r)
    out = []
    for (model, dataset, a, b), rs in grouped.items():
        stats_block = _paired_stats_block(rs)
        out.append({"model": model, "dataset": dataset, "condition_a": a, "condition_b": b, **stats_block})
    return out


def build_evidence_loss_to_answer_harm(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dataset"], r["condition_a"], r["condition_b"], r["evidence_loss_stratum"])].append(r)
    out = []
    for (model, dataset, a, b, stratum), rs in grouped.items():
        stats_block = _paired_stats_block(rs)
        out.append({"model": model, "dataset": dataset, "condition_a": a, "condition_b": b, "evidence_loss_stratum": stratum, **stats_block})
    return out


def build_mechanism_chain_summary(
    earlier_completion_rows: list[dict], answer_harm_rows: list[dict]
) -> list[dict]:
    """Counts, per (model, dataset, condition_a, condition_b), of:
    earlier-completion x evidence-loss x answer-harm combinations, matching
    the required mechanism-pathway categories.
    """
    ec_by_pair_example: dict[tuple, dict] = {
        (r["model"], r["dataset"], r["condition_a"], r["condition_b"], r["example_id"]): r
        for r in earlier_completion_rows
    }
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in answer_harm_rows:
        key = (r["model"], r["dataset"], r["condition_a"], r["condition_b"])
        ec = ec_by_pair_example.get((*key, r["example_id"]))
        stratum_ec = ec["stratum"] if ec else "unknown"
        grouped[key].append({**r, "completion_stratum": stratum_ec})

    out = []
    for key, rs in grouped.items():
        model, dataset, a, b = key
        n = len(rs)

        def count(pred) -> int:
            return sum(1 for r in rs if pred(r))

        earlier = lambda r: r["completion_stratum"] == "attack_completed_earlier"
        harmed = lambda r: r["control_correct"] and not r["attack_correct"]
        loss = lambda r: r["evidence_loss_stratum"] == STRATUM_UNRECOVERED
        recovered = lambda r: r["evidence_loss_stratum"] == STRATUM_RECOVERED
        no_loss = lambda r: r["evidence_loss_stratum"] == STRATUM_NO_LOSS
        unresolved = lambda r: r["evidence_loss_stratum"] == "unresolved_missing_evidence_state"

        out.append(
            {
                "model": model, "dataset": dataset, "condition_a": a, "condition_b": b, "n": n,
                "earlier_completion_and_unrecovered_loss_and_harm": count(lambda r: earlier(r) and loss(r) and harmed(r)),
                "earlier_completion_and_unrecovered_loss_and_no_harm": count(lambda r: earlier(r) and loss(r) and not harmed(r)),
                "earlier_completion_and_no_persistent_loss": count(lambda r: earlier(r) and not loss(r)),
                "no_earlier_completion_and_unrecovered_loss": count(lambda r: not earlier(r) and loss(r)),
                "evidence_restored_by_supervisor": count(recovered),
                "no_evidence_loss_at_all": count(no_loss),
                "unresolved_missing_state": count(unresolved),
            }
        )
    return out


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
