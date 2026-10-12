"""Pilot analysis: PCR curves, paired contrasts, outcome tables, figures, cue audit.

PCR@k = (# pcr_eligible runs completing via a numerator route with T_i <= k)
/ (# pcr_eligible runs) -- the canonical definition, computed from labels
produced by outcomes.label_run(). Paired contrasts use the canonical bundle's
bootstrap (same iterations, same seed) on the strict paired set: both arms ran
and both are pcr_eligible. Questions where a bridge arm could not run
(NO_ANCHOR / INVALID_CUE / DECOMPOSE_INVALID) are reported in the coverage
table, never silently dropped; per-arm PCR is reported over each arm's own
runs, and every contrast states its paired n.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from analysis import config as analysis_config
from analysis.canonical.canonical_completion import PCR_NUMERATOR_ROUTES
from analysis.canonical.results_bundle.paired_deltas import STEPWISE_BOOTSTRAP_ITERATIONS, _bootstrap_all_steps
from attack.dynamic_bridge_pilot import config as C

CONTRASTS = (
    # (name, cue arm, control arm, tier, question it answers)
    ("structure", "bridge_sufficiency_k3", "entity_sufficiency", "primary", "does relational structure matter?"),
    ("truth", "bridge_sufficiency_k3", "unsupported_sufficiency", "primary", "does grounding in a true fact matter?"),
    ("selection", "bridge_sufficiency_k3", "bridge_sufficiency_k1", "primary", "is the effect driven by best-of-K selection?"),
    ("bridge_k3_vs_fixed_sufficiency", "bridge_sufficiency_k3", "fixed_sufficiency", "secondary", "dynamic vs fixed (Suff.)"),
    ("bridge_sat_k3_vs_fixed_saturation", "bridge_saturation_k3", "fixed_saturation", "secondary", "dynamic vs fixed (Sat.)"),
    ("bridge_sat_k3_vs_bridge_suff_k3", "bridge_saturation_k3", "bridge_sufficiency_k3", "secondary", "mechanism, dynamic cues"),
    ("fixed_sufficiency_vs_neutral", "fixed_sufficiency", "neutral", "secondary", "replicates canonical fixed-cue effect"),
    ("fixed_saturation_vs_neutral", "fixed_saturation", "neutral", "secondary", "replicates canonical fixed-cue effect"),
    ("bridge_k3_vs_neutral", "bridge_sufficiency_k3", "neutral", "secondary", "total effect"),
    ("bridge_k1_vs_neutral", "bridge_sufficiency_k1", "neutral", "secondary", "total effect"),
    ("entity_vs_neutral", "entity_sufficiency", "neutral", "secondary", "total effect"),
    ("unsupported_vs_neutral", "unsupported_sufficiency", "neutral", "secondary", "total effect"),
    ("bridge_sat_k3_vs_neutral", "bridge_saturation_k3", "neutral", "secondary", "total effect"),
)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def binary_outcome_contrasts(labels: list[dict], outcome, eligible, outcome_name: str) -> list[dict]:
    """Paired binary outcome per contrast: both arms eligible on the same question.
    Rates, paired difference (cue - control) with bootstrap CI, discordant counts,
    exact McNemar p."""
    by: dict = defaultdict(dict)
    for lab in labels:
        if eligible(lab):
            by[(lab["victim"], lab["dataset"], lab["arm"])][lab["example_id"]] = bool(outcome(lab))
    rows = []
    for victim, dataset in sorted({(v, d) for v, d, _ in by}):
        for name, cue, ctl, tier, meaning in CONTRASTS:
            a, b = by.get((victim, dataset, cue), {}), by.get((victim, dataset, ctl), {})
            ids = sorted(a.keys() & b.keys())
            row = {"victim": victim, "dataset": dataset, "outcome": outcome_name, "contrast": name, "tier": tier,
                   "cue_arm": cue, "control_arm": ctl, "question": meaning, "n_paired": len(ids)}
            if ids:
                diff = np.array([int(a[e]) - int(b[e]) for e in ids], dtype=float)
                lo, hi = _paired_mean_ci(diff)
                cue_only = sum(1 for e in ids if a[e] and not b[e])
                ctl_only = sum(1 for e in ids if b[e] and not a[e])
                row.update(cue_rate=round(float(np.mean([a[e] for e in ids])), 4),
                           control_rate=round(float(np.mean([b[e] for e in ids])), 4),
                           difference=round(float(diff.mean()), 4), ci_lower=round(lo, 4), ci_upper=round(hi, 4),
                           cue_stop_only=cue_only, control_stop_only=ctl_only,
                           mcnemar_p=round(mcnemar_exact_p(cue_only, ctl_only), 6))
            rows.append(row)
    return rows


def primary_outcome_contrasts(labels: list[dict]) -> list[dict]:
    """PRIMARY OUTCOME: premature stop (no second-hop evidence), paired per question."""
    return binary_outcome_contrasts(labels, lambda l: l["premature_stop_no_hop2"],
                                    lambda l: bool(l.get("primary_eligible")), C.PRIMARY_OUTCOME)


def stopped_by_k_contrasts(labels: list[dict], k: int = C.GATE_FALLBACK_PCR_K) -> list[dict]:
    """Stopped by decision step k (canonical PCR@k event), paired per question."""
    return binary_outcome_contrasts(
        labels, lambda l: l["terminal_route"] in PCR_NUMERATOR_ROUTES and l.get("t_i") is not None and l["t_i"] <= k,
        lambda l: bool(l.get("pcr_eligible")), f"stopped_by_step_{k}")


def primary_by_cell(gate_rows: list[dict], premature: list[dict], stopped_k: list[dict]) -> list[dict]:
    """Pre-specified rule: PASS -> premature-stop table; FAIL (or no gate) -> stopped-by-step-k."""
    gate = {(r["victim"], r["dataset"]): r["gate"] for r in gate_rows}
    out = []
    for r in premature:
        if gate.get((r["victim"], r["dataset"])) == "PASS":
            out.append({**r, "gate": "PASS"})
    for r in stopped_k:
        g = gate.get((r["victim"], r["dataset"]), "MISSING")
        if g != "PASS":
            out.append({**r, "gate": g})
    return out


def secondary_by_arm(labels: list[dict]) -> list[dict]:
    """Primary-outcome rate plus secondary outcomes per arm (each with its own denominator)."""
    groups: dict = defaultdict(list)
    for lab in labels:
        groups[(lab["victim"], lab["dataset"], lab["arm"])].append(lab)
    rows = []
    for (v, d, a), labs in sorted(groups.items()):
        prim = [x["premature_stop_no_hop2"] for x in labs if x.get("primary_eligible")]
        absn = [x["abstained"] for x in labs if x.get("abstained") is not None]
        sc = [x["search_calls_after_d1"] for x in labs if x.get("search_calls_after_d1") is not None]
        na = Counter(x.get("primary_na_reason") for x in labs if not x.get("primary_eligible"))
        rows.append({"victim": v, "dataset": d, "arm": a, "n_runs": sum(1 for x in labs if x.get("status") == "ran"),
                     "n_primary_eligible": len(prim),
                     "premature_stop_rate": round(float(np.mean(prim)), 4) if prim else None,
                     "primary_not_scored": dict(na),
                     "abstention_rate": round(float(np.mean(absn)), 4) if absn else None,
                     "mean_search_calls_after_d1": round(float(np.mean(sc)), 3) if sc else None})
    return rows


def baseline_gate(anchor_labels: list[dict]) -> list[dict]:
    """Go/no-go on the clean anchor runs, per victim x dataset (thresholds in config)."""
    groups: dict = defaultdict(list)
    for lab in anchor_labels:
        groups[(lab["victim"], lab["dataset"])].append(lab)
    rows = []
    for (v, d), labs in sorted(groups.items()):
        elig = [x for x in labs if x.get("pcr_eligible")]
        stop1 = (sum(1 for x in elig if x["terminal_route"] in PCR_NUMERATOR_ROUTES and x.get("t_i") == 1)
                 / len(elig)) if elig else None
        prim = [x for x in labs if x.get("primary_eligible")]
        hop2 = (1 - float(np.mean([x["premature_stop_no_hop2"] for x in prim]))) if prim else None
        acc = [x["answer_correct"] for x in labs if x.get("answer_correct") is not None]
        ok = (stop1 is not None and hop2 is not None and stop1 <= C.BASELINE_GATE_MAX_STOP_AT_1
              and hop2 >= C.BASELINE_GATE_MIN_HOP2_RETRIEVAL)
        rows.append({"victim": v, "dataset": d, "n_clean_runs": len(labs), "n_pcr_eligible": len(elig),
                     "stop_at_step1_rate": None if stop1 is None else round(stop1, 4),
                     "max_allowed_stop_at_1": C.BASELINE_GATE_MAX_STOP_AT_1,
                     "n_primary_eligible": len(prim),
                     "hop2_retrieval_rate": None if hop2 is None else round(hop2, 4),
                     "min_required_hop2_retrieval": C.BASELINE_GATE_MIN_HOP2_RETRIEVAL,
                     "clean_accuracy": round(float(np.mean(acc)), 4) if acc else None,
                     "gate": "PASS" if ok else "FAIL"})
    return rows


def _t(label: dict) -> float | None:
    if not label.get("pcr_eligible"):
        return None
    if label.get("terminal_route") in PCR_NUMERATOR_ROUTES and label.get("t_i") is not None:
        return float(label["t_i"])
    return np.inf


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def pcr_curves(labels: list[dict]) -> list[dict]:
    steps = np.array(C.PCR_STEPS)
    groups: dict = defaultdict(list)
    for lab in labels:
        groups[(lab["victim"], lab["dataset"], lab["arm"])].append(lab)
    rows = []
    for (victim, dataset, arm), labs in sorted(groups.items()):
        ts = np.array([t for t in (_t(x) for x in labs) if t is not None])
        n_run = sum(1 for x in labs if x.get("status") == "ran")
        for s in steps:
            done = int((ts <= s).sum()) if len(ts) else 0
            rc = sum(1 for x in labs if _t(x) is not None and x["terminal_route"] == "RESEARCH_COMPLETE" and x["t_i"] <= s)
            rows.append({"victim": victim, "dataset": dataset, "arm": arm, "step": int(s), "n_questions": len(labs),
                         "n_ran": n_run, "n_eligible": len(ts), "n_completed_by_step": done,
                         "pcr": done / len(ts) if len(ts) else None,
                         "research_complete_by_step": rc, "direct_answer_by_step": done - rc})
    return rows


def all_arms_ran_ids(labels: list[dict]) -> dict:
    """{(victim, dataset): example_ids where all six arms ran} -- the bridge-available subset."""
    ran: dict = defaultdict(lambda: defaultdict(set))
    for lab in labels:
        if lab.get("status") == "ran":
            ran[(lab["victim"], lab["dataset"])][lab["example_id"]].add(lab["arm"])
    return {cell: {e for e, arms in exs.items() if set(C.ARM_ORDER) <= arms} for cell, exs in ran.items()}


def paired_contrasts(labels: list[dict]) -> list[dict]:
    """Every contrast on two question sets: `all_paired` (both arms ran and are
    pcr_eligible) and `all_six_arms_ran` (only questions where a bridge was
    found and both cues were valid), so fixed-vs-neutral can be compared on
    exactly the questions the bridge arms cover."""
    steps = np.array(C.PCR_STEPS)
    by: dict = defaultdict(dict)
    for lab in labels:
        by[(lab["victim"], lab["dataset"], lab["arm"])][lab["example_id"]] = _t(lab)
    subset = all_arms_ran_ids(labels)
    rows = []
    cells = sorted({(v, d) for v, d, _ in by})
    for victim, dataset in cells:
        for name, cue, ctl, _tier, meaning in CONTRASTS:
            a, b = by.get((victim, dataset, cue), {}), by.get((victim, dataset, ctl), {})
            paired = sorted(e for e in a.keys() & b.keys() if a[e] is not None and b[e] is not None)
            for subset_name, ids in (("all_paired", paired),
                                     ("all_six_arms_ran", [e for e in paired if e in subset.get((victim, dataset), set())])):
                base = {"victim": victim, "dataset": dataset, "contrast": name, "question_set": subset_name,
                        "cue_arm": cue, "control_arm": ctl, "question": meaning}
                if not ids:
                    rows.append({**base, "n_paired": 0})
                    continue
                ct, cl, d, lo, hi = _bootstrap_all_steps(np.array([a[e] for e in ids]), np.array([b[e] for e in ids]),
                                                         steps, STEPWISE_BOOTSTRAP_ITERATIONS,
                                                         analysis_config.DETERMINISTIC_SEED)
                for i, s in enumerate(steps):
                    if int(s) not in C.PCR_CHECKPOINTS:
                        continue
                    rows.append({**base, "step": int(s), "n_paired": len(ids),
                                 "control_pcr": round(float(ct[i]), 4), "cue_pcr": round(float(cl[i]), 4),
                                 "delta_pcr": round(float(d[i]), 4), "ci_lower": round(float(lo[i]), 4),
                                 "ci_upper": round(float(hi[i]), 4)})
    return rows


def _paired_mean_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(analysis_config.DETERMINISTIC_SEED)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, size=n)].mean() for _ in range(STEPWISE_BOOTSTRAP_ITERATIONS)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def answer_accuracy_table(labels: list[dict]) -> list[dict]:
    """Final-answer accuracy per arm. Denominator: runs that ran and have a
    readable trace (a run with no final report counts as incorrect, as in the
    canonical answer evaluation)."""
    groups: dict = defaultdict(list)
    for lab in labels:
        if lab.get("status") == "ran" and lab.get("answer_correct") is not None:
            groups[(lab["victim"], lab["dataset"], lab["arm"])].append(lab)
    rows = []
    for (v, d, a), labs in sorted(groups.items()):
        n = len(labs)
        correct = sum(1 for x in labs if x["answer_correct"])
        rows.append({"victim": v, "dataset": d, "arm": a, "n": n, "n_correct": correct,
                     "accuracy": round(correct / n, 4) if n else None})
    return rows


def evidence_summary_table(labels: list[dict]) -> list[dict]:
    """Evidence coverage at the final-report boundary (canonical boundary 4,
    falling back to the episode-0 terminal boundary), per arm."""
    groups: dict = defaultdict(list)
    for lab in labels:
        if lab.get("status") == "ran" and lab.get("evidence_coverage") is not None:
            groups[(lab["victim"], lab["dataset"], lab["arm"])].append(lab)
    rows = []
    for (v, d, a), labs in sorted(groups.items()):
        cov = np.array([x["evidence_coverage"] for x in labs], dtype=float)
        rows.append({"victim": v, "dataset": d, "arm": a, "n": len(cov),
                     "mean_coverage": round(float(cov.mean()), 4),
                     "full_coverage_rate": round(float((cov >= 1 - 1e-9).mean()), 4),
                     "boundaries_used": dict(Counter(x.get("evidence_boundary") for x in labs))})
    return rows


def paired_answer_and_evidence(labels: list[dict]) -> list[dict]:
    """Per contrast: answer flip counts + accuracy delta CI, and paired evidence
    loss (control coverage - cue coverage; positive = cue arm saw less evidence)."""
    by: dict = defaultdict(dict)
    for lab in labels:
        if lab.get("status") == "ran":
            by[(lab["victim"], lab["dataset"], lab["arm"])][lab["example_id"]] = lab
    rows = []
    for victim, dataset in sorted({(v, d) for v, d, _ in by}):
        for name, cue, ctl, _tier, _meaning in CONTRASTS:
            a, b = by.get((victim, dataset, cue), {}), by.get((victim, dataset, ctl), {})
            row = {"victim": victim, "dataset": dataset, "contrast": name, "cue_arm": cue, "control_arm": ctl}
            ans = [e for e in a.keys() & b.keys() if a[e].get("answer_correct") is not None
                   and b[e].get("answer_correct") is not None]
            if ans:
                cc = sum(1 for e in ans if a[e]["answer_correct"] and b[e]["answer_correct"])
                harm = sum(1 for e in ans if b[e]["answer_correct"] and not a[e]["answer_correct"])
                gain = sum(1 for e in ans if a[e]["answer_correct"] and not b[e]["answer_correct"])
                diff = np.array([int(a[e]["answer_correct"]) - int(b[e]["answer_correct"]) for e in ans], dtype=float)
                lo, hi = _paired_mean_ci(diff)
                row.update(n_answer_paired=len(ans), both_correct=cc, control_correct_cue_wrong=harm,
                           control_wrong_cue_correct=gain, both_wrong=len(ans) - cc - harm - gain,
                           net_harm=harm - gain, accuracy_delta=round(float(diff.mean()), 4),
                           accuracy_delta_ci=[round(lo, 4), round(hi, 4)])
            ev = [e for e in a.keys() & b.keys() if a[e].get("evidence_coverage") is not None
                  and b[e].get("evidence_coverage") is not None]
            if ev:
                loss = np.array([b[e]["evidence_coverage"] - a[e]["evidence_coverage"] for e in ev], dtype=float)
                lo, hi = _paired_mean_ci(loss)
                row.update(n_evidence_paired=len(ev), paired_evidence_loss=round(float(loss.mean()), 4),
                           evidence_loss_ci=[round(lo, 4), round(hi, 4)])
            rows.append(row)
    return rows


REPLICATION_MAP = {"neutral": "NaturalClean", "fixed_sufficiency": C.CUE_1, "fixed_saturation": C.CUE_2}


def replication_vs_canonical(labels: list[dict], canonical_run_level: Path) -> list[dict]:
    """Full split only: the neutral and fixed arms re-run canonical conditions on
    the same questions with the same settings. Compare PCR and per-question
    stopping behaviour with the canonical runs (temperature 0, but vLLM batching
    is not bit-deterministic, so agreement is reported, not assumed)."""
    if not canonical_run_level.exists():
        return []
    canon: dict = {}
    with canonical_run_level.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] in REPLICATION_MAP.values() and r["pcr_eligible"] == "True":
                t = (float(r["first_terminal_post_observation_step"])
                     if r["terminal_route"] in PCR_NUMERATOR_ROUTES and r["first_terminal_post_observation_step"] else np.inf)
                canon[(r["model"], r["dataset"], r["condition"], r["example_id"])] = (t, r["terminal_route"])
    rows = []
    groups: dict = defaultdict(list)
    for lab in labels:
        if lab["arm"] in REPLICATION_MAP and _t(lab) is not None:
            key = (lab["victim"], lab["dataset"], REPLICATION_MAP[lab["arm"]], lab["example_id"])
            if key in canon:
                groups[(lab["victim"], lab["dataset"], lab["arm"])].append((_t(lab), lab["terminal_route"], *canon[key]))
    for (v, d, arm), pairs in sorted(groups.items()):
        ours = np.array([p[0] for p in pairs]); theirs = np.array([p[2] for p in pairs])
        row = {"victim": v, "dataset": d, "arm": arm, "canonical_condition": REPLICATION_MAP[arm], "n_matched": len(pairs),
               "same_stop_step_rate": round(float(np.mean(ours == theirs)), 4),
               "same_route_rate": round(float(np.mean([p[1] == p[3] for p in pairs])), 4)}
        for k in (1, 2, 5, 30):
            row[f"pcr@{k}_pilot"] = round(float((ours <= k).mean()), 4)
            row[f"pcr@{k}_canonical"] = round(float((theirs <= k).mean()), 4)
        rows.append(row)
    return rows


def outcome_table(labels: list[dict]) -> list[dict]:
    groups: dict = defaultdict(Counter)
    for lab in labels:
        groups[(lab["victim"], lab["dataset"], lab["arm"])][lab["outcome_label"]] += 1
    cats = ("correct_completion", "incorrect_completion", "premature_direct_answer", "premature_research_complete",
            "other_termination_or_failure", "victim_execution_failure", "not_run")
    return [{"victim": v, "dataset": d, "arm": a, **{c: cnt.get(c, 0) for c in cats}, "total": sum(cnt.values())}
            for (v, d, a), cnt in sorted(groups.items())]


def coverage_table(bridge_recs: list[dict], cue_recs: list[dict], arm_recs: list[dict],
                   expected: dict | None = None) -> list[dict]:
    """Per cell: bridge/cue status counts, arms run/not run, injection failures,
    and completeness against the manifest (catches a missing or unfinished shard)."""
    rows = []
    for (victim, dataset) in sorted({(r["victim"], r["dataset"]) for r in bridge_recs}):
        br = [r for r in bridge_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        cr = [r for r in cue_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        ar = [r for r in arm_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        row = {"victim": victim, "dataset": dataset, "n_questions": len(br),
               **{f"bridge_{k}": v for k, v in Counter(r["status"] for r in br).items()}}
        for unit in C.UNITS:
            row[f"pool_ok_{unit}"] = sum(1 for r in cr if r.get("unit") == unit and r["status"] == "OK")
        for arm in C.ARM_ORDER:
            row[f"ran_{arm}"] = sum(1 for r in ar if r["arm"] == arm and r["status"] == "ran")
        row["arms_ran"] = sum(1 for r in ar if r["status"] == "ran")
        row["arms_not_run"] = sum(1 for r in ar if r["status"] != "ran")
        row["injection_failures"] = sum(1 for r in ar if r["status"] == "ran" and not r.get("injection_ok"))
        if expected is not None:
            row["n_expected_questions"] = expected.get(dataset)
            row["n_questions_with_all_arm_records"] = sum(
                1 for e in {r["example_id"] for r in ar}
                if {r["arm"] for r in ar if r["example_id"] == e} >= set(C.ARM_ORDER))
            row["complete"] = row["n_questions_with_all_arm_records"] == row["n_expected_questions"]
        rows.append(row)
    return rows


def cue_audit_rows(bridge_recs: list[dict], cue_recs: list[dict], selections: dict) -> list[dict]:
    """One row per generated variant: grounding, condition, text, source, validity, similarity, selection."""
    bridges = {(r["victim"], r["dataset"], r["example_id"]): r for r in bridge_recs}
    rows = []
    for c in cue_recs:
        key = (c["victim"], c["dataset"], c["example_id"])
        b = bridges.get(key, {})
        a = b.get("anchor") or {}
        chosen_arms = {arm: sel for (vk, dk, ek, arm), sel in selections.items() if (vk, dk, ek) == key
                       and C.ARMS[arm].get("unit") == c.get("unit")}
        for v in c.get("variants", []):
            rows.append({
                "victim": c["victim"], "dataset": c["dataset"], "example_id": c["example_id"],
                "unit": c.get("unit"), "mode": c.get("mode"), "condition": c["condition"],
                "variant_index": v["index"], "seed": v["seed"],
                "bridge_status": b.get("status"), "bridge_fact": a.get("bridge_fact"),
                "bridge_span": a.get("bridge_span"), "source_document_id": a.get("source_document_id"),
                "source_document_title": a.get("source_document_title"), "delta": a.get("delta"),
                "cue_text": v["cue"], "cue_tokens": v.get("token_count"), "valid": v["valid"],
                **{f"check_{k}": v["checks"].get(k) for k in ("V1", "V2", "V3", "V4", "V5", "COPY", "ENTITY")},
                "errors": {k: e for k, e in v.get("errors", {}).items() if e},
                "fixed_similarity_max": (v.get("fixed_similarity") or {}).get("max_similarity"),
                "fixed_similarity_own": ((v.get("fixed_similarity") or {}).get(c["condition"]) or {}).get("similarity"),
                **{f"feat_{k}": (v.get("features") or {}).get(k) for k in C.SCORE_FEATURES},
                "judge_reason": (v.get("judge") or {}).get("mechanism_reason"),
                "audit_gold_answer_leak": v.get("audit_gold_answer_leak"),
                "selected_for_arms": sorted(arm for arm, sel in chosen_arms.items()
                                            if sel and sel.get("index") == v["index"]),
                "distractor": ((c.get("distractor") or {}).get("distractor")),
            })
    return rows


def plot_pcr(curves: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"neutral": ("#555555", "-"), "fixed_sufficiency": ("#7570b3", "--"),
              "entity_sufficiency": ("#1b9e77", "-."), "bridge_sufficiency_k1": ("#7570b3", ":"),
              "bridge_sufficiency_k3": ("#7570b3", "-"), "unsupported_sufficiency": ("#e7298a", "-"),
              "fixed_saturation": ("#d95f02", "--"), "bridge_saturation_k3": ("#d95f02", "-")}
    paths = []
    for victim, dataset in sorted({(r["victim"], r["dataset"]) for r in curves}):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for arm in C.ARM_ORDER:
            pts = [r for r in curves if (r["victim"], r["dataset"], r["arm"]) == (victim, dataset, arm)
                   and r["pcr"] is not None and r["step"] <= 10]
            if not pts:
                continue
            color, ls = styles[arm]
            n = pts[0]["n_eligible"]
            ax.plot([p["step"] for p in pts], [100 * p["pcr"] for p in pts], ls, color=color, marker="o", ms=3,
                    label=f"{C.ARM_DISPLAY[arm]} (n={n})")
        ax.set_xlabel("research decision step k")
        ax.set_ylabel("cumulative PCR@k (%)")
        ax.set_title(f"{victim} / {dataset}")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = out_dir / f"pcr_{victim}_{dataset}.{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150)
            paths.append(p)
        plt.close(fig)
    return paths


def summarize(labels: Iterable[dict], contrasts: list[dict], answers: list[dict] | None = None,
              paired_ae: list[dict] | None = None, evidence: list[dict] | None = None,
              primary: list[dict] | None = None, secondary: list[dict] | None = None) -> str:
    lines = ["# Pilot summary (generated)", "",
             "Runs are never truncated: every run reaches final_report_generation. PCR@30 = end of budget.", "",
             "## PRIMARY (per cell: premature stop if the baseline gate PASSED, else stopped by step k), paired per question", "",
             "| victim | dataset | contrast | tier | n | cue rate | control rate | diff [95% CI] | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in primary or []:
        if not r.get("n_paired"):
            lines.append(f"| {r['victim']} | {r['dataset']} | {r['contrast']} [{r.get('outcome', '')}] | {r['tier']} | 0 | | | | |")
            continue
        lines.append(f"| {r['victim']} | {r['dataset']} | {r['contrast']} [{r.get('outcome', '')}] | {r['tier']} | {r['n_paired']} | "
                     f"{r['cue_rate']:.3f} | {r['control_rate']:.3f} | {100 * r['difference']:+.1f} "
                     f"[{100 * r['ci_lower']:.1f}, {100 * r['ci_upper']:.1f}] | {r['mcnemar_p']:.4g} |")
    if secondary:
        lines += ["", "## Per arm: primary rate and secondary outcomes", "",
                  "| victim | dataset | arm | runs | primary n | premature stop | abstention | search calls after d1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in secondary:
            lines.append(f"| {r['victim']} | {r['dataset']} | {r['arm']} | {r['n_runs']} | {r['n_primary_eligible']} | "
                         f"{r['premature_stop_rate']} | {r['abstention_rate']} | {r['mean_search_calls_after_d1']} |")
    lines += ["", "## PCR contrasts (all paired questions; Δ in pp, 95% paired bootstrap CI)", "",
             "| victim | dataset | contrast | n | Δ@1 | Δ@2 | Δ@5 | Δ@30 |", "|---|---|---|---|---|---|---|---|"]
    keyed: dict = defaultdict(dict)
    for r in contrasts:
        if r.get("n_paired") and r.get("question_set", "all_paired") == "all_paired":
            keyed[(r["victim"], r["dataset"], r["contrast"])][r["step"]] = r
    for (v, d, name), by_step in sorted(keyed.items()):
        cells = []
        for s in (1, 2, 5, 30):
            r = by_step.get(s)
            cells.append(f"{100 * r['delta_pcr']:+.1f} [{100 * r['ci_lower']:.1f}, {100 * r['ci_upper']:.1f}]" if r else "")
        n = next(iter(by_step.values()))["n_paired"]
        lines.append(f"| {v} | {d} | {name} | {n} | " + " | ".join(cells) + " |")
    if answers or evidence:
        ev = {(r["victim"], r["dataset"], r["arm"]): r for r in (evidence or [])}
        lines += ["", "## Final-answer accuracy and evidence coverage by arm", "",
                  "| victim | dataset | arm | n | accuracy | mean evidence coverage | full coverage |",
                  "|---|---|---|---|---|---|---|"]
        for r in answers or []:
            e = ev.get((r["victim"], r["dataset"], r["arm"]), {})
            lines.append(f"| {r['victim']} | {r['dataset']} | {r['arm']} | {r['n']} | {r['accuracy']} | "
                         f"{e.get('mean_coverage', '')} | {e.get('full_coverage_rate', '')} |")
    if paired_ae:
        lines += ["", "## Paired answer flips and evidence loss per contrast", "",
                  "| victim | dataset | contrast | n | ctl✓→cue✗ | ctl✗→cue✓ | Δacc [CI] | evidence loss [CI] |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in paired_ae:
            if not r.get("n_answer_paired"):
                continue
            acc = f"{100 * r['accuracy_delta']:+.1f} [{100 * r['accuracy_delta_ci'][0]:.1f}, {100 * r['accuracy_delta_ci'][1]:.1f}]"
            loss = (f"{100 * r['paired_evidence_loss']:+.1f} [{100 * r['evidence_loss_ci'][0]:.1f}, "
                    f"{100 * r['evidence_loss_ci'][1]:.1f}]") if r.get("n_evidence_paired") else ""
            lines.append(f"| {r['victim']} | {r['dataset']} | {r['contrast']} | {r['n_answer_paired']} | "
                         f"{r['control_correct_cue_wrong']} | {r['control_wrong_cue_correct']} | {acc} | {loss} |")
    return "\n".join(lines) + "\n"
