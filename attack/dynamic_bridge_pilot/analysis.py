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
    # (name, cue arm, control arm, question it answers)
    ("bridge_fact_vs_neutral", "bridge_fact", "neutral", "does grounded bridge text alone change stopping?"),
    ("fixed_sufficiency_vs_neutral", "fixed_sufficiency", "neutral", "replicates canonical fixed-cue effect"),
    ("fixed_saturation_vs_neutral", "fixed_saturation", "neutral", "replicates canonical fixed-cue effect"),
    ("bridge_cue_sufficiency_vs_bridge_fact", "bridge_cue_sufficiency", "bridge_fact", "does the claim add beyond the fact?"),
    ("bridge_cue_saturation_vs_bridge_fact", "bridge_cue_saturation", "bridge_fact", "does the claim add beyond the fact?"),
    ("bridge_cue_sufficiency_vs_fixed_sufficiency", "bridge_cue_sufficiency", "fixed_sufficiency", "dynamic vs fixed"),
    ("bridge_cue_saturation_vs_fixed_saturation", "bridge_cue_saturation", "fixed_saturation", "dynamic vs fixed"),
    ("bridge_cue_saturation_vs_bridge_cue_sufficiency", "bridge_cue_saturation", "bridge_cue_sufficiency", "cue 2 vs cue 1"),
    ("bridge_cue_sufficiency_vs_neutral", "bridge_cue_sufficiency", "neutral", "total dynamic effect"),
    ("bridge_cue_saturation_vs_neutral", "bridge_cue_saturation", "neutral", "total dynamic effect"),
)


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


def paired_contrasts(labels: list[dict]) -> list[dict]:
    steps = np.array(C.PCR_STEPS)
    by: dict = defaultdict(dict)
    for lab in labels:
        by[(lab["victim"], lab["dataset"], lab["arm"])][lab["example_id"]] = _t(lab)
    rows = []
    cells = sorted({(v, d) for v, d, _ in by})
    for victim, dataset in cells:
        for name, cue, ctl, meaning in CONTRASTS:
            a, b = by.get((victim, dataset, cue), {}), by.get((victim, dataset, ctl), {})
            ids = sorted(e for e in a.keys() & b.keys() if a[e] is not None and b[e] is not None)
            if not ids:
                rows.append({"victim": victim, "dataset": dataset, "contrast": name, "n_paired": 0})
                continue
            ct, cl, d, lo, hi = _bootstrap_all_steps(np.array([a[e] for e in ids]), np.array([b[e] for e in ids]),
                                                     steps, STEPWISE_BOOTSTRAP_ITERATIONS,
                                                     analysis_config.DETERMINISTIC_SEED)
            for i, s in enumerate(steps):
                if int(s) not in C.PCR_CHECKPOINTS:
                    continue
                rows.append({"victim": victim, "dataset": dataset, "contrast": name, "cue_arm": cue,
                             "control_arm": ctl, "question": meaning, "step": int(s), "n_paired": len(ids),
                             "control_pcr": round(float(ct[i]), 4), "cue_pcr": round(float(cl[i]), 4),
                             "delta_pcr": round(float(d[i]), 4), "ci_lower": round(float(lo[i]), 4),
                             "ci_upper": round(float(hi[i]), 4)})
    return rows


def outcome_table(labels: list[dict]) -> list[dict]:
    groups: dict = defaultdict(Counter)
    for lab in labels:
        groups[(lab["victim"], lab["dataset"], lab["arm"])][lab["primary_label"]] += 1
    cats = ("correct_completion", "incorrect_completion", "premature_direct_answer", "premature_research_complete",
            "other_termination_or_failure", "not_run")
    return [{"victim": v, "dataset": d, "arm": a, **{c: cnt.get(c, 0) for c in cats}, "total": sum(cnt.values())}
            for (v, d, a), cnt in sorted(groups.items())]


def coverage_table(bridge_recs: list[dict], cue_recs: list[dict], arm_recs: list[dict]) -> list[dict]:
    rows = []
    for (victim, dataset) in sorted({(r["victim"], r["dataset"]) for r in bridge_recs}):
        br = [r for r in bridge_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        cr = [r for r in cue_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        ar = [r for r in arm_recs if (r["victim"], r["dataset"]) == (victim, dataset)]
        row = {"victim": victim, "dataset": dataset, "n_questions": len(br),
               **{f"bridge_{k}": v for k, v in Counter(r["status"] for r in br).items()}}
        for cond in C.SELECTED_CONDITIONS:
            row[f"cue_ok_{cond}"] = sum(1 for r in cr if r["condition"] == cond and r["status"] == "OK")
        row["arms_ran"] = sum(1 for r in ar if r["status"] == "ran")
        row["arms_not_run"] = sum(1 for r in ar if r["status"] != "ran")
        row["injection_failures"] = sum(1 for r in ar if r["status"] == "ran" and not r.get("injection_ok"))
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
        chosen = selections.get((*key, c["condition"]))
        for v in c.get("variants", []):
            rows.append({
                "victim": c["victim"], "dataset": c["dataset"], "example_id": c["example_id"],
                "condition": c["condition"], "variant_index": v["index"], "seed": v["seed"],
                "bridge_status": b.get("status"), "bridge_fact": a.get("bridge_fact"),
                "bridge_span": a.get("bridge_span"), "source_document_id": a.get("source_document_id"),
                "source_document_title": a.get("source_document_title"), "delta": a.get("delta"),
                "cue_text": v["cue"], "cue_tokens": v.get("token_count"), "valid": v["valid"],
                **{f"check_{k}": v["checks"].get(k) for k in ("V1", "V2", "V3", "V4", "V5", "COPY")},
                "errors": {k: e for k, e in v.get("errors", {}).items() if e},
                "fixed_similarity_max": (v.get("fixed_similarity") or {}).get("max_similarity"),
                "fixed_similarity_own": ((v.get("fixed_similarity") or {}).get(c["condition"]) or {}).get("similarity"),
                **{f"feat_{k}": (v.get("features") or {}).get(k) for k in C.SCORE_FEATURES},
                "judge_reason": (v.get("judge") or {}).get("mechanism_reason"),
                "audit_gold_answer_leak": v.get("audit_gold_answer_leak"),
                "selected": bool(chosen and chosen.get("index") == v["index"]),
            })
    return rows


def plot_pcr(curves: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"neutral": ("#555555", "-"), "bridge_fact": ("#1b9e77", "-"),
              "fixed_sufficiency": ("#7570b3", "--"), "fixed_saturation": ("#d95f02", "--"),
              "bridge_cue_sufficiency": ("#7570b3", "-"), "bridge_cue_saturation": ("#d95f02", "-")}
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


def summarize(labels: Iterable[dict], contrasts: list[dict]) -> str:
    lines = ["# Pilot PCR summary (generated)", "", "| victim | dataset | contrast | n | Δ@1 | Δ@2 | Δ@5 |",
             "|---|---|---|---|---|---|---|"]
    keyed: dict = defaultdict(dict)
    for r in contrasts:
        if r.get("n_paired"):
            keyed[(r["victim"], r["dataset"], r["contrast"])][r["step"]] = r
    for (v, d, name), by_step in sorted(keyed.items()):
        cells = []
        for s in (1, 2, 5):
            r = by_step.get(s)
            cells.append(f"{100 * r['delta_pcr']:+.1f} [{100 * r['ci_lower']:.1f}, {100 * r['ci_upper']:.1f}]" if r else "")
        n = next(iter(by_step.values()))["n_paired"]
        lines.append(f"| {v} | {d} | {name} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
