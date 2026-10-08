"""Fit the scorer weights lambda from manual audit ratings on the dev split.

Inputs are the stored features of VALID dev variants plus a human rating per
variant (1-5, see AUDIT_RUBRIC.md). No victim output is read. lambda is chosen
on the simplex grid (step config.LAMBDA_GRID_STEP) to maximize, over
(victim, question, condition) groups with >= 2 rated valid variants:
  1. mean within-group pairwise concordance between Q(u) and the rating
     (pairs tied in rating are skipped);
  2. tie-break: share of groups whose argmax-Q variant has the group's top rating;
  3. tie-break: smallest L1 distance to uniform weights (conservative).
The result is written with its hash and frozen by freeze.py before held-out.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from attack.dynamic_bridge_pilot import config as C
from attack.dynamic_bridge_pilot.cues import score
from attack.runtime.hashing import stable_hash

EQUAL_LAMBDA = {k: 1.0 / len(C.SCORE_FEATURES) for k in C.SCORE_FEATURES}


def simplex_grid(step: float = C.LAMBDA_GRID_STEP) -> list[dict]:
    n = round(1 / step)
    out = []
    for combo in itertools.product(range(n + 1), repeat=len(C.SCORE_FEATURES) - 1):
        if sum(combo) <= n:
            w = list(combo) + [n - sum(combo)]
            out.append({k: round(x / n, 6) for k, x in zip(C.SCORE_FEATURES, w)})
    return out


def _groups(variants: list[dict], ratings: Mapping[tuple, float]) -> list[list[tuple[dict, float]]]:
    g: dict = defaultdict(list)
    for v in variants:
        key = (v["victim"], v["dataset"], v["example_id"], v["condition"], v["index"])
        if v["valid"] and key in ratings:
            g[key[:4]].append((v, ratings[key]))
    return [items for items in g.values() if len(items) >= 2]


def objective(lam: Mapping[str, float], groups) -> tuple[float, float, float]:
    conc, top1 = [], []
    for items in groups:
        q = [(score(v["features"], lam), r) for v, r in items]
        agree = total = 0
        for (qa, ra), (qb, rb) in itertools.combinations(q, 2):
            if ra == rb:
                continue
            total += 1
            agree += 1 if (qa - qb) * (ra - rb) > 0 else (0.5 if qa == qb else 0)
        if total:
            conc.append(agree / total)
        best = max(range(len(items)), key=lambda i: (round(q[i][0], 9), -items[i][0]["index"]))
        top1.append(1.0 if items[best][1] == max(r for _, r in items) else 0.0)
    dist = sum(abs(lam[k] - EQUAL_LAMBDA[k]) for k in C.SCORE_FEATURES)
    return (sum(conc) / len(conc) if conc else 0.0, sum(top1) / len(top1) if top1 else 0.0, -dist)


def fit(variants: list[dict], ratings: Mapping[tuple, float]) -> dict:
    groups = _groups(variants, ratings)
    if not groups:
        raise ValueError("no rated groups with >= 2 valid variants; cannot fit lambda")
    best = max(simplex_grid(), key=lambda lam: (objective(lam, groups), tuple(-lam[k] for k in C.SCORE_FEATURES)))
    conc, top1, negdist = objective(best, groups)
    eq = objective(EQUAL_LAMBDA, groups)
    return {"lambda": best, "n_groups": len(groups), "n_rated_variants": sum(len(g) for g in groups),
            "pairwise_concordance": round(conc, 4), "top1_agreement": round(top1, 4),
            "l1_from_equal": round(-negdist, 4),
            "equal_weights_baseline": {"pairwise_concordance": round(eq[0], 4), "top1_agreement": round(eq[1], 4)},
            "grid_step": C.LAMBDA_GRID_STEP}


def flatten_variants(cue_recs: list[dict]) -> list[dict]:
    return [{**v, "victim": c["victim"], "dataset": c["dataset"], "example_id": c["example_id"],
             "condition": c["condition"]} for c in cue_recs for v in c.get("variants", [])]


def load_ratings(path: Path) -> tuple[dict, set[str]]:
    ratings, raters = {}, set()
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("rating") or "").strip():
                continue
            key = (r["victim"], r["dataset"], r["example_id"], r["condition"], int(r["variant_index"]))
            ratings[key] = float(r["rating"])
            raters.add((r.get("rater") or "").strip() or "unspecified")
    return ratings, raters


def write_lambda(path: Path, result: dict, *, source: str, raters: set[str]) -> dict:
    payload = {**result, "source": source, "raters": sorted(raters)}
    payload["lambda_hash"] = stable_hash({"lambda": payload["lambda"]}, length=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
