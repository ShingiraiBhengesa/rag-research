"""Human audit of the automated judges (Zhen point 5).

Three sheets of config.JUDGE_AUDIT_N items each, drawn with config.JUDGE_AUDIT_SEED
from a stable sort (no item is chosen by outcome):
  entailment  span |= statement      (step 2, NLI)        auto = span_backed
  match       Match(fact, edge)      (step 3, rule)       auto = match()
  validity    V3, V4, V5, ENTITY     (step 5, LLM judge)  auto = the gate result
The annotator fills each `human_*` column with yes/no (blank = not labelled).
`agreement()` reports, per check, n labelled, percent agreement and Cohen's kappa.
V1/V2/COPY are deterministic string/length checks and are not audited.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from attack.dynamic_bridge_pilot import config as C
from attack.dynamic_bridge_pilot.bridge import Edge, Fact, match

VALIDITY_CHECKS = ("V3", "V4", "V5", "ENTITY")
_YES = {"yes", "y", "1", "true", "t"}
_NO = {"no", "n", "0", "false", "f"}


def _sample(items: list[dict], key) -> list[dict]:
    items = sorted(items, key=key)
    rng = random.Random(C.JUDGE_AUDIT_SEED)
    return items if len(items) <= C.JUDGE_AUDIT_N else rng.sample(items, C.JUDGE_AUDIT_N)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _yn(b) -> str:
    return "yes" if b else "no"


def export_sheets(bridges: list[dict], cues: list[dict], out_dir: Path) -> dict[str, Path]:
    ent, mat, val = [], [], []
    for b in bridges:
        base = {"victim": b["victim"], "dataset": b["dataset"], "example_id": b["example_id"]}
        edges = [Edge(**e) for e in (b.get("graph") or {}).get("edges", [])]
        for i, f in enumerate(b.get("facts", [])):
            if f.get("span_offset") is None:
                continue
            ent.append({**base, "fact_index": i, "span": f["span"], "statement": f["statement"],
                        "auto_entails": _yn(f.get("span_backed")), "entail_prob": f.get("entail_prob"),
                        "human_entails": ""})
            if not f.get("span_backed"):
                continue
            fact = Fact(**{k: f[k] for k in ("relation", "head", "tail", "statement", "span")})
            for e in edges:
                mat.append({**base, "fact_index": i, "fact": f"({f['head']}) --[{f['relation']}]--> ({f['tail']})",
                            "edge_id": e.id, "edge": f"({e.head}) --[{e.relation}]--> ({e.tail})",
                            "question": b.get("question", ""), "auto_match": _yn(match(fact, e)),
                            "human_match": ""})
    for c in cues:
        for v in c.get("variants", []):
            if not v.get("cue") or not v.get("judge", {}).get("ok"):
                continue
            row = {"victim": c["victim"], "dataset": c["dataset"], "example_id": c["example_id"],
                   "unit": c.get("unit"), "condition": c["condition"], "variant_index": v["index"],
                   "cue_text": v["cue"], "bridge_fact": c.get("bridge_fact", "")}
            for k in VALIDITY_CHECKS:
                if k == "ENTITY" and c.get("mode") != "entity":
                    row[f"auto_{k}"], row[f"human_{k}"] = "n/a", "n/a"
                else:
                    row[f"auto_{k}"], row[f"human_{k}"] = _yn(v["checks"].get(k)), ""
            val.append(row)
    k = lambda r: (r["victim"], r["dataset"], r["example_id"], str(r.get("fact_index", "")),
                   str(r.get("edge_id", "")), str(r.get("unit", "")), str(r.get("variant_index", "")))
    paths = {"entailment": out_dir / "judge_audit_entailment.csv", "match": out_dir / "judge_audit_match.csv",
             "validity": out_dir / "judge_audit_validity.csv"}
    _write(paths["entailment"], _sample(ent, k))
    _write(paths["match"], _sample(mat, k))
    _write(paths["validity"], _sample(val, k))
    return paths


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _parse(x: str) -> bool | None:
    x = (x or "").strip().lower()
    return True if x in _YES else False if x in _NO else None


def agreement(out_dir: Path) -> list[dict]:
    specs = [("entailment", "judge_audit_entailment.csv", [("entailment", "auto_entails", "human_entails")]),
             ("match", "judge_audit_match.csv", [("match", "auto_match", "human_match")]),
             ("validity", "judge_audit_validity.csv",
              [(k, f"auto_{k}", f"human_{k}") for k in VALIDITY_CHECKS])]
    rows = []
    for sheet, fname, checks in specs:
        path = out_dir / fname
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            data = list(csv.DictReader(fh))
        for check, auto_col, human_col in checks:
            pairs = [(_parse(r[auto_col]), _parse(r[human_col])) for r in data]
            pairs = [(a, h) for a, h in pairs if a is not None and h is not None]
            kappa = cohen_kappa(pairs)
            rows.append({"sheet": sheet, "check": check, "n_labelled": len(pairs),
                         "percent_agreement": round(sum(a == h for a, h in pairs) / len(pairs), 4) if pairs else None,
                         "cohen_kappa": None if kappa is None else round(kappa, 4)})
    return rows
