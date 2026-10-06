"""Build frozen development / held-out pilot splits for HotpotQA and 2Wiki.

Reuses the canonical eligibility filter and corpus/evidence builders
(attack/runtime/production_manifest.py, attack/runtime/corpus.py) unchanged, so
a pilot example is constructed exactly like a production example. It then:

  * excludes every example id already used by the canonical n450 production
    manifests and the cue-development dev_bundle, and records that exclusion;
  * stratifies the remainder by question type and draws a fixed-seed
    development set (10 per dataset) and a disjoint held-out set
    (20 per dataset);
  * writes a pilot manifest, corpus, and evidence file per split, plus a
    provenance summary with prior-exposure status.

Selection never looks at attack outcomes. If genuinely unused eligible
examples are too few, the summary downgrades the label to
"reserved_pilot_evaluation_set" and discloses the overlap rather than
silently reusing exposed ids.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attack.runtime.corpus import write_jsonl
from attack.runtime.production_manifest import (
    REPO_ROOT,
    _eligible_candidates,
    _load_dataset,
    _build_rows,
)

PILOT_SEED = 20261005
PILOT_DATASETS = ("hotpot", "2wiki")
DEV_PER_DATASET = 10
HELDOUT_PER_DATASET = 20
DEFAULT_OUT = REPO_ROOT / "results" / "adaptive_pilot" / "splits"

_PRIOR_SOURCES = {
    "production_n450": lambda ds: [REPO_ROOT / "results" / "manifests" / f"{ds}_n450.jsonl"],
    "cue_development": lambda ds: [REPO_ROOT / "results" / "nonproduction_validation" / "dev_bundle" / "dev_examples.jsonl"],
    "runtime_gate": lambda ds: [REPO_ROOT / "results" / "nonproduction_validation" / "dev_bundle" / "smoke_examples.jsonl"],
}


def _ids_from(path: Path, dataset: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") in (dataset, None) or str(path).endswith("_n450.jsonl"):
            eid = row.get("example_id")
            if eid:
                ids.add(str(eid))
    return ids


def _prior_ids(dataset: str) -> dict[str, set[str]]:
    return {label: set().union(*[_ids_from(p, dataset) for p in paths(dataset)]) if paths(dataset) else set()
            for label, paths in _PRIOR_SOURCES.items()}


def _question_type(dataset: str, example) -> str:
    return str(example.get("type") or "unspecified")


def _stratified_sample(candidates: list, n: int, seed: int, dataset: str) -> list:
    by_type: dict[str, list] = {}
    for cand in candidates:
        by_type.setdefault(_question_type(dataset, cand.example), []).append(cand)
    for key in by_type:
        by_type[key] = sorted(by_type[key], key=lambda c: c.source_id)
        random.Random(seed + sum(ord(ch) for ch in key)).shuffle(by_type[key])
    picked: list = []
    types = sorted(by_type)
    idx = 0
    # round-robin across types for a type mixture, deterministic
    while len(picked) < n and any(by_type[t] for t in types):
        t = types[idx % len(types)]
        if by_type[t]:
            picked.append(by_type[t].pop())
        idx += 1
    return picked[:n]


def build_split(dataset: str, *, out_dir: Path, local_files_only: bool = True) -> dict:
    ds = _load_dataset(dataset, local_files_only=local_files_only)
    candidates = _eligible_candidates(dataset, ds)
    prior = _prior_ids(dataset)
    exposed = set().union(*prior.values()) if prior else set()

    unused = [c for c in candidates if c.source_id not in exposed]
    need = DEV_PER_DATASET + HELDOUT_PER_DATASET
    label = "genuinely_unused"
    pool = unused
    if len(unused) < need:
        label = "reserved_pilot_evaluation_set"
        pool = candidates  # disclose exposure rather than fail

    # Draw dev first, then held-out from the remainder (disjoint by construction).
    dev = _stratified_sample(pool, DEV_PER_DATASET, PILOT_SEED, dataset)
    dev_ids = {c.source_id for c in dev}
    remainder = [c for c in pool if c.source_id not in dev_ids]
    heldout = _stratified_sample(remainder, HELDOUT_PER_DATASET, PILOT_SEED + 1, dataset)

    assert not (dev_ids & {c.source_id for c in heldout}), "dev/held-out overlap"

    summaries = {}
    for split_name, picks in (("dev", dev), ("heldout", heldout)):
        man_rows, corpus_rows, ev_rows = [], [], []
        for ordinal, cand in enumerate(sorted(picks, key=lambda c: c.source_id)):
            docs, evidence = _build_rows(dataset, cand.example, cand.source_index)
            man_rows.append({
                "dataset": dataset,
                "example_id": cand.source_id,
                "sample_ordinal": ordinal,
                "question": evidence.question,
                "answer": evidence.answer,
                "answer_aliases": list(evidence.answer_aliases),
                "hop_count": evidence.hop_count,
                "question_type": _question_type(dataset, cand.example),
                "required_dependency_ids": [d.dependency_id for d in evidence.dependencies],
                "required_paragraph_ids": list(dict.fromkeys(d.supporting_paragraph_id or d.document_id for d in evidence.dependencies)),
                "split": split_name,
                "prior_exposure": {k: (cand.source_id in v) for k, v in prior.items()},
                "pilot_split_label": label,
            })
            corpus_rows.extend(d.to_json() for d in docs)
            ev_rows.append(evidence.to_json())
        base = out_dir / dataset / split_name
        write_jsonl(base / f"{dataset}_{split_name}_manifest.jsonl", man_rows)
        write_jsonl(base / f"{dataset}_{split_name}_corpus.jsonl", corpus_rows)
        write_jsonl(base / f"{dataset}_{split_name}_evidence.jsonl", ev_rows)
        summaries[split_name] = {
            "n": len(man_rows),
            "type_counts": {t: sum(1 for r in man_rows if r["question_type"] == t) for t in sorted({r["question_type"] for r in man_rows})},
            "manifest": str(base / f"{dataset}_{split_name}_manifest.jsonl"),
            "corpus": str(base / f"{dataset}_{split_name}_corpus.jsonl"),
            "evidence": str(base / f"{dataset}_{split_name}_evidence.jsonl"),
        }

    summary = {
        "dataset": dataset,
        "seed": PILOT_SEED,
        "pilot_split_label": label,
        "n_eligible_total": len(candidates),
        "n_prior_exposed": len(exposed),
        "n_genuinely_unused": len(unused),
        "prior_exposure_counts": {k: len(v) for k, v in prior.items()},
        "splits": summaries,
    }
    (out_dir / dataset / f"{dataset}_split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build adaptive-pilot dev/held-out splits.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--datasets", nargs="+", default=list(PILOT_DATASETS))
    ap.add_argument("--allow-download", action="store_true", help="permit HF download (default offline)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for dataset in args.datasets:
        all_summaries[dataset] = build_split(dataset, out_dir=args.out_dir, local_files_only=not args.allow_download)
        print(json.dumps(all_summaries[dataset], indent=2, sort_keys=True))
    (args.out_dir / "pilot_splits_summary.json").write_text(json.dumps(all_summaries, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
