"""Smoke / dev / held-out splits for the dynamic bridge-cue pilot.

dev      = the adaptive pilot's existing dev split (10 per dataset, seed
           20261005), reused unchanged for lambda fitting.
smoke    = 3 per dataset, drawn first (seed SPLIT_SEED) from questions never
           used in production, cue development, the runtime gate, or dev.
heldout  = 100 per dataset, drawn (seed SPLIT_SEED + 1) from the remainder.
All three are disjoint by construction (asserted). Draws are stratified by
question type and fixed before any pilot run; no question is chosen or
dropped by outcome.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from attack.adaptive_pilot.split_manifest import _prior_ids, _question_type, _stratified_sample
from attack.dynamic_bridge_pilot import config as C
from attack.dynamic_bridge_pilot.paths import DEFAULT_SPLITS, split_paths
from attack.runtime.corpus import write_jsonl
from attack.runtime.production_manifest import REPO_ROOT, _build_rows, _eligible_candidates, _load_dataset

ADAPTIVE_SPLITS = REPO_ROOT / "results" / "adaptive_pilot" / "splits"
DEFAULT_OUT = DEFAULT_SPLITS
SPLITS = ("smoke", "dev", "heldout")


def _write(out_dir: Path, dataset: str, split: str, picks: list, prior: dict) -> dict:
    man, corpus, ev = [], [], []
    for ordinal, cand in enumerate(sorted(picks, key=lambda c: c.source_id)):
        docs, evidence = _build_rows(dataset, cand.example, cand.source_index)
        man.append({
            "dataset": dataset, "example_id": cand.source_id, "sample_ordinal": ordinal,
            "question": evidence.question, "answer": evidence.answer,
            "answer_aliases": list(evidence.answer_aliases), "hop_count": evidence.hop_count,
            "question_type": _question_type(dataset, cand.example),
            "required_dependency_ids": [d.dependency_id for d in evidence.dependencies],
            "split": split, "prior_exposure": {k: (cand.source_id in v) for k, v in prior.items()},
        })
        corpus.extend(d.to_json() for d in docs)
        ev.append(evidence.to_json())
    paths = split_paths(out_dir, dataset, split)
    write_jsonl(paths["manifest"], man)
    write_jsonl(paths["corpus"], corpus)
    write_jsonl(paths["evidence"], ev)
    return {"n": len(man), "type_counts": {t: sum(1 for r in man if r["question_type"] == t)
                                           for t in sorted({r["question_type"] for r in man})}}


def build(dataset: str, out_dir: Path, *, local_files_only: bool = True) -> dict:
    ds = _load_dataset(dataset, local_files_only=local_files_only)
    candidates = _eligible_candidates(dataset, ds)
    prior = _prior_ids(dataset)
    dev_src = split_paths(ADAPTIVE_SPLITS, dataset, "dev")
    dev_ids = {json.loads(l)["example_id"] for l in dev_src["manifest"].read_text().splitlines() if l.strip()}
    exposed = set().union(*prior.values()) | dev_ids
    pool = [c for c in candidates if c.source_id not in exposed]
    smoke = _stratified_sample(pool, C.SMOKE_PER_DATASET, C.SPLIT_SEED, dataset)
    smoke_ids = {c.source_id for c in smoke}
    heldout = _stratified_sample([c for c in pool if c.source_id not in smoke_ids], C.HELDOUT_PER_DATASET,
                                 C.SPLIT_SEED + 1, dataset)
    held_ids = {c.source_id for c in heldout}
    assert not (smoke_ids & held_ids) and not (dev_ids & (smoke_ids | held_ids)), "split overlap"
    assert len(heldout) == C.HELDOUT_PER_DATASET, f"pool too small for held-out: {len(heldout)}"
    summary = {"dataset": dataset, "seed": C.SPLIT_SEED, "n_eligible": len(candidates),
               "n_pool_after_exclusions": len(pool),
               "smoke": _write(out_dir, dataset, "smoke", smoke, prior),
               "heldout": _write(out_dir, dataset, "heldout", heldout, prior)}
    dst = split_paths(out_dir, dataset, "dev")
    for k in dst:
        dst[k].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dev_src[k], dst[k])
    summary["dev"] = {"n": len(dev_ids), "source": str(dev_src["manifest"].relative_to(REPO_ROOT))}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build dynamic bridge-cue pilot splits.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args(argv)
    summaries = {ds: build(ds, args.out_dir, local_files_only=not args.allow_download) for ds in C.PILOT_DATASETS}
    (args.out_dir / "splits_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
