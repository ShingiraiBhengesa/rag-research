"""Dev / full (and optional smoke) splits for the dynamic bridge-cue pilot.

dev    = the adaptive pilot's existing dev split (10 per dataset, seed
         20261005; never used in production). Lambda is fitted on its cue
         ratings, then all six arms run on it as the end-to-end check.
full   = the canonical production n450 questions per dataset, copied BYTE FOR
         BYTE with the canonical corpus and evidence files
         (results/{manifests,corpus,evidence}/), so d1, retrieval, and gold
         answers are identical to the 12-condition study. SHA-256 of each
         source and copy is recorded and asserted equal.
smoke  = optional, 3 per dataset from questions never used anywhere; not part
         of the dev -> full sequence.
dev, smoke and full are disjoint (asserted). Nothing is chosen or dropped by
outcome.
"""

from __future__ import annotations

import argparse
import hashlib
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
CANONICAL = {k: REPO_ROOT / "results" / sub for k, sub in
             (("manifest", "manifests"), ("corpus", "corpus"), ("evidence", "evidence"))}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical_source(dataset: str) -> dict[str, Path]:
    return {"manifest": CANONICAL["manifest"] / f"{dataset}_n450.jsonl",
            "corpus": CANONICAL["corpus"] / f"{dataset}_corpus.jsonl",
            "evidence": CANONICAL["evidence"] / f"{dataset}_evidence.jsonl"}


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

    src, dst_full = canonical_source(dataset), split_paths(out_dir, dataset, "full")
    full_hashes = {}
    for k in src:
        dst_full[k].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src[k], dst_full[k])
        a, b = _sha(src[k]), _sha(dst_full[k])
        assert a == b, f"copy mismatch {src[k]}"
        full_hashes[k] = {"source": str(src[k].relative_to(REPO_ROOT)), "sha256": a}
    full_ids = {json.loads(l)["example_id"] for l in dst_full["manifest"].read_text().splitlines() if l.strip()}
    assert len(full_ids) == C.FULL_PER_DATASET, f"{dataset}: canonical manifest has {len(full_ids)} rows"
    assert not (smoke_ids & full_ids) and not (dev_ids & (smoke_ids | full_ids)), "split overlap"
    summary = {"dataset": dataset, "seed": C.SPLIT_SEED, "n_eligible": len(candidates),
               "n_pool_after_exclusions": len(pool),
               "smoke": _write(out_dir, dataset, "smoke", smoke, prior),
               "full": {"n": len(full_ids), "files": full_hashes}}
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
