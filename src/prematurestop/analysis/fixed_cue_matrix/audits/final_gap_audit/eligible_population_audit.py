"""Phase 2 of the final gap audit: pre-sampling eligible-population counts
and exact sampling-reproduction check.

READ-ONLY: imports and calls the exact functions from
`prematurestop.data.manifests` (never reimplements the eligibility
or sampling logic), but NEVER calls `freeze_dataset_manifest`/
`freeze_all_manifests` (which write files) and never calls `write_jsonl`.
Uses `local_files_only=True` when loading each dataset, so this script can
only succeed against already-cached HuggingFace dataset files -- it will
raise, not fetch anything, if a dataset is not already cached.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from prematurestop.data import manifests as pm
from prematurestop.data.musique_scope_audit import validate_musique_example
from prematurestop.paths import RESULTS_E1

HERE = RESULTS_E1 / "audits" / "final_gap_audit"  # outputs
REPO_ROOT = pm.REPO_ROOT
PRODUCTION_ROOT = pm.DEFAULT_PRODUCTION_ROOT


def classify_and_collect(dataset: str, ds) -> dict:
    """Mirrors `_eligible_candidates`'s filter chain exactly (same function
    calls, same order), but additionally records WHY each rejected row was
    rejected and counts duplicate IDs -- `_eligible_candidates` itself does
    not expose rejection reasons, only the final eligible list.
    """
    reasons: Counter = Counter()
    candidates: list[pm.Candidate] = []
    seen_ids: dict[str, int] = {}
    duplicate_ids: list[str] = []
    parse_failures = 0

    for source_index, ex in enumerate(ds):
        try:
            docs, evidence = pm._build_rows(dataset, ex, source_index)
        except Exception:
            parse_failures += 1
            reasons["parse_exception_in_build_rows"] += 1
            continue
        if not docs:
            reasons["no_corpus_documents"] += 1
            continue
        if not evidence.dependencies:
            reasons["no_evidence_dependencies"] += 1
            continue
        if any(dep.ambiguous for dep in evidence.dependencies):
            reasons["ambiguous_dependency"] += 1
            continue
        if not str(evidence.answer).strip():
            reasons["empty_gold_answer"] += 1
            continue
        if dataset == "musique" and not validate_musique_example(ex)["valid"]:
            reasons["musique_invalid_example"] += 1
            continue
        if dataset == "multihoprag" and not pm._multihoprag_oracle_allowed(ex):
            reasons["multihoprag_oracle_not_allowed"] += 1
            continue

        seen_ids[evidence.example_id] = seen_ids.get(evidence.example_id, 0) + 1
        if seen_ids[evidence.example_id] > 1:
            duplicate_ids.append(evidence.example_id)
            reasons["duplicate_id_among_eligible"] += 1
            continue

        candidates.append(
            pm.Candidate(
                source_index=source_index,
                source_id=evidence.example_id,
                hop_count=evidence.hop_count,
                example=ex,
                evidence=evidence,
            )
        )

    return {
        "raw_split_size": len(ds),
        "rows_successfully_parsed": len(ds) - parse_failures,
        "rows_rejected_total": len(ds) - len(candidates),
        "eligible_population": len(candidates),
        "rejection_reasons": dict(reasons),
        "duplicate_ids": duplicate_ids,
        "candidates": candidates,
    }


def load_existing_manifest_ids(dataset: str) -> list[str]:
    path = PRODUCTION_ROOT / "manifests" / f"{pm.MANIFEST_DATASET_NAMES[dataset]}_n450.jsonl"
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ids.append(row["example_id"])
    return ids


def main() -> None:
    rows_out = []
    repro_lines = ["# Sampling reproduction check (Phase 2)", ""]

    for dataset in pm.DATASETS:
        print(f"Loading {dataset} (local_files_only=True, read-only)...")
        ds = pm._load_dataset(dataset, local_files_only=True)
        dataset_version = str(ds.info.version or "unknown")

        result = classify_and_collect(dataset, ds)
        candidates = result["candidates"]

        if len(candidates) < pm.N_SAMPLE:
            repro_lines.append(f"## {dataset}: FAILED -- only {len(candidates)} eligible candidates, need {pm.N_SAMPLE}")
            rows_out.append(
                {
                    "dataset": dataset,
                    "raw_split_size": result["raw_split_size"],
                    "rows_successfully_parsed": result["rows_successfully_parsed"],
                    "eligible_population_before_sampling": len(candidates),
                    "rows_rejected_total": result["rows_rejected_total"],
                    **{f"rejected_{k}": v for k, v in result["rejection_reasons"].items()},
                    "duplicate_ids_rejected": len(result["duplicate_ids"]),
                    "final_sampled_count": None,
                    "sampling_seed": pm.DEFAULT_SEED,
                    "sampling_method": "musique_stratified" if dataset == "musique" else "simple_random_shuffle",
                    "manifest_sha256_reproduced": None,
                    "manifest_sha256_existing": None,
                    "reproduction_matches": False,
                }
            )
            continue

        selected = pm._select_candidates(dataset, candidates, seed=pm.DEFAULT_SEED)
        selected_sorted = sorted(selected, key=lambda row: row.source_id)
        reproduced_ids_in_order = [c.source_id for c in selected_sorted]

        manifest_rows = [pm._manifest_row(dataset, candidate, ordinal) for ordinal, candidate in enumerate(selected_sorted)]
        reproduced_manifest_bytes = "\n".join(json.dumps(r, sort_keys=True) for r in manifest_rows).encode("utf-8")
        import hashlib

        reproduced_manifest_sha256 = hashlib.sha256(reproduced_manifest_bytes).hexdigest()

        existing_path = PRODUCTION_ROOT / "manifests" / f"{pm.MANIFEST_DATASET_NAMES[dataset]}_n450.jsonl"
        existing_ids = load_existing_manifest_ids(dataset)
        existing_manifest_sha256 = pm._file_sha256(existing_path)

        population_digest = pm._population_sha256(dataset, candidates, dataset_version=dataset_version)
        sample_digest = pm._sample_sha256(dataset, selected_sorted, dataset_version=dataset_version)

        existing_summary_path = PRODUCTION_ROOT / "provenance" / f"{pm.MANIFEST_DATASET_NAMES[dataset]}_manifest_summary.json"
        existing_summary = json.loads(existing_summary_path.read_text()) if existing_summary_path.exists() else {}
        existing_population_digest = existing_summary.get("eligible_population_sha256")
        existing_sample_digest = existing_summary.get("ordered_sample_id_sha256")

        id_set_matches = set(reproduced_ids_in_order) == set(existing_ids)
        id_order_matches = reproduced_ids_in_order == existing_ids
        population_digest_matches = population_digest == existing_population_digest
        sample_digest_matches = sample_digest == existing_sample_digest

        corpus_path = PRODUCTION_ROOT / "corpus" / f"{pm.MANIFEST_DATASET_NAMES[dataset]}_corpus.jsonl"
        evidence_path = PRODUCTION_ROOT / "evidence" / f"{pm.MANIFEST_DATASET_NAMES[dataset]}_evidence.jsonl"
        existing_corpus_sha256 = pm._file_sha256(corpus_path) if corpus_path.exists() else None
        existing_evidence_sha256 = pm._file_sha256(evidence_path) if evidence_path.exists() else None

        divergence = None
        if not id_set_matches:
            missing = set(existing_ids) - set(reproduced_ids_in_order)
            extra = set(reproduced_ids_in_order) - set(existing_ids)
            divergence = f"ID SET mismatch: {len(missing)} existing IDs not reproduced, {len(extra)} reproduced IDs not in existing manifest"
        elif not id_order_matches:
            for i, (a, b) in enumerate(zip(reproduced_ids_in_order, existing_ids)):
                if a != b:
                    divergence = f"ID ORDER mismatch first at index {i}: reproduced={a!r} vs existing={b!r}"
                    break
        elif not population_digest_matches:
            divergence = f"eligible_population_sha256 mismatch: reproduced={population_digest} vs existing={existing_population_digest}"
        elif not sample_digest_matches:
            divergence = f"ordered_sample_id_sha256 mismatch: reproduced={sample_digest} vs existing={existing_sample_digest}"

        overall_reproduces = id_set_matches and id_order_matches and population_digest_matches and sample_digest_matches

        repro_lines.append(f"## {dataset}")
        repro_lines.append(f"- Eligible population (reproduced, read-only): {len(candidates)}")
        repro_lines.append(f"- Example-ID SET match: {id_set_matches}")
        repro_lines.append(f"- Example-ID ORDER match: {id_order_matches}")
        repro_lines.append(f"- eligible_population_sha256 match: {population_digest_matches} (reproduced={population_digest}, existing={existing_population_digest})")
        repro_lines.append(f"- ordered_sample_id_sha256 match: {sample_digest_matches} (reproduced={sample_digest}, existing={existing_sample_digest})")
        repro_lines.append(f"- Manifest file SHA-256 note: this script's `manifest_rows` are recomputed independently and hashed as newline-joined sorted-key JSON for a structural check; this does NOT need to equal the on-disk manifest file's raw-bytes SHA-256 ({existing_manifest_sha256}) because JSON key order in the raw file may differ -- the authoritative reproducibility signals are the ID set/order and the two SHA-256 digests already embedded in the production provenance file (`ordered_sample_id_sha256`, `eligible_population_sha256`), both checked above.")
        repro_lines.append(f"- Overall exact reproduction: {'PASS' if overall_reproduces else 'FAIL -- first divergence: ' + (divergence or 'unknown')}")
        repro_lines.append("")

        rows_out.append(
            {
                "dataset": dataset,
                "raw_split_size": result["raw_split_size"],
                "rows_successfully_parsed": result["rows_successfully_parsed"],
                "eligible_population_before_sampling": len(candidates),
                "rows_rejected_total": result["rows_rejected_total"],
                **{f"rejected_{k}": v for k, v in result["rejection_reasons"].items()},
                "duplicate_ids_rejected": len(result["duplicate_ids"]),
                "final_sampled_count": len(selected_sorted),
                "sampling_seed": pm.DEFAULT_SEED,
                "sampling_method": "musique_stratified" if dataset == "musique" else "simple_random_shuffle",
                "manifest_sha256_reproduced_structural": reproduced_manifest_sha256,
                "manifest_sha256_existing_file": existing_manifest_sha256,
                "existing_corpus_sha256": existing_corpus_sha256,
                "existing_evidence_sha256": existing_evidence_sha256,
                "eligible_population_sha256_match": population_digest_matches,
                "ordered_sample_id_sha256_match": sample_digest_matches,
                "example_id_set_match": id_set_matches,
                "example_id_order_match": id_order_matches,
                "reproduction_matches": overall_reproduces,
            }
        )

    fieldnames: list[str] = []
    seen = set()
    for r in rows_out:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(HERE / "eligible_population_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    (HERE / "sampling_reproduction_check.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")
    print("Done. Wrote eligible_population_counts.csv and sampling_reproduction_check.md")


if __name__ == "__main__":
    main()
