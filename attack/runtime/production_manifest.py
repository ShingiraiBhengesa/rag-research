"""Freeze production manifests, corpus rows, and evidence rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from attack.runtime.corpus import (
    EvidenceRow,
    build_2wiki_rows,
    build_hotpot_rows,
    build_musique_rows,
    build_multihoprag_rows,
    write_jsonl,
)
from attack.runtime.musique_scope_audit import (
    PROPORTIONAL_STRATIFIED_TARGETS,
    canonical_rows_sha256,
    validate_musique_example,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTION_ROOT = REPO_ROOT / "results"
DEFAULT_SEED = 20260820
N_SAMPLE = 450
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")
MANIFEST_DATASET_NAMES = {"hotpot": "hotpot", "2wiki": "2wiki", "musique": "musique", "multihoprag": "multihoprag"}


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_id: str
    hop_count: int
    example: Mapping[str, object]
    evidence: EvidenceRow


def freeze_all_manifests(
    *,
    production_root: Path = DEFAULT_PRODUCTION_ROOT,
    seed: int = DEFAULT_SEED,
    local_files_only: bool = False,
) -> dict:
    production_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = production_root / "manifests"
    corpus_dir = production_root / "corpus"
    evidence_dir = production_root / "evidence"
    provenance_dir = production_root / "provenance"
    for path in (manifest_dir, corpus_dir, evidence_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset in DATASETS:
        summary = freeze_dataset_manifest(
            dataset,
            production_root=production_root,
            seed=seed,
            local_files_only=local_files_only,
        )
        summaries.append(summary)

    combined = {
        "production_root": str(production_root),
        "seed": seed,
        "n_sample_per_dataset": N_SAMPLE,
        "datasets": summaries,
        "all_manifests_exist": all(Path(row["manifest_path"]).exists() for row in summaries),
        "all_manifests_have_450_unique_ids": all(row["n_selected"] == N_SAMPLE and row["unique_example_ids"] == N_SAMPLE for row in summaries),
        "total_examples": sum(row["n_selected"] for row in summaries),
    }
    (provenance_dir / "manifest_provenance.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return combined


def freeze_dataset_manifest(
    dataset: str,
    *,
    production_root: Path = DEFAULT_PRODUCTION_ROOT,
    seed: int = DEFAULT_SEED,
    local_files_only: bool = False,
) -> dict:
    ds = _load_dataset(dataset, local_files_only=local_files_only)
    dataset_version = str(ds.info.version or "unknown")
    candidates = _eligible_candidates(dataset, ds)
    if len(candidates) < N_SAMPLE:
        raise ValueError(f"{dataset} has only {len(candidates)} eligible candidates; need {N_SAMPLE}")

    selected = _select_candidates(dataset, candidates, seed=seed)
    selected = sorted(selected, key=lambda row: row.source_id)
    _reject_duplicate_ids(candidate.source_id for candidate in selected)

    manifest_rows = [_manifest_row(dataset, candidate, ordinal) for ordinal, candidate in enumerate(selected)]
    corpus_rows = []
    evidence_rows = []
    for candidate in selected:
        docs, evidence = _build_rows(dataset, candidate.example, candidate.source_index)
        corpus_rows.extend(docs)
        evidence_rows.append(evidence)

    manifest_path = production_root / "manifests" / f"{MANIFEST_DATASET_NAMES[dataset]}_n450.jsonl"
    corpus_path = production_root / "corpus" / f"{MANIFEST_DATASET_NAMES[dataset]}_corpus.jsonl"
    evidence_path = production_root / "evidence" / f"{MANIFEST_DATASET_NAMES[dataset]}_evidence.jsonl"
    summary_path = production_root / "provenance" / f"{MANIFEST_DATASET_NAMES[dataset]}_manifest_summary.json"

    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(corpus_path, [row.to_json() for row in corpus_rows])
    write_jsonl(evidence_path, [row.to_json() for row in evidence_rows])

    source_files = _source_file_reports(ds.cache_files)
    population_digest = _population_sha256(dataset, candidates, dataset_version=dataset_version)
    sample_digest = _sample_sha256(dataset, selected, dataset_version=dataset_version)
    if dataset == "musique":
        expected = "1cc2cce1e67761819ced7e093dc7fd81905850da93614776250a7882b93e29ff"
        if sample_digest != expected:
            raise ValueError(f"MuSiQue Candidate B SHA mismatch: {sample_digest} != {expected}")

    overlap = prior_sample_overlap(dataset, {candidate.source_id for candidate in selected})
    summary = {
        "dataset": dataset,
        "manifest_dataset_name": MANIFEST_DATASET_NAMES[dataset],
        "dataset_identifier": _dataset_identifier(dataset),
        "config": _dataset_config(dataset),
        "split": _dataset_split(dataset),
        "upstream_revision": "unknown",
        "dataset_version": dataset_version,
        "seed": seed,
        "n_selected": len(selected),
        "unique_example_ids": len({candidate.source_id for candidate in selected}),
        "manifest_path": str(manifest_path),
        "corpus_path": str(corpus_path),
        "evidence_path": str(evidence_path),
        "source_dataset_files": source_files,
        "source_file_sha256": source_files[0]["sha256"] if len(source_files) == 1 else "",
        "eligible_population_sha256": population_digest,
        "sampling_code_sha256": _file_sha256(Path(__file__)),
        "ordered_sample_id_sha256": sample_digest,
        "evidence_annotation_validity": {
            "all_selected_valid": all(row["evidence_valid"] for row in manifest_rows),
            "ambiguous_dependency_rows": sum(1 for row in evidence_rows if any(dep.ambiguous for dep in row.dependencies)),
        },
        "answer_reference_availability": {
            "answer_available": sum(1 for row in manifest_rows if row["answer_available"]),
            "answer_alias_rows": sum(1 for row in manifest_rows if row["answer_aliases"]),
        },
        "hop_distribution": _counter_to_dict(candidate.hop_count for candidate in selected),
        "type_distribution": _type_distribution(dataset, selected),
        "prior_sample_overlap": overlap,
        "non_overlapping_subset_sensitivity_analysis": {
            "required": any(value["count"] > 0 for value in overlap.values() if isinstance(value, Mapping)),
            "plan": "Report primary production results on the frozen sample and secondary sensitivity on rows excluding historical v1/v2 overlap.",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_dataset(dataset: str, *, local_files_only: bool):
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    if dataset == "hotpot":
        return load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", download_config=download_config)
    if dataset == "2wiki":
        return load_dataset("voidful/2WikiMultihopQA", split="validation", download_config=download_config)
    if dataset == "musique":
        return load_dataset("bdsaglam/musique", "answerable", split="validation", download_config=download_config)
    if dataset == "multihoprag":
        return load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", download_config=download_config)
    raise ValueError(f"unsupported dataset: {dataset}")


def _eligible_candidates(dataset: str, ds) -> list[Candidate]:
    candidates = []
    for source_index, ex in enumerate(ds):
        try:
            docs, evidence = _build_rows(dataset, ex, source_index)
        except Exception:
            continue
        if not docs or not evidence.dependencies:
            continue
        if any(dep.ambiguous for dep in evidence.dependencies):
            continue
        if not str(evidence.answer).strip():
            continue
        if dataset == "musique" and not validate_musique_example(ex)["valid"]:
            continue
        if dataset == "multihoprag" and not _multihoprag_oracle_allowed(ex):
            continue
        candidates.append(
            Candidate(
                source_index=source_index,
                source_id=evidence.example_id,
                hop_count=evidence.hop_count,
                example=ex,
                evidence=evidence,
            )
        )
    _reject_duplicate_ids(candidate.source_id for candidate in candidates)
    return candidates


def _build_rows(dataset: str, ex: Mapping[str, object], source_index: int):
    if dataset == "hotpot":
        return build_hotpot_rows(ex)
    if dataset == "2wiki":
        return build_2wiki_rows(ex)
    if dataset == "musique":
        return build_musique_rows(ex)
    if dataset == "multihoprag":
        return build_multihoprag_rows(ex, row_index=source_index)[:2]
    raise ValueError(f"unsupported dataset: {dataset}")


def _select_candidates(dataset: str, candidates: list[Candidate], *, seed: int) -> list[Candidate]:
    if dataset == "musique":
        by_hop: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            by_hop.setdefault(str(candidate.hop_count), []).append(candidate)
        selected = []
        for hop, n_target in sorted(PROPORTIONAL_STRATIFIED_TARGETS.items(), key=lambda item: int(item[0])):
            stratum = sorted(by_hop.get(hop, []), key=lambda row: row.source_id)
            rng = random.Random(seed + int(hop) * 1_000_003)
            rng.shuffle(stratum)
            selected.extend(stratum[: int(n_target)])
        return selected
    rng = random.Random(seed + sum(ord(char) for char in dataset))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:N_SAMPLE]


def _manifest_row(dataset: str, candidate: Candidate, ordinal: int) -> dict:
    evidence = candidate.evidence
    row = {
        "dataset": dataset,
        "example_id": candidate.source_id,
        "source_index": candidate.source_index,
        "sample_ordinal": ordinal,
        "question": evidence.question,
        "answer": evidence.answer,
        "answer_aliases": list(evidence.answer_aliases),
        "hop_count": evidence.hop_count,
        "evidence_dependency_count": len(evidence.dependencies),
        "required_dependency_ids": [dep.dependency_id for dep in evidence.dependencies],
        "required_paragraph_ids": list(dict.fromkeys(dep.supporting_paragraph_id or dep.document_id for dep in evidence.dependencies)),
        "answer_available": bool(str(evidence.answer).strip()),
        "evidence_valid": bool(evidence.dependencies) and not any(dep.ambiguous for dep in evidence.dependencies),
        "source_dataset": _dataset_identifier(dataset),
        "source_split": _dataset_split(dataset),
        "upstream_revision": "unknown",
        "production_manifest": True,
    }
    row["manifest_row_sha256"] = _json_sha256({key: value for key, value in row.items() if key != "manifest_row_sha256"})
    return row


def _population_sha256(dataset: str, candidates: list[Candidate], *, dataset_version: str) -> str:
    return canonical_rows_sha256(
        [{"id": candidate.source_id, "hop_count": candidate.hop_count} for candidate in candidates],
        dataset=_dataset_identifier(dataset),
        dataset_version=dataset_version,
        split=_dataset_split(dataset),
    )


def _sample_sha256(dataset: str, candidates: list[Candidate], *, dataset_version: str) -> str:
    return canonical_rows_sha256(
        [{"id": candidate.source_id, "hop_count": candidate.hop_count} for candidate in candidates],
        dataset=_dataset_identifier(dataset),
        dataset_version=dataset_version,
        split=_dataset_split(dataset),
    )


def _dataset_identifier(dataset: str) -> str:
    return {
        "hotpot": "hotpotqa/hotpot_qa",
        "2wiki": "voidful/2WikiMultihopQA",
        "musique": "bdsaglam/musique",
        "multihoprag": "yixuantt/MultiHopRAG",
    }[dataset]


def _dataset_config(dataset: str) -> str:
    return {"hotpot": "distractor", "2wiki": "default", "musique": "answerable", "multihoprag": "MultiHopRAG"}[dataset]


def _dataset_split(dataset: str) -> str:
    return "train" if dataset == "multihoprag" else "validation"


def _multihoprag_oracle_allowed(ex: Mapping[str, object]) -> bool:
    evidence_list = list(ex.get("evidence_list") or [])
    return bool(evidence_list) and all(
        isinstance(row, Mapping) and str(row.get("fact") or row.get("text") or "").strip()
        for row in evidence_list
    )


def multihoprag_oracle_audit(production_root: Path = DEFAULT_PRODUCTION_ROOT) -> dict:
    evidence_path = production_root / "evidence" / "multihoprag_evidence.jsonl"
    if not evidence_path.exists():
        return {"allowed": False, "reason": "missing multihoprag evidence file", "n": 0}
    rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ambiguous = [
        row["example_id"]
        for row in rows
        if any(dep.get("ambiguous") for dep in row.get("dependencies", []))
        or not row.get("dependencies")
    ]
    return {
        "allowed": not ambiguous and len(rows) == N_SAMPLE,
        "reason": "" if not ambiguous and len(rows) == N_SAMPLE else "ambiguous or missing dependencies",
        "n": len(rows),
        "ambiguous_count": len(ambiguous),
        "ambiguous_example_ids": ambiguous[:20],
    }


def prior_sample_overlap(dataset: str, production_ids: set[str]) -> dict[str, dict]:
    # NOTE (2026-09-14 top-level reorg): v1_exploratory/v2_evaluation now live
    # under archive/legacy_results/ (moved out of the publish-facing tree,
    # never deleted) -- this sensitivity check still finds them there, so it
    # is not silently degraded by the archival move. See analysis/CHANGELOG.md.
    paths = {
        "v1_exploratory": [REPO_ROOT / "archive" / "legacy_results" / "pcr_full_completion_v1" / "samples" / f"{MANIFEST_DATASET_NAMES[dataset]}_n450.jsonl"],
        "v2_evaluation": sorted((REPO_ROOT / "archive" / "legacy_results").glob(f"**/*v2*{MANIFEST_DATASET_NAMES[dataset]}*.jsonl")),
        "cue_development": [REPO_ROOT / "results" / "nonproduction_validation" / "dev_bundle" / "dev_examples.jsonl"],
        "runtime_gate": [REPO_ROOT / "results" / "nonproduction_validation" / "dev_bundle" / "smoke_examples.jsonl"],
    }
    report = {}
    for label, label_paths in paths.items():
        seen_ids = set()
        existing_paths = []
        for path in label_paths:
            if not path.exists():
                continue
            existing_paths.append(str(path))
            seen_ids.update(_ids_from_jsonl(path, dataset))
        overlap = sorted(production_ids.intersection(seen_ids))
        report[label] = {"count": len(overlap), "ids": overlap[:50], "paths": existing_paths}
    return report


def _ids_from_jsonl(path: Path, dataset: str) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            row_dataset = row.get("source") or row.get("dataset")
            if row_dataset and str(row_dataset) != dataset:
                continue
            value = row.get("example_id") or row.get("name") or row.get("id") or row.get("_id")
            if value is not None:
                ids.add(str(value))
    return ids


def _source_file_reports(cache_files: Iterable[Mapping[str, object]]) -> list[dict]:
    reports = []
    for row in cache_files:
        filename = row.get("filename")
        if not filename:
            continue
        path = Path(str(filename))
        reports.append({"path": str(path), "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    return sorted(reports, key=lambda row: row["path"])


def _type_distribution(dataset: str, candidates: list[Candidate]) -> dict[str, int]:
    if dataset != "2wiki":
        return {}
    values = []
    for candidate in candidates:
        raw_type = candidate.example.get("type") if isinstance(candidate.example, Mapping) else None
        values.append(str(raw_type or "unknown"))
    return _counter_to_dict(values)


def _counter_to_dict(values: Iterable[object]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def _reject_duplicate_ids(ids: Iterable[str]) -> None:
    seen = set()
    duplicates = []
    for value in ids:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate example ids: {duplicates[:10]}")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze production manifests and retrieval artifacts.")
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    report = freeze_all_manifests(
        production_root=args.production_root,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
