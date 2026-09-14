"""Read-only MuSiQue hop-distribution and proposed-sample audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_MUSIQUE_DATASET = "bdsaglam/musique"
DEFAULT_MUSIQUE_CONFIG = "answerable"
DEFAULT_MUSIQUE_SPLIT = "validation"
CANONICAL_FIELD_SEPARATOR = "\x1f"
CANONICAL_NEWLINE = "\n"
CANONICAL_FIELDS = ("dataset", "dataset_version", "source_split", "example_id", "hop_count")
PROPORTIONAL_STRATIFIED_TARGETS = {"2": 233, "3": 142, "4": 75}


def audit_musique_scope(
    *,
    n_sample: int = 450,
    seed: int = 20260820,
    local_files_only: bool = False,
    dev_id_paths: Iterable[Path] = (),
    include_ids: bool = False,
) -> dict:
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    ds = load_dataset(
        DEFAULT_MUSIQUE_DATASET,
        DEFAULT_MUSIQUE_CONFIG,
        split=DEFAULT_MUSIQUE_SPLIT,
        download_config=download_config,
    )

    dataset_version = str(ds.info.version or "")
    valid_rows = []
    invalid_rows = []
    schema_fields = sorted(ds.column_names)
    for ex in ds:
        validation = validate_musique_example(ex)
        row = {
            "id": str(ex.get("id")),
            "hop_count": len(ex.get("question_decomposition") or []),
            "valid": validation["valid"],
            "invalid_reasons": validation["invalid_reasons"],
        }
        if validation["valid"]:
            valid_rows.append(row)
        else:
            invalid_rows.append(row)

    _reject_duplicate_ids(valid_rows)
    candidate_a = _simple_random_sample(valid_rows, n_sample=n_sample, seed=seed)
    candidate_b = _proportional_stratified_sample(
        valid_rows,
        targets=PROPORTIONAL_STRATIFIED_TARGETS,
        seed=seed,
    )
    dev_ids = load_id_set(dev_id_paths)
    source_files = _source_file_reports(ds.cache_files)

    report = {
        "dataset": DEFAULT_MUSIQUE_DATASET,
        "config": DEFAULT_MUSIQUE_CONFIG,
        "split": DEFAULT_MUSIQUE_SPLIT,
        "dataset_version": dataset_version,
        "schema_fields": schema_fields,
        "canonical_serialization": {
            "fields": list(CANONICAL_FIELDS),
            "field_separator": "ASCII Unit Separator U+001F",
            "newline": "LF \\n after every record, including the final record",
            "encoding": "UTF-8",
            "sorting": "lexicographic by example_id, then numeric hop_count",
            "duplicate_id_policy": "reject duplicate example_id values before hashing",
        },
        "fields_used_for_dependencies": [
            "id",
            "question",
            "answer",
            "answer_aliases",
            "question_decomposition",
            "question_decomposition[].id",
            "question_decomposition[].paragraph_support_idx",
            "question_decomposition[].question",
            "question_decomposition[].answer",
            "paragraphs",
            "paragraphs[].idx if present, else paragraph-list index",
            "paragraphs[].title",
            "paragraphs[].paragraph_text",
            "paragraphs[].is_supporting",
        ],
        "total_examples": len(ds),
        "valid_examples": len(valid_rows),
        "invalid_examples": len(invalid_rows),
        "official_hop_distribution_valid": _counter_to_dict(row["hop_count"] for row in valid_rows),
        "official_hop_distribution_all": _counter_to_dict(row["hop_count"] for row in valid_rows + invalid_rows),
        "source_dataset_files": source_files,
        "source_dataset_file_sha256": source_files[0]["sha256"] if len(source_files) == 1 else "",
        "eligible_population_sha256": canonical_rows_sha256(
            valid_rows,
            dataset=DEFAULT_MUSIQUE_DATASET,
            dataset_version=dataset_version,
            split=DEFAULT_MUSIQUE_SPLIT,
        ),
        "sampling_code_sha256": _file_sha256(Path(__file__)),
        "candidate_a_simple_random_sample": _candidate_report(
            candidate_a,
            n_requested=n_sample,
            seed=seed,
            dataset_version=dataset_version,
            include_ids=include_ids,
            dev_ids=dev_ids,
            sampling_method="simple_random_sample",
            sampling_description=(
                "Fixed-seed simple random sample over all valid MuSiQue answerable "
                "validation examples; no hop-count filter."
            ),
        ),
        "candidate_b_proportional_stratified_sample": _candidate_report(
            candidate_b,
            n_requested=sum(PROPORTIONAL_STRATIFIED_TARGETS.values()),
            seed=seed,
            dataset_version=dataset_version,
            include_ids=include_ids,
            dev_ids=dev_ids,
            sampling_method="proportional_stratified_sample",
            sampling_description=(
                "Fixed-seed sampling within each hop stratum using targets "
                f"{PROPORTIONAL_STRATIFIED_TARGETS}."
            ),
        ),
        "sampling_policy_comparison": {
            "candidate_a_advantages": [
                "Simplest fixed-seed design.",
                "Lets the sample reflect ordinary random variation from the official split.",
                "Retains the currently proposed realized counts if the same seed is used.",
            ],
            "candidate_a_disadvantages": [
                "Per-hop counts can drift between seeds.",
                "The 4-hop stratum may be smaller than desired for stable diagnostics.",
            ],
            "candidate_b_advantages": [
                "Preserves the official distribution closely.",
                "Stabilizes per-hop analysis before production IDs are frozen.",
                "Makes the intended hop composition explicit and reproducible.",
            ],
            "candidate_b_disadvantages": [
                "Adds a small amount of sampling machinery.",
                "No longer represents pure simple random sampling variance.",
            ],
            "recommendation": "Candidate B, proportional stratified sampling, because it preserves the official distribution while keeping 2/3/4-hop analyses stable.",
        },
        "invalid_reason_counts": _counter_to_dict(reason for row in invalid_rows for reason in row["invalid_reasons"]),
    }
    report["proposed_sample"] = report["candidate_a_simple_random_sample"]
    return report


def canonical_rows_sha256(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    dataset_version: str,
    split: str,
) -> str:
    return hashlib.sha256(
        canonical_rows_bytes(rows, dataset=dataset, dataset_version=dataset_version, split=split)
    ).hexdigest()


def canonical_rows_bytes(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    dataset_version: str,
    split: str,
) -> bytes:
    records = []
    seen_ids: set[str] = set()
    for row in rows:
        example_id = str(row["id"])
        if example_id in seen_ids:
            raise ValueError(f"duplicate MuSiQue example id: {example_id}")
        seen_ids.add(example_id)
        fields = (dataset, dataset_version, split, example_id, str(int(row["hop_count"])))
        for field in fields:
            if CANONICAL_FIELD_SEPARATOR in field or "\n" in field or "\r" in field:
                raise ValueError(f"canonical field contains forbidden separator/newline: {field!r}")
        records.append(fields)
    records.sort(key=lambda values: (values[3], int(values[4])))
    text = "".join(CANONICAL_FIELD_SEPARATOR.join(values) + CANONICAL_NEWLINE for values in records)
    return text.encode("utf-8")


def _simple_random_sample(rows: list[dict], *, n_sample: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:n_sample]


def _proportional_stratified_sample(rows: list[dict], *, targets: Mapping[str, int], seed: int) -> list[dict]:
    by_hop: dict[str, list[dict]] = {}
    for row in rows:
        by_hop.setdefault(str(row["hop_count"]), []).append(row)
    selected = []
    for hop, n_target in sorted(targets.items(), key=lambda item: int(item[0])):
        stratum = sorted(by_hop.get(hop, []), key=lambda row: row["id"])
        if len(stratum) < int(n_target):
            raise ValueError(f"not enough {hop}-hop MuSiQue rows for target {n_target}: {len(stratum)} available")
        rng = random.Random(seed + int(hop) * 1_000_003)
        rng.shuffle(stratum)
        selected.extend(stratum[: int(n_target)])
    return selected


def _candidate_report(
    rows: list[dict],
    *,
    n_requested: int,
    seed: int,
    dataset_version: str,
    include_ids: bool,
    dev_ids: set[str],
    sampling_method: str,
    sampling_description: str,
) -> dict:
    selected_ids = {row["id"] for row in rows}
    overlap = sorted(selected_ids.intersection(dev_ids))
    return {
        "n_requested": n_requested,
        "n_selected": len(rows),
        "sampling_seed": seed,
        "sampling_method": sampling_method,
        "sampling_description": sampling_description,
        "hop_distribution": _counter_to_dict(row["hop_count"] for row in rows),
        "candidate_sample_sha256": canonical_rows_sha256(
            rows,
            dataset=DEFAULT_MUSIQUE_DATASET,
            dataset_version=dataset_version,
            split=DEFAULT_MUSIQUE_SPLIT,
        ),
        "example_ids": [row["id"] for row in rows] if include_ids else [],
        "ids_omitted_from_report": not include_ids,
        "dev_production_overlap_count": len(overlap),
        "dev_production_overlap_ids": overlap[:50],
    }


def _reject_duplicate_ids(rows: Iterable[Mapping[str, object]]) -> None:
    seen = set()
    duplicates = []
    for row in rows:
        example_id = str(row["id"])
        if example_id in seen:
            duplicates.append(example_id)
        seen.add(example_id)
    if duplicates:
        raise ValueError(f"duplicate MuSiQue ids: {duplicates[:10]}")


def _source_file_reports(cache_files: Iterable[Mapping[str, object]]) -> list[dict]:
    reports = []
    for row in cache_files:
        filename = row.get("filename")
        if not filename:
            continue
        path = Path(str(filename))
        reports.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    reports.sort(key=lambda item: item["path"])
    return reports


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_musique_example(ex: Mapping[str, object]) -> dict:
    reasons = []
    decomp = ex.get("question_decomposition")
    paragraphs = ex.get("paragraphs")
    if not isinstance(decomp, list) or not decomp:
        reasons.append("missing_question_decomposition")
    if not isinstance(paragraphs, list) or not paragraphs:
        reasons.append("missing_paragraphs")

    if isinstance(decomp, list) and isinstance(paragraphs, list):
        for hop_index, step in enumerate(decomp):
            if not isinstance(step, Mapping):
                reasons.append(f"hop_{hop_index}_not_mapping")
                continue
            support_idx = step.get("paragraph_support_idx")
            if support_idx is None:
                reasons.append(f"hop_{hop_index}_missing_paragraph_support_idx")
                continue
            try:
                support_idx_int = int(support_idx)
            except (TypeError, ValueError):
                reasons.append(f"hop_{hop_index}_noninteger_paragraph_support_idx")
                continue
            if support_idx_int < 0 or support_idx_int >= len(paragraphs):
                reasons.append(f"hop_{hop_index}_support_idx_out_of_range")
                continue
            paragraph = paragraphs[support_idx_int]
            if not isinstance(paragraph, Mapping):
                reasons.append(f"hop_{hop_index}_paragraph_not_mapping")
                continue
            if not paragraph.get("paragraph_text"):
                reasons.append(f"hop_{hop_index}_missing_paragraph_text")
            if "is_supporting" in paragraph and not paragraph.get("is_supporting"):
                reasons.append(f"hop_{hop_index}_support_idx_not_marked_supporting")

    return {"valid": not reasons, "invalid_reasons": reasons}


def load_id_set(paths: Iterable[Path]) -> set[str]:
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    ids.add(raw)
                    continue
                ids.add(str(row.get("id") or row.get("name") or row.get("example_id") or raw))
    return ids


def _counter_to_dict(values: Iterable[object]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MuSiQue hop distribution without writing a manifest.")
    parser.add_argument("--n-sample", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dev-ids", nargs="*", type=Path, default=())
    parser.add_argument("--include-ids", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_musique_scope(
        n_sample=args.n_sample,
        seed=args.seed,
        local_files_only=args.local_files_only,
        dev_id_paths=args.dev_ids,
        include_ids=args.include_ids,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
