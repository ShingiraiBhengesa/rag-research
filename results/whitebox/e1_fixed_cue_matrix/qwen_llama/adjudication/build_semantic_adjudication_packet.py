"""Builds the blinded semantic-adjudication packet for the 543 rows where
`qwen_llama_results_bundle/data/answer_run_level.csv::needs_adjudication_flag
== True`.

Automatic semantic accuracy already exists for every row
(`automatic_semantic_label`, computed by
`attack/answer_scoring/answer_evaluator_v2.py::semantic_accuracy`) -- this
script does not recompute it. It only assembles a blinded review queue for
the subset the automatic rule itself flagged as ambiguous.

Reused, unmodified: `qwen_llama_results_bundle/data/answer_run_level.csv`
(source), `results/manifests/{dataset}_n450.jsonl` (question text, joined
by `example_id`), `analysis/config.py::DETERMINISTIC_SEED` (row shuffle).

Fails loudly (raises, does not warn-and-continue) on: a flagged row whose
`example_id` has no manifest entry, a manifest with a duplicate
`example_id` key, or a final queue/key review_id mismatch.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANSWER_RUN_LEVEL = REPO_ROOT / "qwen_llama_results_bundle" / "data" / "answer_run_level.csv"
MANIFEST_DIR = REPO_ROOT / "results" / "manifests"
OUTPUT_DIR = Path(__file__).resolve().parent

DETERMINISTIC_SEED = 20260910  # analysis/config.py::DETERMINISTIC_SEED, reused verbatim

EXPECTED_TOTAL = 543
EXPECTED_COUNTS_BY_MODEL_DATASET = {
    ("llama8b", "2wiki"): 56,
    ("llama8b", "hotpot"): 107,
    ("llama8b", "multihoprag"): 93,
    ("llama8b", "musique"): 91,
    ("qwen14b", "2wiki"): 41,
    ("qwen14b", "hotpot"): 38,
    ("qwen14b", "multihoprag"): 38,
    ("qwen14b", "musique"): 79,
}


def load_manifest_questions(dataset: str) -> dict[str, str]:
    path = MANIFEST_DIR / f"{dataset}_n450.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for dataset {dataset!r}: {path}")
    questions: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            question = row.get("question")
            if question is None:
                raise ValueError(f"{path}:{line_no}: example_id={example_id!r} has no 'question' field")
            if example_id in questions:
                raise ValueError(
                    f"{path}: duplicate example_id {example_id!r} at line {line_no} "
                    f"(first seen with question={questions[example_id]!r}) -- manifest join key must be unique"
                )
            questions[example_id] = question
    return questions


def load_flagged_rows() -> list[dict]:
    if not ANSWER_RUN_LEVEL.exists():
        raise FileNotFoundError(f"Source answer table not found: {ANSWER_RUN_LEVEL}")
    with open(ANSWER_RUN_LEVEL, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    flagged = [r for r in rows if r["needs_adjudication_flag"] == "True"]
    return flagged


def build_packet() -> tuple[list[dict], list[dict]]:
    flagged = load_flagged_rows()

    # Verify per-cell counts before doing anything else -- this task's own
    # explicit expected values, checked against the fresh count rather than
    # assumed.
    actual_counts: dict[tuple[str, str], int] = {}
    for r in flagged:
        key = (r["model"], r["dataset"])
        actual_counts[key] = actual_counts.get(key, 0) + 1
    mismatches = []
    for key, expected in EXPECTED_COUNTS_BY_MODEL_DATASET.items():
        actual = actual_counts.get(key, 0)
        if actual != expected:
            mismatches.append((key, expected, actual))
    for key in actual_counts:
        if key not in EXPECTED_COUNTS_BY_MODEL_DATASET:
            mismatches.append((key, 0, actual_counts[key]))
    if mismatches:
        raise AssertionError(
            f"Flagged-row counts do not match the expected per-(model,dataset) breakdown: {mismatches}. "
            f"Stopping -- do not build a packet against an unexpected row set."
        )

    if len(flagged) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected exactly {EXPECTED_TOTAL} flagged rows, found {len(flagged)}. Stopping.")

    # Join question text per dataset (memoized -- each manifest read once).
    questions_by_dataset: dict[str, dict[str, str]] = {}
    for dataset in {r["dataset"] for r in flagged}:
        questions_by_dataset[dataset] = load_manifest_questions(dataset)

    missing_joins = []
    for r in flagged:
        q_map = questions_by_dataset[r["dataset"]]
        if r["example_id"] not in q_map:
            missing_joins.append((r["model"], r["dataset"], r["condition"], r["example_id"]))
    if missing_joins:
        raise AssertionError(
            f"{len(missing_joins)} flagged row(s) have no matching manifest question -- "
            f"failing loudly per spec, not silently dropping. First few: {missing_joins[:5]}"
        )

    # Deterministic shuffle: sort by the row's own stable composite key
    # first (so shuffling is reproducible regardless of the CSV's on-disk
    # order), then shuffle with the repo's own deterministic seed.
    ordered = sorted(flagged, key=lambda r: (r["model"], r["dataset"], r["condition"], r["example_id"]))
    rng = random.Random(DETERMINISTIC_SEED)
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    queue_rows: list[dict] = []
    key_rows: list[dict] = []
    for i, r in enumerate(shuffled, start=1):
        review_id = f"ADJ-{i:04d}"
        question = questions_by_dataset[r["dataset"]][r["example_id"]]
        queue_rows.append(
            {
                "review_id": review_id,
                "dataset": r["dataset"],
                "question": question,
                "gold_answer": r["gold_answer"],
                "gold_aliases": r["gold_aliases"],
                "extracted_short_answer": r["extracted_short_answer"],
                "raw_final_report": r["raw_final_report"],
                "automatic_semantic_label": r["automatic_semantic_label"],
                "adjudicated_label": "",
                "adjudication_reason": "",
                "reviewer_id": "",
                "review_status": "pending",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "model": r["model"],
                "dataset": r["dataset"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "composite_key": f"{r['model']}|{r['dataset']}|{r['condition']}|{r['example_id']}",
            }
        )

    # Final structural check: queue and key must carry exactly the same
    # unique review_id set.
    queue_ids = {q["review_id"] for q in queue_rows}
    key_ids = {k["review_id"] for k in key_rows}
    if queue_ids != key_ids:
        raise AssertionError(f"Queue/key review_id mismatch: {len(queue_ids)} vs {len(key_ids)}, symmetric diff size {len(queue_ids ^ key_ids)}")
    if len(queue_ids) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} unique review_ids, got {len(queue_ids)}")

    # Blind-queue column check -- assert none of the forbidden identifying
    # columns leaked in.
    forbidden = {"model", "condition", "run_id", "original_run_id"}
    for row in queue_rows:
        leaked = forbidden & set(row.keys())
        if leaked:
            raise AssertionError(f"Blinded queue leaked forbidden column(s): {leaked}")

    return queue_rows, key_rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise AssertionError(f"Refusing to write an empty CSV to {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    queue_rows, key_rows = build_packet()
    write_csv(queue_rows, OUTPUT_DIR / "semantic_adjudication_queue.csv")
    write_csv(key_rows, OUTPUT_DIR / "semantic_adjudication_key.csv")
    print(f"Wrote {len(queue_rows)} rows to semantic_adjudication_queue.csv and semantic_adjudication_key.csv")
    print("Per-(model,dataset) counts verified against the expected breakdown -- see EXPECTED_COUNTS_BY_MODEL_DATASET.")
    print("No human label was prefilled; every review_status is 'pending'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
