"""Blinded semantic-adjudication packet for any results bundle.

Collects the rows of `<bundle>/data/answer_run_level.csv` whose automatic
semantic label was flagged ambiguous (`needs_adjudication_flag == True`),
joins each to its question text from the frozen manifest, shuffles them with
the repo's deterministic seed, and writes two files to `<bundle>/adjudication/`:

* `semantic_adjudication_queue.csv` -- what a reviewer sees: question, gold
  answer, the system's answer and report, empty label columns. No model,
  condition or run identifiers (checked).
* `semantic_adjudication_key.csv` -- maps each `review_id` back to
  (model, dataset, condition, example_id), for after the review.

It also writes `flagged_counts.csv` (rows per model x dataset) so the packet's
size is recorded next to it. It never relabels anything: automatic labels
already exist for every row. It fails loudly on a flagged row with no manifest
question, a duplicate manifest key, or a queue/key mismatch.

Generalises the original Qwen/Llama-only script in
`results/whitebox/e1_fixed_cue_matrix/qwen_llama/adjudication/`.

    python -m prematurestop.analysis.fixed_cue_matrix.results_bundle.adjudication_packet \\
        --bundle-root results/whitebox/e1_fixed_cue_matrix/gemma_gptoss
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.paths import LOCAL_MANIFESTS

DETERMINISTIC_SEED = 20260910  # same seed as the Qwen/Llama packet
FORBIDDEN_QUEUE_COLUMNS = {"model", "condition", "run_id", "original_run_id"}


def load_manifest_questions(manifest_dir: Path, dataset: str) -> dict[str, str]:
    path = manifest_dir / f"{dataset}_n450.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for dataset {dataset!r}: {path}")
    questions: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if row.get("question") is None:
                raise ValueError(f"{path}:{line_no}: example_id={example_id!r} has no 'question' field")
            if example_id in questions:
                raise ValueError(f"{path}: duplicate example_id {example_id!r} at line {line_no}")
            questions[example_id] = row["question"]
    return questions


def build_packet(answer_run_level: Path, manifest_dir: Path) -> tuple[list[dict], list[dict], Counter]:
    with answer_run_level.open(newline="", encoding="utf-8") as fh:
        flagged = [r for r in csv.DictReader(fh) if r["needs_adjudication_flag"] == "True"]
    if not flagged:
        raise AssertionError(f"No flagged rows in {answer_run_level}; nothing to adjudicate")
    counts = Counter((r["model"], r["dataset"]) for r in flagged)

    questions = {ds: load_manifest_questions(manifest_dir, ds) for ds in {r["dataset"] for r in flagged}}
    missing = [(r["model"], r["dataset"], r["condition"], r["example_id"])
               for r in flagged if r["example_id"] not in questions[r["dataset"]]]
    if missing:
        raise AssertionError(f"{len(missing)} flagged row(s) have no manifest question; first: {missing[:5]}")

    # Sort by a stable key first so the shuffle does not depend on CSV row order.
    ordered = sorted(flagged, key=lambda r: (r["model"], r["dataset"], r["condition"], r["example_id"]))
    random.Random(DETERMINISTIC_SEED).shuffle(ordered)

    queue, key = [], []
    for i, r in enumerate(ordered, start=1):
        review_id = f"ADJ-{i:04d}"
        queue.append({
            "review_id": review_id, "dataset": r["dataset"],
            "question": questions[r["dataset"]][r["example_id"]],
            "gold_answer": r["gold_answer"], "gold_aliases": r["gold_aliases"],
            "extracted_short_answer": r["extracted_short_answer"], "raw_final_report": r["raw_final_report"],
            "automatic_semantic_label": r["automatic_semantic_label"],
            "adjudicated_label": "", "adjudication_reason": "", "reviewer_id": "", "review_status": "pending",
        })
        key.append({
            "review_id": review_id, "model": r["model"], "dataset": r["dataset"],
            "condition": r["condition"], "example_id": r["example_id"],
            "composite_key": f"{r['model']}|{r['dataset']}|{r['condition']}|{r['example_id']}",
        })

    if {q["review_id"] for q in queue} != {k["review_id"] for k in key}:
        raise AssertionError("queue/key review_id mismatch")
    leaked = FORBIDDEN_QUEUE_COLUMNS & set(queue[0])
    if leaked:
        raise AssertionError(f"blinded queue leaked identifying column(s): {leaked}")
    return queue, key, counts


def _write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-root", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, default=LOCAL_MANIFESTS)
    args = p.parse_args()

    queue, key, counts = build_packet(args.bundle_root / "data" / "answer_run_level.csv", args.manifest_dir)
    out = args.bundle_root / "adjudication"
    out.mkdir(parents=True, exist_ok=True)
    _write(queue, out / "semantic_adjudication_queue.csv")
    _write(key, out / "semantic_adjudication_key.csv")
    _write([{"model": m, "dataset": d, "display_model": config.DISPLAY_MODEL_NAMES.get(m, m), "n_flagged": n}
            for (m, d), n in sorted(counts.items())], out / "flagged_counts.csv")
    print(f"Wrote {len(queue)} rows to {out}; every review_status is 'pending'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
