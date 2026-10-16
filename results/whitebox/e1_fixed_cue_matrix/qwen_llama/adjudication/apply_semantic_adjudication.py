"""Folds a COMPLETED semantic-adjudication review back into new,
adjudicated answer-accuracy outputs.

**Do not run this script until human review of `semantic_adjudication_queue.csv`
is complete.** It validates that every one of the 543 rows has reached a
final state before doing anything else, and refuses to run otherwise (see
`validate_queue_complete` below).

Reused, unmodified: `analysis.answer_evaluation.TECHNICAL_FAILURE_REASONS`
(the same technical-failure exclusion the pre-adjudication pipeline uses),
`analysis.canonical.results_bundle.comparisons.COMPARISONS` (the same
comparison manifest, no second list). The pairing rule below is a direct
port of `analysis/canonical/results_bundle/answer_ext.py::_build_strictly_paired_example_ids`
and `analysis.answer_evaluation.compute_paired_answer_transitions`'s logic,
operating on CSV row dicts instead of `AnswerRecord` objects (the CSV
already carries every field either function needs except
`AnswerRecord.extractable`, which is not reconstructed here -- see the
`answer_summary_adjudicated.csv` note below).

Never overwrites `answer_run_level.csv` / `answer_summary.csv` /
`answer_paired_flips.csv` -- writes three new `*_adjudicated.csv` files
instead.
"""

from __future__ import annotations

import csv
from pathlib import Path

from analysis.answer_evaluation import TECHNICAL_FAILURE_REASONS
from analysis.canonical.results_bundle.comparisons import COMPARISONS

REPO_ROOT = Path(__file__).resolve().parents[2]
ADJUDICATION_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "qwen_llama_results_bundle" / "data"

QUEUE_PATH = ADJUDICATION_DIR / "semantic_adjudication_queue.csv"
KEY_PATH = ADJUDICATION_DIR / "semantic_adjudication_key.csv"
ANSWER_RUN_LEVEL_PATH = DATA_DIR / "answer_run_level.csv"

EXPECTED_TOTAL = 543
VALID_LABELS = {"Correct", "Incorrect", "Unclear"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_and_validate_review() -> dict[str, dict]:
    """Returns review_id -> {model, dataset, condition, example_id,
    adjudicated_label}, only after every completeness/consistency check
    passes. Raises (does not warn) on any violation."""
    queue_rows = _read_csv(QUEUE_PATH)
    key_rows = _read_csv(KEY_PATH)

    if len(queue_rows) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} queue rows, found {len(queue_rows)}. Stopping.")
    if len(key_rows) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} key rows, found {len(key_rows)}. Stopping.")

    queue_ids = [r["review_id"] for r in queue_rows]
    key_ids = [r["review_id"] for r in key_rows]
    if len(set(queue_ids)) != len(queue_ids):
        dupes = {i for i in queue_ids if queue_ids.count(i) > 1}
        raise AssertionError(f"Duplicate review_id(s) in queue: {dupes}")
    if len(set(key_ids)) != len(key_ids):
        dupes = {i for i in key_ids if key_ids.count(i) > 1}
        raise AssertionError(f"Duplicate review_id(s) in key: {dupes}")
    if set(queue_ids) != set(key_ids):
        raise AssertionError(
            f"Queue/key review_id sets do not match: "
            f"{len(set(queue_ids) - set(key_ids))} in queue only, "
            f"{len(set(key_ids) - set(queue_ids))} in key only"
        )

    # Completeness: every row must have reached Correct, Incorrect, or a
    # RESOLVED Unclear (review_status == "resolved" when the label is
    # Unclear; "pending"/"escalated" Unclear is not a final state).
    incomplete = []
    unknown_label = []
    for r in queue_rows:
        label = r["adjudicated_label"].strip()
        status = r["review_status"].strip()
        if label not in VALID_LABELS:
            unknown_label.append((r["review_id"], label))
            continue
        if label == "Unclear" and status != "resolved":
            incomplete.append((r["review_id"], label, status))
        elif label in ("Correct", "Incorrect") and status not in ("reviewed", "resolved"):
            incomplete.append((r["review_id"], label, status))

    if unknown_label:
        raise AssertionError(f"{len(unknown_label)} row(s) have an unrecognized adjudicated_label (must be Correct/Incorrect/Unclear): {unknown_label[:10]}")
    if incomplete:
        raise AssertionError(
            f"{len(incomplete)} row(s) are not in a final state -- human review is not complete. "
            f"Not proceeding. First few: {incomplete[:10]}"
        )

    key_by_id = {r["review_id"]: r for r in key_rows}
    review: dict[str, dict] = {}
    for r in queue_rows:
        k = key_by_id[r["review_id"]]
        review[r["review_id"]] = {
            "model": k["model"], "dataset": k["dataset"], "condition": k["condition"], "example_id": k["example_id"],
            "adjudicated_label": r["adjudicated_label"].strip(),
            "automatic_semantic_label": r["automatic_semantic_label"],
        }
    return review


def build_adjudicated_rows(review_by_id: dict[str, dict]) -> tuple[list[dict], dict]:
    """Returns (adjudicated run-level rows, stats dict). Every original row
    is preserved; only the 543 reviewed rows get an adjudicated label
    layered on top of (never replacing) the original automatic one."""
    original_rows = _read_csv(ANSWER_RUN_LEVEL_PATH)
    original_by_key = {(r["model"], r["dataset"], r["condition"], r["example_id"]): r for r in original_rows}

    # Unmatched-key check: every reviewed row must correspond to exactly
    # one original answer_run_level.csv row.
    unmatched = []
    override_by_key: dict[tuple, dict] = {}
    for review_id, r in review_by_id.items():
        key = (r["model"], r["dataset"], r["condition"], r["example_id"])
        if key not in original_by_key:
            unmatched.append((review_id, key))
            continue
        override_by_key[key] = r
    if unmatched:
        raise AssertionError(f"{len(unmatched)} reviewed row(s) do not match any row in {ANSWER_RUN_LEVEL_PATH.name}: {unmatched[:10]}")

    n_flagged_seen = 0
    adjudicated_rows: list[dict] = []
    for orig in original_rows:
        key = (orig["model"], orig["dataset"], orig["condition"], orig["example_id"])
        row = dict(orig)
        override = override_by_key.get(key)
        if override is not None:
            n_flagged_seen += 1
            label = override["adjudicated_label"]
            row["adjudicated_label"] = label
            row["adjudicated_semantic_correct"] = "True" if label == "Correct" else ("False" if label == "Incorrect" else "")
            row["adjudicated_excluded_unclear"] = "True" if label == "Unclear" else "False"
        else:
            # Not in the reviewed set: adjudicated view carries the
            # automatic label through unchanged.
            row["adjudicated_label"] = ""
            row["adjudicated_semantic_correct"] = "True" if orig["automatic_semantic_label"] == "correct" else "False"
            row["adjudicated_excluded_unclear"] = "False"
        adjudicated_rows.append(row)

    if n_flagged_seen != EXPECTED_TOTAL:
        raise AssertionError(f"Expected to apply exactly {EXPECTED_TOTAL} reviewed overrides, applied {n_flagged_seen}. Stopping.")

    return adjudicated_rows, {"n_total_rows": len(adjudicated_rows), "n_reviewed_applied": n_flagged_seen}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"Refusing to write an empty CSV to {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(rows: list[dict], correct_key: str, exclude_unclear: bool) -> list[dict]:
    """Same primary-denominator convention as
    `analysis.answer_evaluation.summarize_by_cell`: excludes
    TECHNICAL_FAILURE_REASONS from the denominator. `extractable` is not
    reconstructed here (not present in `answer_run_level.csv`), so
    `n_valid_answer`/`pct_valid_answer` are intentionally omitted from this
    summary rather than approximated -- see module docstring."""
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model"], r["dataset"], r["condition"]), []).append(r)

    out = []
    for (model, dataset, condition), recs in grouped.items():
        n_total = len(recs)
        primary = [r for r in recs if r["extraction_reason"] not in TECHNICAL_FAILURE_REASONS]
        if exclude_unclear:
            primary = [r for r in primary if r.get("adjudicated_excluded_unclear") != "True"]
        n_primary = len(primary)
        n_correct = sum(1 for r in primary if r[correct_key] == "True")
        n_needs_adjudication = sum(1 for r in primary if r["automatic_semantic_label"] == "needs_adjudication")
        em_vals = [float(r["exact_match"]) for r in primary if r["exact_match"] not in ("", "None")]
        f1_vals = [float(r["token_f1"]) for r in primary if r["token_f1"] not in ("", "None")]
        out.append(
            {
                "model": model, "dataset": dataset, "condition": condition,
                "n_total_rows": n_total,
                "n_primary_denominator": n_primary,
                "n_technical_failures_excluded": n_total - len(primary) if not exclude_unclear else (n_total - n_primary),
                "n_semantic_correct": n_correct,
                "pct_semantic_correct": (100.0 * n_correct / n_primary) if n_primary else None,
                "n_needs_adjudication": n_needs_adjudication,
                "exact_match_mean": (sum(em_vals) / len(em_vals)) if em_vals else None,
                "token_f1_mean": (sum(f1_vals) / len(f1_vals)) if f1_vals else None,
            }
        )
    return out


def build_strictly_paired_example_ids(rows: list[dict]) -> dict[tuple[str, str, str, str], set[str]]:
    by_model_dataset_example: dict[tuple[str, str, str], set[str]] = {}
    for r in rows:
        by_model_dataset_example.setdefault((r["model"], r["dataset"], r["example_id"]), set()).add(r["condition"])

    out: dict[tuple[str, str, str, str], set[str]] = {}
    for (model, dataset, example_id), conditions_present in by_model_dataset_example.items():
        for comp in COMPARISONS:
            if comp.control_condition in conditions_present and comp.cue_condition in conditions_present:
                key = (model, dataset, comp.control_condition, comp.cue_condition)
                out.setdefault(key, set()).add(example_id)
    return out


def compute_paired_flips(rows: list[dict], correct_key: str, exclude_unclear: bool) -> list[dict]:
    """Direct port of `analysis.answer_evaluation.compute_paired_answer_transitions`,
    reading CSV row dicts instead of `AnswerRecord` objects."""
    by_key = {(r["model"], r["dataset"], r["condition"], r["example_id"]): r for r in rows}
    strictly_paired = build_strictly_paired_example_ids(rows)

    out = []
    for (model, dataset, control, cue), eids in strictly_paired.items():
        both_correct = control_correct_cue_wrong = control_wrong_cue_correct = both_wrong = 0
        n_technical_excluded = 0
        n_unclear_excluded = 0
        for eid in eids:
            rec_a = by_key.get((model, dataset, control, eid))
            rec_b = by_key.get((model, dataset, cue, eid))
            if rec_a is None or rec_b is None:
                continue
            if rec_a["extraction_reason"] in TECHNICAL_FAILURE_REASONS or rec_b["extraction_reason"] in TECHNICAL_FAILURE_REASONS:
                n_technical_excluded += 1
                continue
            if exclude_unclear and (rec_a.get("adjudicated_excluded_unclear") == "True" or rec_b.get("adjudicated_excluded_unclear") == "True"):
                n_unclear_excluded += 1
                continue
            a_ok = rec_a[correct_key] == "True"
            b_ok = rec_b[correct_key] == "True"
            if a_ok and b_ok:
                both_correct += 1
            elif a_ok and not b_ok:
                control_correct_cue_wrong += 1
            elif not a_ok and b_ok:
                control_wrong_cue_correct += 1
            else:
                both_wrong += 1
        n = both_correct + control_correct_cue_wrong + control_wrong_cue_correct + both_wrong
        out.append(
            {
                "model": model, "dataset": dataset, "condition_a": control, "condition_b": cue,
                "n_pairs": n, "n_technical_excluded": n_technical_excluded,
                "n_unclear_excluded": n_unclear_excluded if exclude_unclear else 0,
                "both_correct": both_correct,
                "control_correct_cue_wrong": control_correct_cue_wrong,
                "control_wrong_cue_correct": control_wrong_cue_correct,
                "both_wrong": both_wrong,
                "net_harm": control_correct_cue_wrong - control_wrong_cue_correct,
                "net_harm_pct_points": (100.0 * (control_correct_cue_wrong - control_wrong_cue_correct) / n) if n else None,
            }
        )
    return out


def check_denominators_and_population(automatic_rows: list[dict], adjudicated_rows: list[dict], automatic_flips: list[dict], adjudicated_flips: list[dict]) -> None:
    """Fails if anything other than the deliberate, counted Unclear
    exclusion changed the row count, the technical-failure exclusion, or
    the paired population for any comparison."""
    if len(automatic_rows) != len(adjudicated_rows):
        raise AssertionError(f"Row count changed: automatic={len(automatic_rows)}, adjudicated={len(adjudicated_rows)}")

    auto_by_cell = {(f["model"], f["dataset"], f["condition_a"], f["condition_b"]): f for f in automatic_flips}
    adj_by_cell = {(f["model"], f["dataset"], f["condition_a"], f["condition_b"]): f for f in adjudicated_flips}
    if set(auto_by_cell) != set(adj_by_cell):
        raise AssertionError("Paired-comparison cell set changed between automatic and adjudicated flips -- stopping.")
    for cell, auto_f in auto_by_cell.items():
        adj_f = adj_by_cell[cell]
        auto_population = auto_f["n_pairs"] + auto_f["n_technical_excluded"]
        adj_population = adj_f["n_pairs"] + adj_f["n_technical_excluded"] + adj_f["n_unclear_excluded"]
        if auto_population != adj_population:
            raise AssertionError(
                f"Paired population changed for {cell} beyond the deliberate Unclear exclusion: "
                f"automatic total={auto_population}, adjudicated total={adj_population}. Stopping."
            )
        if auto_f["n_technical_excluded"] != adj_f["n_technical_excluded"]:
            raise AssertionError(f"Technical-failure exclusion count changed for {cell}: {auto_f['n_technical_excluded']} -> {adj_f['n_technical_excluded']}. Stopping.")


def main() -> int:
    print("Validating completed review...")
    review_by_id = load_and_validate_review()
    print(f"  {len(review_by_id)} rows validated as complete (Correct/Incorrect/resolved-Unclear)")

    print("Applying adjudication to answer_run_level.csv rows...")
    adjudicated_rows, stats = build_adjudicated_rows(review_by_id)
    print(f"  {stats}")

    automatic_rows = _read_csv(ANSWER_RUN_LEVEL_PATH)
    for r in automatic_rows:
        r["adjudicated_semantic_correct"] = "True" if r["automatic_semantic_label"] == "correct" else "False"
        r["adjudicated_excluded_unclear"] = "False"

    print("Building automatic vs adjudicated summaries...")
    automatic_summary = build_summary(automatic_rows, "adjudicated_semantic_correct", exclude_unclear=False)
    adjudicated_summary = build_summary(adjudicated_rows, "adjudicated_semantic_correct", exclude_unclear=True)

    print("Recomputing paired answer flips (automatic vs adjudicated)...")
    automatic_flips = compute_paired_flips(automatic_rows, "adjudicated_semantic_correct", exclude_unclear=False)
    adjudicated_flips = compute_paired_flips(adjudicated_rows, "adjudicated_semantic_correct", exclude_unclear=True)

    print("Checking denominators and paired populations are unchanged apart from the deliberate Unclear exclusion...")
    check_denominators_and_population(automatic_rows, adjudicated_rows, automatic_flips, adjudicated_flips)

    print("Writing outputs (new files, pre-adjudication CSVs untouched)...")
    _write_csv(DATA_DIR / "answer_run_level_adjudicated.csv", adjudicated_rows)
    _write_csv(DATA_DIR / "answer_summary_adjudicated.csv", adjudicated_summary)
    _write_csv(DATA_DIR / "answer_paired_flips_adjudicated.csv", adjudicated_flips)

    print("\n--- Automatic vs adjudicated semantic accuracy, pooled ---")
    auto_correct = sum(r["n_semantic_correct"] for r in automatic_summary)
    auto_n = sum(r["n_primary_denominator"] for r in automatic_summary)
    adj_correct = sum(r["n_semantic_correct"] for r in adjudicated_summary)
    adj_n = sum(r["n_primary_denominator"] for r in adjudicated_summary)
    print(f"Automatic:   {auto_correct}/{auto_n} = {100.0*auto_correct/auto_n:.2f}%")
    print(f"Adjudicated: {adj_correct}/{adj_n} = {100.0*adj_correct/adj_n:.2f}% (denominator excludes rows resolved Unclear)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
