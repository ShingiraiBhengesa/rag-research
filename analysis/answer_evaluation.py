"""Sections 12-13: final-answer evaluation.

Scores ONLY the text produced by `final_report_generation` (`ParsedTrace.final_report`)
-- never `ResearchComplete`, no-tool-exit text, compression text, or
supervisor messages. Reuses the existing, already-validated
`attack.answer_scoring.answer_evaluator_v2.evaluate_answer_v2` (imported
read-only, never modified) as the primary semantic-accuracy procedure, per
the task's explicit instruction to reuse the validated evaluator rather
than re-implementing answer scoring.

Gold answers/aliases are NOT present on the `runs/*.jsonl` result rows
(verified: those rows carry `gold_answer` but no aliases) -- they are read
from the frozen per-dataset manifests
(`production/manifests/<dataset>_n450.jsonl`), which is also the source of
the exact benchmark `question` text the evaluator's condition-blind
extractor requires.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis import config
from analysis.completion_inventory import RowRecord
from analysis.trace_parser import parse_trace_file
from attack.answer_scoring.answer_evaluator_v2 import EVALUATOR_VERSION, evaluate_answer_v2


@dataclass
class ManifestEntry:
    question: str
    answer: str
    answer_aliases: tuple[str, ...]


_MANIFEST_CACHE: dict[str, dict[str, ManifestEntry]] = {}


def load_manifest(dataset: str, production_root: Path) -> dict[str, ManifestEntry]:
    if dataset in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[dataset]
    path = production_root / "manifests" / f"{dataset}_n450.jsonl"
    entries: dict[str, ManifestEntry] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries[str(row["example_id"])] = ManifestEntry(
                question=str(row.get("question") or ""),
                answer=str(row.get("answer") or ""),
                answer_aliases=tuple(row.get("answer_aliases") or ()),
            )
    _MANIFEST_CACHE[dataset] = entries
    return entries


@dataclass
class AnswerRecord:
    model: str
    dataset: str
    condition: str
    example_id: str
    reached_final_report_generation: bool
    raw_final_report: str
    extracted_short_answer: Optional[str]
    extractable: bool
    extraction_reason: str
    semantic_label: str
    semantic_correct: bool
    semantic_reason: str
    exact_match: float
    token_f1: float
    gold_containment: bool
    gold_answer: str
    evaluator_version: str = EVALUATOR_VERSION


def build_answer_record(
    model: str, dataset: str, condition: str, example_id: str, trace_path: str, production_root: Path
) -> AnswerRecord:
    manifest = load_manifest(dataset, production_root)
    entry = manifest.get(example_id)
    full_path = trace_path if Path(trace_path).is_absolute() else config.REPO_ROOT / trace_path
    pt = parse_trace_file(Path(full_path))

    if entry is None or not pt.reached_final_report_generation or not pt.final_report:
        return AnswerRecord(
            model=model, dataset=dataset, condition=condition, example_id=example_id,
            reached_final_report_generation=pt.reached_final_report_generation,
            raw_final_report=pt.final_report, extracted_short_answer=None, extractable=False,
            extraction_reason="no_manifest_entry" if entry is None else "no_final_report",
            semantic_label="incorrect", semantic_correct=False,
            semantic_reason="no_manifest_entry" if entry is None else "no_final_report",
            exact_match=0.0, token_f1=0.0, gold_containment=False,
            gold_answer=entry.answer if entry else "",
        )

    evaluation = evaluate_answer_v2(
        question=entry.question, raw_final_answer=pt.final_report,
        gold_answer=entry.answer, gold_aliases=entry.answer_aliases,
    )
    return AnswerRecord(
        model=model, dataset=dataset, condition=condition, example_id=example_id,
        reached_final_report_generation=True, raw_final_report=pt.final_report,
        extracted_short_answer=evaluation.extraction.short_answer,
        extractable=evaluation.extraction.extractable,
        extraction_reason=evaluation.extraction.reason,
        semantic_label=evaluation.semantic.semantic_label,
        semantic_correct=evaluation.semantic.semantic_correct,
        semantic_reason=evaluation.semantic.reason,
        exact_match=evaluation.official_scores.exact_match,
        token_f1=evaluation.official_scores.token_f1,
        gold_containment=evaluation.gold_containment,
        gold_answer=entry.answer,
    )


def _build_one_args(args: tuple) -> AnswerRecord:
    model, dataset, condition, example_id, trace_path, production_root_str = args
    return build_answer_record(model, dataset, condition, example_id, trace_path, Path(production_root_str))


def build_answer_records(rows: list[RowRecord], production_root: Path, max_workers: int = 10) -> list[AnswerRecord]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = [(r.model, r.dataset, r.condition, r.example_id, r.trace_path, str(production_root)) for r in rows]
    records: list[AnswerRecord] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_build_one_args, a) for a in args]
        for fut in as_completed(futures):
            records.append(fut.result())
    return records


TECHNICAL_FAILURE_REASONS = {"no_manifest_entry", "no_final_report"}


def summarize_by_cell(records: list[AnswerRecord]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[AnswerRecord]] = {}
    for r in records:
        grouped.setdefault((r.model, r.dataset, r.condition), []).append(r)

    rows = []
    for (model, dataset, condition), recs in grouped.items():
        n_total = len(recs)
        # Primary accuracy excludes unresolved technical failures (Section 12/14).
        primary = [r for r in recs if r.extraction_reason not in TECHNICAL_FAILURE_REASONS]
        n_primary = len(primary)
        n_valid_answer = sum(1 for r in primary if r.extractable)
        n_correct = sum(1 for r in primary if r.semantic_correct)
        n_needs_adjudication = sum(1 for r in primary if r.semantic_label == "needs_adjudication")
        em_mean = sum(r.exact_match for r in primary) / n_primary if n_primary else None
        f1_mean = sum(r.token_f1 for r in primary) / n_primary if n_primary else None
        gold_containment_rate = 100.0 * sum(1 for r in primary if r.gold_containment) / n_primary if n_primary else None
        rows.append(
            {
                "model": model, "dataset": dataset, "condition": condition,
                "n_total_rows": n_total,
                "n_primary_denominator": n_primary,
                "n_technical_failures_excluded": n_total - n_primary,
                "n_valid_answer": n_valid_answer,
                "pct_valid_answer": (100.0 * n_valid_answer / n_primary) if n_primary else None,
                "n_semantic_correct": n_correct,
                "pct_semantic_correct": (100.0 * n_correct / n_primary) if n_primary else None,
                "n_needs_adjudication": n_needs_adjudication,
                "exact_match_mean": em_mean,
                "token_f1_mean": f1_mean,
                "gold_containment_pct_diagnostic_only": gold_containment_rate,
            }
        )
    return rows


def compute_paired_answer_transitions(
    records: list[AnswerRecord],
    strictly_paired_example_ids: dict[tuple[str, str, str, str], set[str]],
    comparisons: list[tuple[str, str]],
) -> list[dict]:
    """Section 13: for each strictly-paired (baseline, condition) comparison,
    classify every paired example into both-correct / control-correct-attack-wrong
    / control-wrong-attack-correct / both-wrong, using ONLY the strict
    shared-prefix pairs (never the full per-condition sample) so this can be
    read as a genuine paired causal transition, not a marginal difference.
    """
    by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in records}
    rows = []
    grouped_keys: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in strictly_paired_example_ids:
        model, dataset, baseline, cond = key
        grouped_keys.setdefault((model, dataset), []).append((baseline, cond))

    for (model, dataset), pairs in grouped_keys.items():
        for baseline, cond in pairs:
            if (baseline, cond) not in comparisons:
                continue
            eids = strictly_paired_example_ids[(model, dataset, baseline, cond)]
            both_correct = control_correct_attack_wrong = control_wrong_attack_correct = both_wrong = 0
            n_technical_excluded = 0
            for eid in eids:
                rec_a = by_key.get((model, dataset, baseline, eid))
                rec_b = by_key.get((model, dataset, cond, eid))
                if rec_a is None or rec_b is None:
                    continue
                if rec_a.extraction_reason in TECHNICAL_FAILURE_REASONS or rec_b.extraction_reason in TECHNICAL_FAILURE_REASONS:
                    n_technical_excluded += 1
                    continue
                a_ok, b_ok = rec_a.semantic_correct, rec_b.semantic_correct
                if a_ok and b_ok:
                    both_correct += 1
                elif a_ok and not b_ok:
                    control_correct_attack_wrong += 1
                elif not a_ok and b_ok:
                    control_wrong_attack_correct += 1
                else:
                    both_wrong += 1
            n = both_correct + control_correct_attack_wrong + control_wrong_attack_correct + both_wrong
            rows.append(
                {
                    "model": model, "dataset": dataset, "condition_a": baseline, "condition_b": cond,
                    "n_pairs": n, "n_technical_excluded": n_technical_excluded,
                    "both_correct": both_correct,
                    "control_correct_attack_wrong": control_correct_attack_wrong,
                    "control_wrong_attack_correct": control_wrong_attack_correct,
                    "both_wrong": both_wrong,
                    "net_harm": (control_correct_attack_wrong - control_wrong_attack_correct),
                    "net_harm_pct_points": (100.0 * (control_correct_attack_wrong - control_wrong_attack_correct) / n) if n else None,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    _write_csv(path, rows)


def write_predictions_jsonl(records: list[AnswerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(
                json.dumps(
                    {
                        "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                        "reached_final_report_generation": r.reached_final_report_generation,
                        "raw_final_report": r.raw_final_report,
                        "extracted_short_answer": r.extracted_short_answer,
                        "extractable": r.extractable, "extraction_reason": r.extraction_reason,
                        "semantic_label": r.semantic_label, "semantic_correct": r.semantic_correct,
                        "semantic_reason": r.semantic_reason,
                        "exact_match": r.exact_match, "token_f1": r.token_f1,
                        "gold_containment": r.gold_containment, "gold_answer": r.gold_answer,
                        "evaluator_version": r.evaluator_version,
                    }
                )
                + "\n"
            )
