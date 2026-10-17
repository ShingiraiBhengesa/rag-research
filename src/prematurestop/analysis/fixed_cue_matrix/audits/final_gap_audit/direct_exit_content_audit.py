"""Phase 3 of the final gap audit: content audit of every NO_TOOL_DIRECT_EXIT
row's actual researcher response text.

READ-ONLY: loads existing trace files only, never modifies them. Reuses
`prematurestop.evaluation.trace_parser.parse_trace_file` for episode/step
structure (does not reimplement it), and adds one new, narrowly-scoped
extraction step this task specifically needs and no existing module provides:
pulling the raw AIMessage `content=` text out of the SAME on_chain_end event
`trace_parser.py` already locates the `tool_calls=` kwarg in (via the same
literal-safe `repr_parsing.extract_last_kwarg` helper, reused unmodified).

The terminal-route DEFINITION is not touched here -- this only inspects,
after the fact, what text a NO_TOOL_DIRECT_EXIT row's terminal decision
actually contained. The route and the content category are two different
variables, reported as two different columns; the route is never renamed
based on content.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prematurestop.evaluation import config
from prematurestop.evaluation.completion_inventory import build_inventory
from prematurestop.evaluation.repr_parsing import LiteralScanError, extract_last_kwarg
from prematurestop.evaluation.trace_parser import _iter_odr_events, load_trace_json, parse_trace_file
from prematurestop.paths import RESULTS_E1

HERE = RESULTS_E1 / "audits" / "final_gap_audit"  # outputs
CANONICAL_CSV = HERE.parent / "canonical_run_level_completion.csv"

# --- deterministic content categories -------------------------------------
SUBSTANTIVE_DIRECT_ANSWER = "SUBSTANTIVE_DIRECT_ANSWER"
SUBSTANTIVE_RESEARCH_SUMMARY = "SUBSTANTIVE_RESEARCH_SUMMARY"
REFUSAL = "REFUSAL"
EMPTY_OR_WHITESPACE = "EMPTY_OR_WHITESPACE"
ERROR_MESSAGE = "ERROR_MESSAGE"
PROCESS_COMMENTARY = "PROCESS_COMMENTARY"
MALFORMED_TOOL_ATTEMPT = "MALFORMED_TOOL_ATTEMPT"
UNCLEAR = "UNCLEAR"

ERROR_PATTERNS = re.compile(
    r"\b(Error generating|Error synthesizing|Traceback \(most recent call|Exception:|"
    r"maximum retries exceeded|Error:\s|failed to generate|internal server error)\b",
    re.IGNORECASE,
)
REFUSAL_PATTERNS = re.compile(
    r"\b(I cannot|I can’t|I can't|I'm unable to|I am unable to|as an AI( language model)?|"
    r"I do not have access|I don't have access|cannot provide|unable to provide|"
    r"I apologize, but|I'm sorry, but|I am not able to)\b",
    re.IGNORECASE,
)
MALFORMED_TOOL_PATTERNS = re.compile(
    r'("name"\s*:\s*"(tavily_search|ResearchComplete|think_tool)")|'
    r"(<tool_call>)|(\{\s*\"tool\"\s*:)|(```json\s*\{\s*\"name\")|"
    r"(function_call\s*:)|(\bcall tool\b)|"
    r"(```tool_code)|(\btool_code\b)",  # Gemma-3-27B-IT's actual failure mode: it writes
    # its intended tool invocation as a "```tool_code\ntavily_search(...)```" markdown
    # block instead of a real structured tool call (see
    # analysis/canonical/CHANGELOG.md and
    # gemma27b_analysis/gemma27b_results_synthesis.md). Discovered by direct trace
    # inspection while auditing gemma27b's NO_TOOL_DIRECT_EXIT content 2026-09-13;
    # added here, not just handled ad hoc, so the same file-wide deterministic-rule
    # documentation this module already follows applies to it too.
    re.IGNORECASE,
)
PROCESS_COMMENTARY_PATTERNS = re.compile(
    r"^(i (will|have|am going to|now)|let me|based on (the|my) (research|findings)|"
    r"i have (gathered|collected|completed)|the (research|search) is (complete|done)|"
    r"i (have|now) (sufficient|enough) (information|evidence))",
    re.IGNORECASE,
)

PROCESS_COMMENTARY_MAX_LEN = 200
SUBSTANTIVE_SUMMARY_MIN_LEN = 600


def classify_content(text: str) -> tuple[str, bool]:
    """Returns (category, needs_human_review). Deterministic rules first;
    anything not confidently matched by a deterministic rule is flagged
    needs_human_review=True with a PRELIMINARY heuristic label only.
    """
    if text is None or not text.strip():
        return EMPTY_OR_WHITESPACE, False
    stripped = text.strip()

    if ERROR_PATTERNS.search(stripped):
        return ERROR_MESSAGE, False
    if MALFORMED_TOOL_PATTERNS.search(stripped):
        return MALFORMED_TOOL_ATTEMPT, False
    if REFUSAL_PATTERNS.search(stripped[:400]):
        return REFUSAL, False
    if len(stripped) <= PROCESS_COMMENTARY_MAX_LEN and PROCESS_COMMENTARY_PATTERNS.match(stripped):
        return PROCESS_COMMENTARY, False

    # Everything past this point requires semantic judgment -- deterministic
    # rules cannot reliably distinguish a genuinely substantive short answer
    # from process commentary that doesn't match the fixed phrase list above,
    # nor a research summary from a direct answer, purely from length. These
    # get a PRELIMINARY length-based guess and are flagged for human review.
    if len(stripped) >= SUBSTANTIVE_SUMMARY_MIN_LEN:
        return SUBSTANTIVE_RESEARCH_SUMMARY, True
    if len(stripped) <= 40:
        return UNCLEAR, True
    return SUBSTANTIVE_DIRECT_ANSWER, True


def _extract_content_at_step(events: list, target_step: int) -> Optional[str]:
    for step, ev in _iter_odr_events(events):
        if step != target_step or ev.get("name") != "researcher" or ev.get("event") != "on_chain_end":
            continue
        output = ev.get("data", {}).get("output")
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        try:
            extracted = extract_last_kwarg(text, "content")
        except LiteralScanError:
            return None
        if extracted is None:
            return None
        value = extracted.value
        return value if isinstance(value, str) else str(value)
    return None


def _normalize_answer_contains(text: str, gold: str, aliases: list[str]) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    t = norm(text or "")
    candidates = [gold] + (aliases or [])
    return any(norm(c) and norm(c) in t for c in candidates if c)


@dataclass
class DirectExitRow:
    model: str
    dataset: str
    condition: str
    example_id: str
    run_id: str
    researcher_episode_id: str
    first_terminal_post_observation_step: int
    pcr_eligible: bool
    trace_path: str


def _load_target_rows() -> list[DirectExitRow]:
    rows = []
    with open(CANONICAL_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["terminal_route"] != "NO_TOOL_DIRECT_EXIT":
                continue
            step = r["first_terminal_post_observation_step"]
            rows.append(
                DirectExitRow(
                    model=r["model"], dataset=r["dataset"], condition=r["condition"], example_id=r["example_id"],
                    run_id=r["run_id"], researcher_episode_id=r["researcher_episode_id"],
                    first_terminal_post_observation_step=int(step) if step else -1,
                    pcr_eligible=(r["pcr_eligible"] == "True"),
                    trace_path="",  # filled in by _attach_trace_paths
                )
            )
    return rows


def _attach_trace_paths(rows: list[DirectExitRow]) -> list[DirectExitRow]:
    inventory = build_inventory(config.DEFAULT_PRODUCTION_ROOT)
    by_key = {(r.model, r.dataset, r.condition, r.example_id): r for r in inventory.rows}
    out = []
    for row in rows:
        rr = by_key.get((row.model, row.dataset, row.condition, row.example_id))
        row.trace_path = rr.trace_path if rr else ""
        out.append(row)
    return out


def _process_row(row: DirectExitRow) -> dict:
    base = {
        "model": row.model, "dataset": row.dataset, "condition": row.condition, "example_id": row.example_id,
        "run_id": row.run_id, "researcher_episode_id": row.researcher_episode_id,
        "post_observation_completion_step": row.first_terminal_post_observation_step,
        "pcr_eligible": row.pcr_eligible,
    }
    if not row.trace_path:
        return {**base, "extraction_error": "trace_path_not_found", "content_category": UNCLEAR, "needs_human_review": False}
    try:
        full_path = config.resolve_stored_path(row.trace_path)
        pt = parse_trace_file(full_path)
        ep0 = pt.first_episode
        if ep0 is None or row.first_terminal_post_observation_step < 1 or row.first_terminal_post_observation_step > len(ep0.post_branch_decisions):
            return {**base, "extraction_error": "decision_index_out_of_range", "content_category": UNCLEAR, "needs_human_review": False}
        decision = ep0.post_branch_decisions[row.first_terminal_post_observation_step - 1]
        trace_raw = load_trace_json(full_path)
        content = _extract_content_at_step(trace_raw.get("events", []), decision.step)
    except Exception as exc:  # noqa: BLE001 -- must never crash the batch
        return {**base, "extraction_error": f"{type(exc).__name__}: {exc}", "content_category": UNCLEAR, "needs_human_review": False}

    if content is None:
        return {**base, "extraction_error": "content_kwarg_not_found_or_unparseable", "content_category": UNCLEAR, "needs_human_review": False}

    category, needs_review = classify_content(content)
    gold = pt.extracted_answer or ""
    # gold answer / alias containment: use the manifest-level gold answer if
    # available via the trace's own dependency-derived fields is out of
    # scope here (trace doesn't carry the raw manifest gold answer string
    # directly) -- record containment against the TRACE's own
    # extracted_answer/final_report only as a diagnostic, and separately
    # record whether the direct-exit content textually reappears in the
    # eventual final_report (downstream agreement), never claiming
    # correctness from either.
    final_report = pt.final_report or ""
    downstream_agrees = bool(content.strip()) and content.strip()[:120] in final_report

    return {
        **base,
        "response_length_chars": len(content),
        "response_length_words_approx": len(content.split()),
        "is_empty": not content.strip(),
        "contains_error_pattern": bool(ERROR_PATTERNS.search(content)),
        "contains_refusal_pattern": bool(REFUSAL_PATTERNS.search(content[:400])),
        "contains_malformed_tool_syntax": bool(MALFORMED_TOOL_PATTERNS.search(content)),
        "content_category": category,
        "needs_human_review": needs_review,
        "final_report_present": bool(final_report),
        "downstream_final_report_agrees_prefix_match": downstream_agrees,
        "extraction_error": "",
        "response_text_preview_first_300_chars": content.strip()[:300].replace("\n", " \\n "),
    }


def main() -> None:
    rows = _load_target_rows()
    print(f"Found {len(rows)} NO_TOOL_DIRECT_EXIT rows in canonical_run_level_completion.csv")
    rows = _attach_trace_paths(rows)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_process_row, r) for r in rows]
        for fut in as_completed(futures):
            results.append(fut.result())

    # -- full audit CSV --
    fieldnames: list[str] = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(HERE / "direct_exit_content_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # -- summary CSV: overall, by model, by dataset, by condition, by step bucket --
    def summarize(rows_subset: list[dict], group_label: str, group_value: str) -> dict:
        n = len(rows_subset)
        cat_counts = Counter(r["content_category"] for r in rows_subset)
        n_review = sum(1 for r in rows_subset if r["needs_human_review"])
        n_gold_present = sum(1 for r in rows_subset if r.get("downstream_final_report_agrees_prefix_match") == True)  # noqa: E712
        entry = {"group": group_label, "value": group_value, "n": n, "n_needs_human_review": n_review, "pct_needs_human_review": (100.0 * n_review / n) if n else None}
        for cat in (SUBSTANTIVE_DIRECT_ANSWER, SUBSTANTIVE_RESEARCH_SUMMARY, REFUSAL, EMPTY_OR_WHITESPACE, ERROR_MESSAGE, PROCESS_COMMENTARY, MALFORMED_TOOL_ATTEMPT, UNCLEAR):
            entry[f"n_{cat}"] = cat_counts.get(cat, 0)
            entry[f"pct_{cat}"] = (100.0 * cat_counts.get(cat, 0) / n) if n else None
        entry["pct_downstream_final_report_prefix_agrees"] = (100.0 * n_gold_present / n) if n else None
        return entry

    summary_rows = [summarize(results, "overall", "all")]
    by_model: dict[str, list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_step_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
        by_dataset[r["dataset"]].append(r)
        by_condition[r["condition"]].append(r)
        step = r.get("post_observation_completion_step")
        bucket = "step_1" if step == 1 else ("step_2_to_5" if isinstance(step, int) and 2 <= step <= 5 else ("step_6_plus" if isinstance(step, int) and step >= 6 else "unknown"))
        by_step_bucket[bucket].append(r)

    for model, rs in sorted(by_model.items()):
        summary_rows.append(summarize(rs, "model", model))
    for dataset, rs in sorted(by_dataset.items()):
        summary_rows.append(summarize(rs, "dataset", dataset))
    for condition, rs in sorted(by_condition.items()):
        summary_rows.append(summarize(rs, "condition", condition))
    for bucket, rs in sorted(by_step_bucket.items()):
        summary_rows.append(summarize(rs, "completion_step_bucket", bucket))
    for model in sorted(by_model):
        for bucket in ("step_1", "step_2_to_5", "step_6_plus"):
            rs = [r for r in by_model[model] if (
                (bucket == "step_1" and r.get("post_observation_completion_step") == 1) or
                (bucket == "step_2_to_5" and isinstance(r.get("post_observation_completion_step"), int) and 2 <= r["post_observation_completion_step"] <= 5) or
                (bucket == "step_6_plus" and isinstance(r.get("post_observation_completion_step"), int) and r["post_observation_completion_step"] >= 6)
            )]
            if rs:
                summary_rows.append(summarize(rs, f"model={model}|step_bucket", bucket))

    sfields: list[str] = []
    sseen = set()
    for r in summary_rows:
        for k in r.keys():
            if k not in sseen:
                sseen.add(k)
                sfields.append(k)
    with open(HERE / "direct_exit_content_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sfields, restval="")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # -- stratified human-review sample, blinded --
    review_pool = [r for r in results if r["needs_human_review"]]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in review_pool:
        step = r.get("post_observation_completion_step")
        step_bucket = "early" if (isinstance(step, int) and step <= 2) else "late"
        strata[(r["model"], r["dataset"], r["condition"], step_bucket)].append(r)

    rng = random.Random(config.DETERMINISTIC_SEED)
    sample_rows = []
    blinded_id = 0
    for key, rs in sorted(strata.items()):
        rng.shuffle(rs)
        take = rs[: min(3, len(rs))]  # up to 3 per stratum -- reproducible, seed-fixed
        for r in take:
            blinded_id += 1
            sample_rows.append(
                {
                    "blinded_id": f"REVIEW_{blinded_id:05d}",
                    "preliminary_auto_category": r["content_category"],
                    "response_text_preview": r["response_text_preview_first_300_chars"],
                    "response_length_chars": r["response_length_chars"],
                    "human_category": "",  # to be filled in by a human reviewer
                    "human_notes": "",
                    # de-anonymization key kept in a separate column block at
                    # the end so reviewers CAN work blinded by simply hiding
                    # these columns; not a separate file, to keep one join key.
                    "_model": r["model"], "_dataset": r["dataset"], "_condition": r["condition"],
                    "_example_id": r["example_id"], "_step": r["post_observation_completion_step"],
                }
            )
    with open(HERE / "direct_exit_human_review_sample.csv", "w", newline="", encoding="utf-8") as fh:
        if sample_rows:
            w = csv.DictWriter(fh, fieldnames=list(sample_rows[0].keys()))
            w.writeheader()
            w.writerows(sample_rows)
        else:
            fh.write("")

    print(f"Wrote direct_exit_content_audit.csv ({len(results)} rows), "
          f"direct_exit_content_summary.csv ({len(summary_rows)} groups), "
          f"direct_exit_human_review_sample.csv ({len(sample_rows)} stratified rows)")


if __name__ == "__main__":
    main()
