"""Section 6: exhaustive, mutually-exclusive termination-route taxonomy.

Every valid AND invalid completed row gets exactly one category. `other`
and `technical error` are never combined (a real bug in the parent
package's earlier display was fixed for exactly this reason -- see
pcr.py's module history). Categories, in classification order:

- RESEARCH_COMPLETE / RESEARCHER_NO_TOOL_EXIT: the episode's actual overall
  outcome (`PCRRecord.overall_terminal_route`, which credits a terminal
  decision regardless of whether the episode ever branched -- distinct from
  the PCR-specific `terminal_route`, which is scoped to post-branch steps
  only and excludes non-branching episodes from that narrower analysis).
- TOOL_OR_PARSER_FAILURE: episode never branched because every search
  attempt errored out (e.g. Llama's stringified-list tool-arg bug) --
  verified as a real, distinct, systematic failure mode, never counted as
  the researcher "choosing" not to search.
- TRACE_LOGGING_FAILURE: either a structural parse failure while building
  episodes (e.g. mismatched on_chain_start/on_chain_end counts, usually a
  mid-decision truncation) or a branch that was reached but never observed
  a terminal decision in the recorded trace -- confirmed by direct
  inspection of one real example that this can happen even when
  `tool_call_iterations` is nowhere near the configured cap and the run
  demonstrably completed (`final_report` present), i.e. it is a gap in
  what got logged, not evidence the run actually failed.
- EXECUTION_FAILURE: the row's own production result was `final_status ==
  "error"` (no trace was ever produced) -- these are the already-tracked
  "invalid completed rows", included here for the exhaustive audit but
  never in `valid_n`.
- OTHER_VALID_TERMINATION: catch-all for a branch-reached episode with a
  genuinely different resolution we haven't specifically named above
  (kept separate from failures so it is never miscounted as broken).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.evaluation.completion_inventory import InventoryResult
from prematurestop.evaluation.pcr import PCRRecord

ROUTE_RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
ROUTE_NO_TOOL_EXIT = "RESEARCHER_NO_TOOL_EXIT"
ROUTE_TOOL_OR_PARSER_FAILURE = "TOOL_OR_PARSER_FAILURE"
ROUTE_TRACE_LOGGING_FAILURE = "TRACE_LOGGING_FAILURE"
ROUTE_EXECUTION_FAILURE = "EXECUTION_FAILURE"
ROUTE_OTHER_VALID_TERMINATION = "OTHER_VALID_TERMINATION"

ALL_ROUTES = (
    ROUTE_RESEARCH_COMPLETE,
    ROUTE_NO_TOOL_EXIT,
    ROUTE_TOOL_OR_PARSER_FAILURE,
    ROUTE_TRACE_LOGGING_FAILURE,
    ROUTE_EXECUTION_FAILURE,
    ROUTE_OTHER_VALID_TERMINATION,
)


def classify_valid_record(record: PCRRecord) -> str:
    if not record.parse_ok:
        return ROUTE_TRACE_LOGGING_FAILURE
    if record.overall_terminal_route == config.ROUTE_RESEARCH_COMPLETE:
        return ROUTE_RESEARCH_COMPLETE
    if record.overall_terminal_route == config.ROUTE_NO_TOOL_EXIT:
        return ROUTE_NO_TOOL_EXIT
    if record.no_branch_reason == "search_tool_error":
        return ROUTE_TOOL_OR_PARSER_FAILURE
    if record.no_branch_reason == "structural_parse_failure":
        return ROUTE_TRACE_LOGGING_FAILURE
    if record.branch_reached and record.overall_terminal_route is None:
        # Branch happened, episode 0 has decisions, but none of them (nor
        # any later one) ever resolved to RC/no-tool in the recorded trace.
        return ROUTE_TRACE_LOGGING_FAILURE
    return ROUTE_OTHER_VALID_TERMINATION


@dataclass
class RouteAuditRow:
    model: str
    dataset: str
    condition: str
    route: str
    n: int


def build_termination_route_table(pcr_records: list[PCRRecord], inventory: InventoryResult) -> list[dict]:
    """One row per (model, dataset, condition, example_id) with its category,
    covering both valid rows (from pcr_records) and invalid/execution-failed
    rows (from the inventory's invalid row list) so every completed row --
    valid or not -- gets exactly one category.
    """
    rows: list[dict] = []
    for r in pcr_records:
        rows.append(
            {
                "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                "route": classify_valid_record(r), "is_valid_row": True,
                "overall_terminal_route": r.overall_terminal_route, "no_branch_reason": r.no_branch_reason,
                "parse_errors": r.parse_errors,
            }
        )
    valid_keys = {(r.model, r.dataset, r.condition, r.example_id) for r in pcr_records}
    for inv_row in inventory.rows:
        key = (inv_row.model, inv_row.dataset, inv_row.condition, inv_row.example_id)
        if key in valid_keys:
            continue  # already covered above as a valid row
        if inv_row.final_status == "ok" and not inv_row.error and inv_row.trace_exists:
            continue  # should have been in pcr_records; not an execution failure
        rows.append(
            {
                "model": inv_row.model, "dataset": inv_row.dataset, "condition": inv_row.condition,
                "example_id": inv_row.example_id, "route": ROUTE_EXECUTION_FAILURE, "is_valid_row": False,
                "overall_terminal_route": None, "no_branch_reason": None,
                "parse_errors": inv_row.error,
            }
        )
    return rows


def summarize_by_cell(route_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {r: 0 for r in ALL_ROUTES})
    for row in route_rows:
        grouped[(row["model"], row["dataset"], row["condition"])][row["route"]] += 1

    summary = []
    for (model, dataset, condition), counts in grouped.items():
        total = sum(counts.values())
        valid_n = total - counts[ROUTE_EXECUTION_FAILURE]
        entry = {"model": model, "dataset": dataset, "condition": condition, "total_n": total, "valid_n": valid_n}
        for route in ALL_ROUTES:
            entry[f"n_{route}"] = counts[route]
            entry[f"pct_{route}_of_total"] = 100.0 * counts[route] / total if total else None
        summary.append(entry)
    return summary


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


def write_termination_route_outputs(route_rows: list[dict], summary_rows: list[dict], out_dir: Path) -> None:
    _write_csv(out_dir / "termination_routes.csv", route_rows)
    _write_csv(out_dir / "termination_route_audit.csv", summary_rows)


def verify_route_sum_identity(summary_rows: list[dict]) -> list[str]:
    """Returns a list of problem descriptions; empty list means all cells pass."""
    problems = []
    for row in summary_rows:
        route_sum = sum(row[f"n_{r}"] for r in ALL_ROUTES)
        if route_sum != row["total_n"]:
            problems.append(
                f"{row['model']}/{row['dataset']}/{row['condition']}: route sum {route_sum} != total_n {row['total_n']}"
            )
    return problems
