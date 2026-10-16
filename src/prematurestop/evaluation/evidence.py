"""Section 11: evidence-loss analysis across four measurement boundaries.

Boundary definitions and how each is actually computed here (all grounded
in fields the production runner itself computes from retrieval metadata,
never from suffix text or answer text -- see src/prematurestop/data/evidence.py's
own docstring, which states the same principle):

1. Shared injection/branch point -- `ParsedTrace.coverage_fraction_at_injection`
   / `.dependencies_seen_at_injection` / `.missing_dependency_ids_at_injection`,
   computed by the runner from the single injection-tagged search event (or
   the first search event for NaturalClean, which has no injection). Valid
   for every trace where a first search happened at all.
2. Attacked researcher's raw terminal evidence -- NOT the runner's whole-run
   `coverage_fraction_at_termination` (verified empirically to be a
   whole-run cumulative measure spanning any redelegation, see
   trace_parser.py docstring). Computed here as the UNION of retrieved
   paragraph IDs across every search event executed in episode 0
   (`_cumulative_union_coverage`), never by trusting a single stored
   "seen so far" snapshot field -- verified necessary: sampling multi-search
   llama8b episodes found the stored `dependencies_seen_after_observation`
   field can lose a previously-found dependency between searches within the
   same episode (most likely from Llama's message-history truncation
   corrupting a history-rescan-based bookkeeping field). Using the last
   event's own field, as an earlier version of this module did, silently
   UNDERSTATES episode-0 coverage whenever this happens -- which in turn
   means it OVERSTATES how much of the injection->final-input gap is
   attributable to supervisor recovery, since the apparent starting point
   was artificially low. Recomputing via raw-ID union fixes this without
   assuming anything about the field's reliability.
3. Compressed notes returned to the supervisor -- `compress_research`'s
   output is free-text prose, but the system prompts it to cite sources
   as `benchmark://<dataset>/<example_id>/<paragraph_id>` URLs (verified
   against real traces). This module extracts those URLs with a strict
   regex and intersects the trailing `<paragraph_id>` against
   `required_paragraph_ids` -- exact-id matching against a controlled
   vocabulary, not fuzzy substring/answer-text matching. When zero
   `benchmark://` citations are found in the compressed text, coverage is
   reported as NA (not zero), per the task's explicit requirement not to
   fabricate a coverage number from unstructured text.
4. Evidence actually passed to `final_report_generation` -- verified
   directly (not assumed) by reading that node's own `on_chain_start` input
   in real traces: it receives `raw_notes` (the concatenated raw retrieved
   passage text, `benchmark://` URLs intact, from every delegation across
   the whole run) plus `notes` (compressed summaries), both as plain state
   fields, for both models. This boundary is therefore computed the same
   way as boundary 3 -- citation-ID extraction from that actual text, not
   from telemetry -- so real compression loss (evidence retrieved
   somewhere in the run but not textually present/traceable in what
   final_report_generation receives) cannot be hidden by an accumulator
   that only proves something was retrieved, not that the final generator
   saw it. The runner's own whole-run `coverage_fraction_at_termination` is
   still recorded, but relabeled `whole_run_cumulative_retrieval_coverage`
   (telemetry / upper bound), never called "final-report input coverage".
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prematurestop.evaluation import config
from prematurestop.evaluation.completion_inventory import RowRecord
from prematurestop.evaluation.trace_parser import load_trace_json, parse_trace

BENCHMARK_URL_RE = re.compile(r"benchmark://[^\s\)\]]+")


@dataclass
class EvidenceRecord:
    model: str
    dataset: str
    condition: str
    example_id: str

    required_dependency_count: int

    # Boundary 1: at injection
    injection_coverage_fraction: Optional[float]
    injection_missing_count: Optional[int]

    # Boundary 2: attacked researcher's raw terminal evidence (episode 0 only)
    episode0_terminal_coverage_fraction: Optional[float]
    episode0_terminal_missing_count: Optional[int]
    episode0_had_any_search: bool

    # Boundary 3: compressed notes (NA unless benchmark:// citations found)
    compressed_notes_available: bool
    compressed_notes_coverage_fraction: Optional[float]  # None == NA
    compressed_notes_cited_paragraph_count: Optional[int]

    # Boundary 4: TRUE final-report input -- citation-based, from the actual
    # raw_notes/notes text passed into final_report_generation's own input.
    final_input_available: bool
    final_input_coverage_fraction: Optional[float]  # None == NA (no citations found)
    final_input_missing_count: Optional[int]
    final_input_cited_paragraph_count: Optional[int]

    # Telemetry only -- the runner's own whole-run retrieval accumulator.
    # NOT the same thing as "what final_report_generation received" (see
    # module docstring); kept for audit/comparison against boundary 4 above,
    # since a gap between the two is itself a compression-loss signal.
    whole_run_cumulative_retrieval_coverage_fraction: Optional[float]
    whole_run_cumulative_retrieval_missing_count: Optional[int]

    full_hop_coverage_at_injection: Optional[bool]
    full_hop_coverage_at_episode0_terminal: Optional[bool]
    full_hop_coverage_at_final_input: Optional[bool]
    full_hop_whole_run_cumulative_retrieval: Optional[bool]


def _extract_compress_research_text(events: list[dict]) -> Optional[str]:
    for entry in events:
        if entry.get("event_type") != "odr_event":
            continue
        ev = entry.get("event") or {}
        if ev.get("name") != "compress_research" or ev.get("event") != "on_chain_end":
            continue
        output = ev.get("data", {}).get("output")
        if isinstance(output, dict):
            text = output.get("compressed_research")
            if isinstance(text, str):
                return text
        elif isinstance(output, str):
            # repr'd Command(...) string -- extract compressed_research value if present
            from prematurestop.evaluation.repr_parsing import LiteralScanError, extract_last_kwarg

            try:
                extracted = extract_last_kwarg(output, "compressed_research")
            except LiteralScanError:
                return None
            if extracted is not None and isinstance(extracted.value, str):
                return extracted.value
        return None  # first compress_research end found but no usable text
    return None


def _extract_final_report_generation_input_text(events: list[dict]) -> Optional[str]:
    """The actual text passed into `final_report_generation` (`raw_notes` +
    `notes` from that node's own `on_chain_start` input) -- verified present
    with intact `benchmark://` URLs for both qwen14b and llama8b traces
    before this function was written, not assumed.
    """
    for entry in events:
        if entry.get("event_type") != "odr_event":
            continue
        ev = entry.get("event") or {}
        if ev.get("name") != "final_report_generation" or ev.get("event") != "on_chain_start":
            continue
        node_input = ev.get("data", {}).get("input")
        if not isinstance(node_input, dict):
            return None
        parts: list[str] = []
        for key in ("raw_notes", "notes"):
            value = node_input.get(key)
            if isinstance(value, list):
                parts.extend(str(v) for v in value)
            elif isinstance(value, str):
                parts.append(value)
        return "\n".join(parts) if parts else None
    return None


def _cumulative_union_coverage(episode) -> tuple[Optional[float], Optional[int]]:
    """Episode-terminal coverage, computed as the union of retrieved paragraph
    IDs across EVERY search event in the episode -- never trusting a single
    stored "seen so far" snapshot field.

    Verified necessary, not just cautious: sampling 8 multi-search llama8b
    episodes found 3 where `dependencies_seen_after_observation` (computed
    upstream, likely by rescanning visible conversation history) actually
    LOSES a previously-found dependency between searches within the same
    episode -- one case resets to empty, one swaps to an unrelated
    dependency instead of accumulating, one drops a found dependency on the
    3rd search. This tracks with Llama's known message-history truncation
    behavior (see trace_parser.py module docstring). Qwen showed no
    violations in the same check, but the raw retrieved-ID union below is
    used for both models so neither depends on that field's reliability.
    """
    search_events = list(episode.search_events)
    if not search_events:
        return None, None
    required_dependency_ids = search_events[0].required_dependency_ids
    required_paragraph_ids = search_events[0].required_paragraph_ids
    if not required_dependency_ids or len(required_dependency_ids) != len(required_paragraph_ids):
        return None, None
    retrieved_union: set[str] = set()
    for event in search_events:
        retrieved_union.update(event.retrieved_paragraph_ids)
    satisfied = sum(1 for pid in required_paragraph_ids if pid in retrieved_union)
    coverage = satisfied / len(required_dependency_ids)
    missing = len(required_dependency_ids) - satisfied
    return coverage, missing


def _paragraph_index(paragraph_id: str) -> str:
    """Normalize a paragraph identifier to its bare per-example index.

    Full compound IDs (hotpot/2wiki/multihoprag) are
    "dataset:example_id:INDEX:slug:hash" -- verified directly (see
    module usage) that the index is always the 3rd colon-separated segment.
    MuSiQue's `required_paragraph_ids` are, uniquely, already bare indices
    in the raw trace data itself (e.g. `('18', '13')`) -- confirmed against
    a real example where those exact numbers ARE the 3rd segment of that
    same example's own retrieved-passage citation URLs
    (`musique:2hop__.../18/zhejiang/...`, `.../13/hokkien/...`). Comparing
    required vs. cited paragraph IDs without this normalization silently
    matches almost nothing for MuSiQue specifically (bare `'18'` never
    equals the full compound string), which is exactly the failure found
    when this was checked against real data -- not a hypothetical edge
    case. Normalizing both sides to the index makes them comparable across
    all 4 datasets without dataset-specific branching. Safe because
    indices are only ever compared within one example's own required/cited
    sets, never across examples.
    """
    parts = paragraph_id.split(":")
    return parts[2] if len(parts) >= 3 else paragraph_id


def _cited_paragraph_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for url in BENCHMARK_URL_RE.findall(text):
        # benchmark://<dataset>/<example_id>/<paragraph_id>
        parts = url[len("benchmark://") :].split("/", 2)
        if len(parts) == 3:
            ids.add(_paragraph_index(parts[2].rstrip(".,;:)")))
    return ids


def _normalized_required_paragraph_ids(required_paragraph_ids: tuple[str, ...]) -> set[str]:
    return {_paragraph_index(pid) for pid in required_paragraph_ids}


def build_evidence_record(model: str, dataset: str, condition: str, example_id: str, trace_path: str) -> EvidenceRecord:
    config.assert_path_not_excluded(trace_path)
    full_path = trace_path if Path(trace_path).is_absolute() else config.REPO_ROOT / trace_path
    raw = load_trace_json(Path(full_path))
    pt = parse_trace(raw, trace_path=str(full_path))

    required_n = pt.required_dependency_count

    def frac_to_missing_count(missing_ids: tuple, frac: Optional[float]) -> Optional[int]:
        return len(missing_ids) if missing_ids or frac is not None else None

    injection_missing = len(pt.missing_dependency_ids_at_injection) if pt.coverage_fraction_at_injection is not None else None

    ep0 = pt.first_episode
    ep0_had_search = bool(ep0 and ep0.search_events)
    if ep0_had_search:
        ep0_frac, ep0_missing = _cumulative_union_coverage(ep0)
    else:
        # No search executed in episode 0 at all: evidence state is unchanged
        # from whatever existed before any retrieval (0 required deps found).
        ep0_frac = 0.0 if required_n else None
        ep0_missing = required_n if required_n else None

    required_paragraph_index_set = _normalized_required_paragraph_ids(pt.required_paragraph_ids)

    compressed_text = _extract_compress_research_text(raw.get("events", []))
    compressed_available = bool(compressed_text)
    compressed_frac = None
    compressed_cited_count = None
    if compressed_text:
        cited_ids = _cited_paragraph_ids(compressed_text)  # already normalized to indices
        compressed_cited_count = len(cited_ids)
        if cited_ids and required_paragraph_index_set:
            covered = cited_ids.intersection(required_paragraph_index_set)
            compressed_frac = len(covered) / len(required_paragraph_index_set)
        # else leave as NA -- no benchmark:// citations to match against, or
        # dataset has no required_paragraph_ids to compare (should not happen
        # for a valid completed row).

    # Boundary 4 (TRUE): citation-based, from what final_report_generation
    # actually received -- same method as boundary 3, different source text.
    final_input_text = _extract_final_report_generation_input_text(raw.get("events", []))
    final_input_available = bool(final_input_text)
    final_input_frac = None
    final_input_missing = None
    final_input_cited_count = None
    if final_input_text:
        cited_ids = _cited_paragraph_ids(final_input_text)  # already normalized to indices
        final_input_cited_count = len(cited_ids)
        if cited_ids and required_paragraph_index_set:
            covered = cited_ids.intersection(required_paragraph_index_set)
            final_input_frac = len(covered) / len(required_paragraph_index_set)
            final_input_missing = len(required_paragraph_index_set) - len(covered)
        # else NA -- no citations to match, or dataset has no required_paragraph_ids

    # Telemetry only (see module docstring) -- NOT "what final_report_generation received".
    whole_run_frac = pt.coverage_fraction_at_termination
    whole_run_missing = len(pt.missing_dependency_ids_at_termination) if whole_run_frac is not None else None

    def full_hop(frac: Optional[float]) -> Optional[bool]:
        return None if frac is None else frac >= 1.0

    return EvidenceRecord(
        model=model, dataset=dataset, condition=condition, example_id=example_id,
        required_dependency_count=required_n,
        injection_coverage_fraction=pt.coverage_fraction_at_injection,
        injection_missing_count=injection_missing,
        episode0_terminal_coverage_fraction=ep0_frac,
        episode0_terminal_missing_count=ep0_missing,
        episode0_had_any_search=ep0_had_search,
        compressed_notes_available=compressed_available,
        compressed_notes_coverage_fraction=compressed_frac,
        compressed_notes_cited_paragraph_count=compressed_cited_count,
        final_input_available=final_input_available,
        final_input_coverage_fraction=final_input_frac,
        final_input_missing_count=final_input_missing,
        final_input_cited_paragraph_count=final_input_cited_count,
        whole_run_cumulative_retrieval_coverage_fraction=whole_run_frac,
        whole_run_cumulative_retrieval_missing_count=whole_run_missing,
        full_hop_coverage_at_injection=full_hop(pt.coverage_fraction_at_injection),
        full_hop_coverage_at_episode0_terminal=full_hop(ep0_frac),
        full_hop_coverage_at_final_input=full_hop(final_input_frac),
        full_hop_whole_run_cumulative_retrieval=full_hop(whole_run_frac),
    )


def _build_one_args(args: tuple) -> EvidenceRecord:
    return build_evidence_record(*args)


def build_evidence_records(rows: list[RowRecord], max_workers: int = 10) -> list[EvidenceRecord]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = [(r.model, r.dataset, r.condition, r.example_id, r.trace_path) for r in rows]
    records: list[EvidenceRecord] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_build_one_args, a) for a in args]
        for fut in as_completed(futures):
            records.append(fut.result())
    return records


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


def write_evidence_state_at_injection(records: list[EvidenceRecord], path: Path) -> None:
    rows = [
        {
            "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
            "required_dependency_count": r.required_dependency_count,
            "injection_coverage_fraction": r.injection_coverage_fraction,
            "injection_missing_count": r.injection_missing_count,
            "full_hop_coverage_at_injection": r.full_hop_coverage_at_injection,
        }
        for r in records
    ]
    _write_csv(path, rows)


def write_evidence_coverage_by_boundary(records: list[EvidenceRecord], path: Path) -> None:
    rows = []
    for r in records:
        for boundary, frac, full_hop in (
            ("injection", r.injection_coverage_fraction, r.full_hop_coverage_at_injection),
            ("episode0_raw_terminal", r.episode0_terminal_coverage_fraction, r.full_hop_coverage_at_episode0_terminal),
            ("compressed_notes", r.compressed_notes_coverage_fraction, None),
            ("final_report_input", r.final_input_coverage_fraction, r.full_hop_coverage_at_final_input),
            # Telemetry, not "what final_report_generation received" -- kept
            # alongside for direct comparison; a gap vs final_report_input
            # above is itself a compression-loss signal.
            ("whole_run_cumulative_retrieval_telemetry", r.whole_run_cumulative_retrieval_coverage_fraction, r.full_hop_whole_run_cumulative_retrieval),
        ):
            rows.append(
                {
                    "model": r.model, "dataset": r.dataset, "condition": r.condition, "example_id": r.example_id,
                    "boundary": boundary,
                    "coverage_fraction": frac if frac is not None else "NA",
                    "full_hop_coverage": full_hop if full_hop is not None else "NA",
                }
            )
    _write_csv(path, rows)


def summarize_by_cell(records: list[EvidenceRecord]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[EvidenceRecord]] = {}
    for r in records:
        grouped.setdefault((r.model, r.dataset, r.condition), []).append(r)

    def mean_or_na(values: list[Optional[float]]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = []
    for (model, dataset, condition), recs in grouped.items():
        n = len(recs)
        n_compressed_available = sum(1 for r in recs if r.compressed_notes_available)
        n_compressed_scoreable = sum(1 for r in recs if r.compressed_notes_coverage_fraction is not None)
        n_final_input_available = sum(1 for r in recs if r.final_input_available)
        n_final_input_scoreable = sum(1 for r in recs if r.final_input_coverage_fraction is not None)
        summary.append(
            {
                "model": model, "dataset": dataset, "condition": condition, "n": n,
                "mean_coverage_at_injection": mean_or_na([r.injection_coverage_fraction for r in recs]),
                "mean_coverage_at_episode0_terminal": mean_or_na([r.episode0_terminal_coverage_fraction for r in recs]),
                "n_compressed_notes_available": n_compressed_available,
                "n_compressed_notes_scoreable_ids": n_compressed_scoreable,
                "mean_coverage_compressed_notes": mean_or_na([r.compressed_notes_coverage_fraction for r in recs]),
                "n_final_input_available": n_final_input_available,
                "n_final_input_scoreable_ids": n_final_input_scoreable,
                "mean_coverage_at_final_input": mean_or_na([r.final_input_coverage_fraction for r in recs]),
                "mean_whole_run_cumulative_retrieval_telemetry": mean_or_na(
                    [r.whole_run_cumulative_retrieval_coverage_fraction for r in recs]
                ),
                "pct_full_hop_at_injection": 100.0 * sum(1 for r in recs if r.full_hop_coverage_at_injection) / n if n else None,
                "pct_full_hop_at_episode0_terminal": 100.0 * sum(1 for r in recs if r.full_hop_coverage_at_episode0_terminal) / n if n else None,
                "pct_full_hop_at_final_input": 100.0 * sum(1 for r in recs if r.full_hop_coverage_at_final_input) / n if n else None,
            }
        )
    return summary
