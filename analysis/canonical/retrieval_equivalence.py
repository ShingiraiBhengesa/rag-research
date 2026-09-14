"""Phase 9D: retrieval-equivalence audit across paired conditions.

Uses EXISTING, already-validated traces only (never reruns retrieval or
model inference). For every strictly-paired example (model + dataset +
example_id, both sides present with a readable trace), compares the FIRST
benchmark_search event recorded in episode 0 across the condition pair, on:

  * first search query (raw `queries[0]`) and its normalized form
  * retrieved document-ID set and rank order (from `results[*].document_id`,
    ordered by `rank`)
  * `base_tool_message_hash` (hash of the retrieval output BEFORE any cue
    suffix is appended -- identical values mean the underlying retrieved
    content, not just the query, was identical)
  * `shared_prefix_hash` (the hash `trace_schema.validate_paired_traces`
    itself uses to detect prefix divergence)
  * `index_hash` / `retriever_config_hash` (from the first result's fields --
    identical for every call against the same corpus file with the same BM25
    parameters, so these should always match; a mismatch would indicate a
    corpus/config drift between conditions, not a retrieval-content drift)

Per attack/runtime/trace_schema.py's own invariant (already enforced at
write time by validate_paired_traces when a paired_replay_anchor is present)
these hashes are the authoritative signal for "did this pair actually share
a prefix." This module measures how often that holds in production data
where -- per chapter3_source_of_truth.md section 7 -- no paired_replay_anchor
is actually used, so equivalence is not structurally guaranteed and must be
measured empirically.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis import config
from analysis.completion_inventory import RowRecord, build_inventory

from .canonical_completion import PAIR_SPECS


@dataclass
class FirstSearchFingerprint:
    found: bool
    query: Optional[str] = None
    normalized_query: Optional[str] = None
    doc_ids_ranked: tuple[str, ...] = ()
    base_tool_message_hash: Optional[str] = None
    tool_message_hash: Optional[str] = None
    shared_prefix_hash: Optional[str] = None
    index_hash: Optional[str] = None
    retriever_config_hash: Optional[str] = None
    parse_error: Optional[str] = None


def _first_search_event(trace: dict) -> Optional[dict]:
    for entry in trace.get("events", []):
        if entry.get("event_type") == "benchmark_search":
            return entry
    return None


def extract_first_search_fingerprint(trace_path: Path) -> FirstSearchFingerprint:
    try:
        with open(trace_path, "r", encoding="utf-8") as fh:
            trace = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return FirstSearchFingerprint(found=False, parse_error=str(exc))
    ev = _first_search_event(trace)
    if ev is None:
        return FirstSearchFingerprint(found=False)
    results = ev.get("results") or []
    ranked = tuple(r.get("document_id") for r in sorted(results, key=lambda r: r.get("rank", 0)))
    index_hash = results[0].get("index_hash") if results else None
    retriever_config_hash = results[0].get("retriever_config_hash") if results else None
    queries = ev.get("queries") or []
    normalized = ev.get("normalized_queries") or []
    return FirstSearchFingerprint(
        found=True,
        query=queries[0] if queries else None,
        normalized_query=normalized[0] if normalized else None,
        doc_ids_ranked=ranked,
        base_tool_message_hash=ev.get("base_tool_message_hash"),
        tool_message_hash=ev.get("tool_message_hash"),
        shared_prefix_hash=ev.get("shared_prefix_hash"),
        index_hash=index_hash,
        retriever_config_hash=retriever_config_hash,
    )


def _process_row(row: RowRecord) -> tuple[str, FirstSearchFingerprint]:
    if row.final_status != "ok" or not row.trace_exists:
        return row.trace_path, FirstSearchFingerprint(found=False, parse_error="execution_failure_or_missing_trace")
    path = row.trace_path if Path(row.trace_path).is_absolute() else config.REPO_ROOT / row.trace_path
    return row.trace_path, extract_first_search_fingerprint(path)


def build_pair_comparison_rows(rows: list[RowRecord], max_workers: int = 10) -> list[dict]:
    by_key: dict[tuple[str, str, str, str], RowRecord] = {
        (r.model, r.dataset, r.condition, r.example_id): r for r in rows
    }
    # Only fetch fingerprints for rows that are actually needed by some pair.
    needed_paths: dict[str, RowRecord] = {}
    pair_membership: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for cond, baseline in PAIR_SPECS:
        pair_membership[(cond, baseline)]
    by_model_dataset_example: dict[tuple[str, str, str], dict[str, RowRecord]] = defaultdict(dict)
    for r in rows:
        by_model_dataset_example[(r.model, r.dataset, r.example_id)][r.condition] = r

    tasks: list[RowRecord] = []
    seen_paths = set()
    for (model, dataset, example_id), by_cond in by_model_dataset_example.items():
        for cond, baseline in PAIR_SPECS:
            if cond in by_cond and baseline in by_cond:
                for c in (cond, baseline):
                    r = by_cond[c]
                    if r.trace_path not in seen_paths:
                        seen_paths.add(r.trace_path)
                        tasks.append(r)

    fingerprints: dict[str, FirstSearchFingerprint] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_row, r) for r in tasks]
        for fut in as_completed(futures):
            path, fp = fut.result()
            fingerprints[path] = fp

    rows_out: list[dict] = []
    for (model, dataset, example_id), by_cond in by_model_dataset_example.items():
        for cond, baseline in PAIR_SPECS:
            if cond not in by_cond or baseline not in by_cond:
                continue
            r_a = by_cond[cond]
            r_b = by_cond[baseline]
            fp_a = fingerprints.get(r_a.trace_path)
            fp_b = fingerprints.get(r_b.trace_path)
            if fp_a is None or fp_b is None or not fp_a.found or not fp_b.found:
                rows_out.append(
                    {
                        "model": model, "dataset": dataset, "example_id": example_id,
                        "condition": cond, "baseline": baseline,
                        "both_found": False,
                        "identical_first_query": None, "identical_normalized_query": None,
                        "identical_doc_id_set": None, "identical_rank_order": None,
                        "identical_base_tool_message_hash": None, "identical_shared_prefix_hash": None,
                        "identical_index_hash": None, "identical_retriever_config_hash": None,
                    }
                )
                continue
            doc_set_a = set(fp_a.doc_ids_ranked)
            doc_set_b = set(fp_b.doc_ids_ranked)
            rows_out.append(
                {
                    "model": model, "dataset": dataset, "example_id": example_id,
                    "condition": cond, "baseline": baseline,
                    "both_found": True,
                    "identical_first_query": fp_a.query == fp_b.query,
                    "identical_normalized_query": fp_a.normalized_query == fp_b.normalized_query,
                    "identical_doc_id_set": doc_set_a == doc_set_b,
                    "identical_rank_order": fp_a.doc_ids_ranked == fp_b.doc_ids_ranked,
                    "identical_base_tool_message_hash": (
                        fp_a.base_tool_message_hash == fp_b.base_tool_message_hash
                        and fp_a.base_tool_message_hash is not None
                    ),
                    "identical_shared_prefix_hash": (
                        fp_a.shared_prefix_hash == fp_b.shared_prefix_hash and fp_a.shared_prefix_hash is not None
                    ),
                    "identical_index_hash": fp_a.index_hash == fp_b.index_hash,
                    "identical_retriever_config_hash": fp_a.retriever_config_hash == fp_b.retriever_config_hash,
                    "query_a": fp_a.query, "query_b": fp_b.query,
                }
            )
    return rows_out


def write_outputs(rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "retrieval_equivalence_by_pair.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
    else:
        fieldnames: list[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    both_found = [r for r in rows if r["both_found"]]
    n = len(both_found)
    n_total = len(rows)

    def pct(key):
        if not n:
            return None
        return 100.0 * sum(1 for r in both_found if r[key]) / n

    summary = {
        "n_pair_rows_attempted": n_total,
        "n_pairs_with_both_traces_readable": n,
        "pct_identical_first_query": pct("identical_first_query"),
        "pct_identical_normalized_query": pct("identical_normalized_query"),
        "pct_identical_doc_id_set": pct("identical_doc_id_set"),
        "pct_identical_rank_order": pct("identical_rank_order"),
        "pct_identical_base_tool_message_hash": pct("identical_base_tool_message_hash"),
        "pct_identical_shared_prefix_hash": pct("identical_shared_prefix_hash"),
        "pct_identical_index_hash": pct("identical_index_hash"),
        "pct_identical_retriever_config_hash": pct("identical_retriever_config_hash"),
    }
    n_diff_query_same_docs = sum(
        1 for r in both_found if not r["identical_first_query"] and r["identical_doc_id_set"]
    )
    n_diff_docs = sum(1 for r in both_found if not r["identical_doc_id_set"])
    summary["n_different_query_but_identical_doc_set"] = n_diff_query_same_docs
    summary["pct_different_query_but_identical_doc_set"] = (100.0 * n_diff_query_same_docs / n) if n else None
    summary["n_different_retrieved_evidence"] = n_diff_docs
    summary["pct_different_retrieved_evidence"] = (100.0 * n_diff_docs / n) if n else None

    lines = [
        "# Retrieval-equivalence audit summary",
        "",
        f"Pair-rows attempted (example present under both condition and baseline): {n_total}",
        f"Pairs with both traces readable and a first-search event found: {n}",
        "",
        "| Check | % identical (of readable pairs) |",
        "|---|---|",
    ]
    for label, key in [
        ("First search query (raw)", "pct_identical_first_query"),
        ("First search query (normalized)", "pct_identical_normalized_query"),
        ("Retrieved document-ID set", "pct_identical_doc_id_set"),
        ("Retrieved rank order", "pct_identical_rank_order"),
        ("Original (pre-suffix) observation hash", "pct_identical_base_tool_message_hash"),
        ("Shared-prefix hash", "pct_identical_shared_prefix_hash"),
        ("BM25 index hash", "pct_identical_index_hash"),
        ("Retriever config hash", "pct_identical_retriever_config_hash"),
    ]:
        v = summary[key]
        lines.append(f"| {label} | {v:.1f}% |" if v is not None else f"| {label} | n/a |")
    lines += [
        "",
        f"Different query but identical retrieved document set: {n_diff_query_same_docs} "
        f"({summary['pct_different_query_but_identical_doc_set']:.1f}% of readable pairs)"
        if n else "Different query but identical retrieved document set: n/a",
        f"Different retrieved evidence (document-ID sets differ): {n_diff_docs} "
        f"({summary['pct_different_retrieved_evidence']:.1f}% of readable pairs)"
        if n else "Different retrieved evidence: n/a",
        "",
        "**Interpretation guardrail:** this audit measures what the production traces "
        "actually contain. It does NOT assume identical retrieval across conditions -- "
        "see chapter3_source_of_truth.md section 7 and chapter3_final_source_of_truth.md "
        "section 11 for why no structural guarantee exists in production "
        "(`run_paired_replay` is not the production entry point).",
    ]
    (out_dir / "retrieval_equivalence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    inventory = build_inventory(config.DEFAULT_PRODUCTION_ROOT)
    rows = build_pair_comparison_rows(inventory.rows)
    summary = write_outputs(rows, out_dir)
    print(summary)
