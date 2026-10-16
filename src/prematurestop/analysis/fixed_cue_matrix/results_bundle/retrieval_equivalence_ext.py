"""Step 6: `data/retrieval_equivalence.csv`.

Reuses `prematurestop.analysis.fixed_cue_matrix.audits.retrieval_equivalence`'s low-level fingerprint
extraction (`extract_first_search_fingerprint`, `_process_row`,
`FirstSearchFingerprint`) entirely unchanged. Only the pair-iteration
orchestration is re-implemented here with an explicit `pair_specs`
parameter (the unified `comparisons.COMPARISONS` manifest) instead of
importing the module-level `canonical_completion.PAIR_SPECS` constant --
this covers the two ablation-vs-parent comparisons that constant doesn't
have, without editing `canonical_completion.py` itself.

Adds two things spec 5.6 asks for that the reused module doesn't expose per
row: `paired_readable_trace_count` (previously only a global number in the
markdown summary) and `identical_observation_sensitivity_population` (the
stricter subset where retrieval was byte-identical before any cue suffix --
`identical_base_tool_message_hash == True` -- for a stricter re-analysis of
the answer-harm effect with its own denominator, per spec 5.6's explicit
"own denominator" requirement).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from prematurestop.evaluation import config
from prematurestop.evaluation import completion as cc
from prematurestop.analysis.fixed_cue_matrix.audits.retrieval_equivalence import FirstSearchFingerprint, _process_row
from prematurestop.evaluation.completion_inventory import RowRecord

from .comparisons import COMPARISONS


def build_pair_comparison_rows_for_manifest(rows: list[RowRecord], max_workers: int = 10) -> list[dict]:
    pair_specs = [(c.cue_condition, c.control_condition) for c in COMPARISONS]

    by_model_dataset_example: dict[tuple[str, str, str], dict[str, RowRecord]] = defaultdict(dict)
    for r in rows:
        by_model_dataset_example[(r.model, r.dataset, r.example_id)][r.condition] = r

    tasks: list[RowRecord] = []
    seen_paths = set()
    for (model, dataset, example_id), by_cond in by_model_dataset_example.items():
        for cond, baseline in pair_specs:
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
        for cond, baseline in pair_specs:
            if cond not in by_cond or baseline not in by_cond:
                continue
            r_a = by_cond[cond]
            r_b = by_cond[baseline]
            fp_a = fingerprints.get(r_a.trace_path)
            fp_b = fingerprints.get(r_b.trace_path)
            base_row = {
                "model": model, "dataset": dataset, "example_id": example_id,
                "condition": cond, "baseline": baseline,
            }
            if fp_a is None or fp_b is None or not fp_a.found or not fp_b.found:
                rows_out.append(
                    {
                        **base_row, "both_found": False,
                        "identical_first_query": None, "identical_normalized_query": None,
                        "identical_doc_id_set": None, "identical_rank_order": None,
                        "identical_base_tool_message_hash": None, "identical_shared_prefix_hash": None,
                        "identical_index_hash": None, "identical_retriever_config_hash": None,
                        "identical_observation_sensitivity_population": False,
                    }
                )
                continue
            doc_set_a = set(fp_a.doc_ids_ranked)
            doc_set_b = set(fp_b.doc_ids_ranked)
            identical_base_hash = (
                fp_a.base_tool_message_hash == fp_b.base_tool_message_hash and fp_a.base_tool_message_hash is not None
            )
            rows_out.append(
                {
                    **base_row, "both_found": True,
                    "identical_first_query": fp_a.query == fp_b.query,
                    "identical_normalized_query": fp_a.normalized_query == fp_b.normalized_query,
                    "identical_doc_id_set": doc_set_a == doc_set_b,
                    "identical_rank_order": fp_a.doc_ids_ranked == fp_b.doc_ids_ranked,
                    "identical_base_tool_message_hash": identical_base_hash,
                    "identical_shared_prefix_hash": (
                        fp_a.shared_prefix_hash == fp_b.shared_prefix_hash and fp_a.shared_prefix_hash is not None
                    ),
                    "identical_index_hash": fp_a.index_hash == fp_b.index_hash,
                    "identical_retriever_config_hash": fp_a.retriever_config_hash == fp_b.retriever_config_hash,
                    "query_a": fp_a.query, "query_b": fp_b.query,
                    # The stricter sensitivity-analysis population: retrieval
                    # was byte-identical (base_tool_message_hash) before any
                    # cue suffix, so any downstream answer-harm effect in
                    # this subset cannot be attributed to a retrieval
                    # difference -- only to the cue text itself.
                    "identical_observation_sensitivity_population": identical_base_hash,
                }
            )
    return rows_out


def write_retrieval_equivalence_csv(rows: list[dict], path: Path) -> None:
    # Per-comparison paired-readable-trace-count, added alongside each row's
    # own data (spec 5.6 -- previously only a global count existed).
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for r in rows:
        if r["both_found"]:
            counts[(r["model"], r["dataset"], r["condition"], r["baseline"])] += 1
    for r in rows:
        r["paired_readable_trace_count"] = counts[(r["model"], r["dataset"], r["condition"], r["baseline"])]
    cc._write_csv(path, rows)  # noqa: SLF001
