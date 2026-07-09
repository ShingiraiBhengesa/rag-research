"""
jsonl_logging.py — Append-only JSONL logging with a fixed schema for all attack runs.

Schema design
-------------
One row per (dataset, model, example, condition). Every row contains every field
in SCHEMA_FIELDS, with None for fields not yet computable at the calling script's
layer. This schema-completeness invariant means:

  - Logs from different experiments (Stage 0 smoke, Stage 1 full, different models)
    can be concatenated and read by a single analysis script without a join step.
  - Adding a new metric field requires only a schema update here and a new key in
    the relevant attack script's log_row() call; existing rows keep their None value.

Condition values used by the two kept attacks
---------------------------------------------
  "clean"                   — unmodified observation (eligibility baseline)
  "epistemic_confirmation"  — Attack A, Epistemic Confirmation cue
  "search_match_cue"        — Attack A, Search-Match cue

Key metric fields
-----------------
  wrong_gap_assessment_success  bool  — PCR@1: agent stopped prematurely at step 1
  delayed_pcr_at_2              bool  — PCR@2: premature at step 1 OR step 2
  delayed_pcr_at_3              bool  — PCR@3: premature at step 1, 2, OR 3
  delta_J_reflect               float — J_reflect_attack − J_reflect_clean (positive
                                        = attack pushed toward premature completion)
  strict_success                bool  — RESEARCH_COMPLETE tool call only (excludes
                                        DIRECT_ANSWER; use for conservative reporting)
  inclusive_success             bool  — same as wrong_gap_assessment_success here;
                                        RESEARCH_COMPLETE OR DIRECT_ANSWER

Fields left as None in Stage 1 (require real search backend, Stage 2 work)
---------------------------------------------------------------------------
  gold_doc_recall_drop, supporting_fact_recall_drop, hop_coverage_drop,
  clean_f1, attack_f1, f1_drop, citation_support_*, unsupported_claim_rate_*
"""
import os
import json
import hashlib
import subprocess

SCHEMA_FIELDS = [
    "run_id", "stage", "dataset", "split", "model", "model_revision", "repo_commit",
    "example_id", "question", "gold_answer", "question_type", "gap_type",
    "method", "attack", "perturbation_family", "condition", "eligible", "ineligibility_reason",
    "clean_observation", "attacked_observation", "perturbation", "perturbation_tokens",
    "edited_spans", "explicitness_level", "forbidden_word_flag",
    "gold_missing_hop_terms", "distractor_terms",
    "clean_next_action", "attack_next_action", "strict_success", "inclusive_success",
    "J_clean", "J_attack", "delta_J",
    "continue_score_clean", "complete_score_clean", "continue_score_attack", "complete_score_attack",
    # Attack 1 (Wrong Next Search) fields
    "clean_next_tool", "clean_next_query", "attack_next_tool", "attack_next_query",
    "wrong_hop_query_success", "gold_query_term_recall_clean", "gold_query_term_recall_attack",
    "query_semantic_drift", "query_failure_type",
    "J_query_clean", "J_query_attack", "delta_J_query",
    # Attack 2 (Wrong Gap Assessment) fields
    "clean_reflection", "attack_reflection", "wrong_gap_assessment_success",
    "delayed_pcr_at_2", "delayed_pcr_at_3",
    "delayed_pcr_at_4", "delayed_pcr_at_5", "delayed_pcr_at_6",
    "delayed_pcr_at_7", "delayed_pcr_at_8", "delayed_pcr_at_9", "delayed_pcr_at_10",
    "J_reflect_clean", "J_reflect_attack", "delta_J_reflect",
    "clean_search_count_after_obs", "attack_search_count_after_obs", "search_suppression",
    "tool_trace_clean", "tool_trace_attack", "tool_trajectory_divergence",
    "gold_docs", "clean_retrieved_docs", "attack_retrieved_docs",
    "gold_doc_recall_clean", "gold_doc_recall_attack", "gold_doc_recall_drop",
    "supporting_fact_recall_clean", "supporting_fact_recall_attack", "supporting_fact_recall_drop",
    "hop_coverage_clean", "hop_coverage_attack", "hop_coverage_drop",
    "clean_answer", "attack_answer", "clean_em", "attack_em",
    "clean_f1", "attack_f1", "f1_drop",
    "citation_support_clean", "citation_support_attack",
    "unsupported_claim_rate_clean", "unsupported_claim_rate_attack",
    "candidate_evals", "wallclock_sec", "gpu", "seed", "prompt_hash",
    "raw_generation_clean", "raw_generation_attack", "parse_error", "notes",
]


def get_repo_commit(repo_dir=None):
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def prompt_hash(prompt_str):
    if prompt_str is None:
        return None
    return hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()[:16]


def make_row(**fields):
    """Build one schema-complete dict; unknown kwargs raise (catches typos/renames),
    missing fields default to None."""
    unknown = set(fields) - set(SCHEMA_FIELDS)
    if unknown:
        raise ValueError(f"unknown jsonl_logging fields: {sorted(unknown)}")
    return {k: fields.get(k) for k in SCHEMA_FIELDS}


def write_row(path, **fields):
    row = make_row(**fields)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
