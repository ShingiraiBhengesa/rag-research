"""Repository-relative configuration for the completed Qwen/Llama/Gemma analysis.

All model, dataset, and condition identifiers are imported from the frozen
production registries (`attack.runtime.production_models`,
`attack.runtime.cues`) rather than re-declared here, so this package can
never drift from the actual production configuration. Nothing in this module
imports, modifies, or executes any production *runtime* module (runner.py,
production_runner.py, benchmark_search.py, runtime_gate.py) -- it only reads
their read-only constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from attack.runtime.cues import (
    DEPRECATED_CONDITION_ALIASES,
    NATURAL_CONDITIONS,
    ORACLE_COMPLETE_CONDITIONS,
    V3_CONDITIONS,
)
from attack.runtime.production_models import (
    FINAL_MODEL_IDS,
    FINAL_PRODUCTION_MODELS,
    MODEL_TOOL_CALL_PARSERS,
)

# ---------------------------------------------------------------------------
# Repository-relative paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_PRODUCTION_ROOT
    / "evaluation"
    / "v3_final_completed_qwen_llama_gemma_20260913"
)

# Every row already written to a production shard JSONL before the
# 2026-09-14 top-level reorg stores its `trace_path` field as this literal,
# now-stale, repo-relative string (moving whitebox_paper_results/e2e_full_system_v3/
# production/ to results/ does not retroactively rewrite bytes already
# persisted inside existing JSONL rows -- confirmed identical across all 3
# completed models x 4 datasets). Every consumer that resolves a stored
# `trace_path` against REPO_ROOT must normalize it through this function
# first, or it will silently look in a directory that no longer exists.
_STALE_PRODUCTION_ROOT_PREFIX = "whitebox_paper_results/e2e_full_system_v3/production/"


def normalize_stored_trace_path(trace_path: str) -> str:
    """Maps a possibly-stale stored `trace_path` string to its current
    `results/`-relative form. Idempotent -- a path already using the new
    prefix, or an absolute path, passes through unchanged.
    """
    if trace_path.startswith(_STALE_PRODUCTION_ROOT_PREFIX):
        return "results/" + trace_path[len(_STALE_PRODUCTION_ROOT_PREFIX):]
    return trace_path

# ---------------------------------------------------------------------------
# Model-family scope: hard include/exclude lists
# ---------------------------------------------------------------------------

# Discovered from attack/runtime/production_models.py -- never hand-typed
# guesses. FINAL_PRODUCTION_MODELS = ("qwen14b", "gemma27b", "gpt_oss_20b", "llama8b").
assert set(FINAL_PRODUCTION_MODELS) == {"qwen14b", "gemma27b", "gpt_oss_20b", "llama8b"}, (
    "production model registry changed shape; re-verify INCLUDED/EXCLUDED "
    "model lists in analysis/config.py before proceeding."
)

INCLUDED_MODELS: tuple[str, ...] = ("qwen14b", "llama8b", "gemma27b")
EXCLUDED_MODEL_FAMILIES: tuple[str, ...] = ("gpt_oss_20b",)

# Extended 2026-09-13: gemma27b's production matrix completed (432/432
# shards, confirmed via squeue showing zero active/queued gemma27b jobs and
# a full shard-file count) and was moved from EXCLUDED_MODEL_FAMILIES into
# INCLUDED_MODELS, per the plan already recorded in PUBLISHING.md's
# "Deferred Cleanup After All Jobs Finish" section. gpt_oss_20b remains
# excluded -- its production arrays are still blocked (see PUBLISHING.md).

# Substrings that must never appear in any path this package opens, writes,
# or reports on. Deliberately broader than the exact directory names so a
# renamed shard/log path is still caught.
_EXCLUDED_PATH_TOKENS: tuple[str, ...] = (
    "gpt_oss_20b",
    "gpt-oss",
    "gptoss",
)

MODEL_IDS: Mapping[str, str] = {m: FINAL_MODEL_IDS[m] for m in INCLUDED_MODELS}
MODEL_TOOL_PARSERS: Mapping[str, str] = {m: MODEL_TOOL_CALL_PARSERS[m] for m in INCLUDED_MODELS}

DISPLAY_MODEL_NAMES: Mapping[str, str] = {
    "qwen14b": "Qwen2.5-14B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
    "gemma27b": "Gemma-3-27B-IT",
}


class ExcludedModelPathError(ValueError):
    """Raised when analysis code is asked to open a path for an excluded model
    (currently: GPT-OSS-20B only -- see EXCLUDED_MODEL_FAMILIES)."""


def assert_included_model(model: str) -> None:
    if model not in INCLUDED_MODELS:
        raise ExcludedModelPathError(
            f"model {model!r} is not in the completed-analysis scope "
            f"{INCLUDED_MODELS!r}; {EXCLUDED_MODEL_FAMILIES!r} are excluded "
            "at the model-family level for this analysis (still-blocked "
            "production jobs)."
        )


def assert_path_not_excluded(path: str) -> None:
    """Hard input guard: refuse to open any path naming an excluded model family."""
    lowered = str(path).lower()
    for token in _EXCLUDED_PATH_TOKENS:
        if token in lowered:
            raise ExcludedModelPathError(
                f"refusing to open path containing excluded model token {token!r}: {path}"
            )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS: tuple[str, ...] = ("hotpot", "2wiki", "musique", "multihoprag")

DISPLAY_DATASET_NAMES: Mapping[str, str] = {
    "hotpot": "HotpotQA",
    "2wiki": "2WikiMultiHopQA",
    "musique": "MuSiQue",
    "multihoprag": "MultiHopRAG",
}

# ---------------------------------------------------------------------------
# Conditions -- identical to the frozen production matrix
# ---------------------------------------------------------------------------

CONDITIONS: tuple[str, ...] = V3_CONDITIONS
assert len(CONDITIONS) == 12, f"expected 12 conditions, found {len(CONDITIONS)}: {CONDITIONS}"

CONTROL_CONDITIONS: tuple[str, ...] = ("NaturalClean", "NaturalNeutral")
DIRECTIONAL_CONTROL_CONDITIONS: tuple[str, ...] = ("NaturalContinueSearch",)
CUE_ONLY_ATTACK_CONDITIONS: tuple[str, ...] = (
    "Natural_epistemic_confirmation",
    "Natural_search_match_cue",
    "Natural_source_coverage",
)
SOURCE_ATTACHED_ATTACK_CONDITIONS: tuple[str, ...] = (
    "Natural_epistemic_confirmation_source",
    "Natural_search_match_source",
)
PREREGISTERED_SUFFICIENCY_CONDITIONS: tuple[str, ...] = (
    "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)
ORACLE_CONDITIONS: tuple[str, ...] = tuple(ORACLE_COMPLETE_CONDITIONS)

assert set(CONTROL_CONDITIONS)
assert (
    set(CONTROL_CONDITIONS)
    | set(DIRECTIONAL_CONTROL_CONDITIONS)
    | set(CUE_ONLY_ATTACK_CONDITIONS)
    | set(SOURCE_ATTACHED_ATTACK_CONDITIONS)
    | set(PREREGISTERED_SUFFICIENCY_CONDITIONS)
    | set(ORACLE_CONDITIONS)
) == set(CONDITIONS), "condition grouping is missing or duplicating a condition"

CONDITION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "Clean and Neutral": CONTROL_CONDITIONS,
    "ContinueSearch": DIRECTIONAL_CONTROL_CONDITIONS,
    "Natural cue-only attacks": CUE_ONLY_ATTACK_CONDITIONS,
    "Natural source-attached attacks": SOURCE_ATTACHED_ATTACK_CONDITIONS,
    "Preregistered sufficiency attacks": PREREGISTERED_SUFFICIENCY_CONDITIONS,
    "Oracle calibration controls": ORACLE_CONDITIONS,
}

# Human-readable display names for Markdown/paper tables. CSV output must
# always retain the raw identifier from CONDITIONS above.
CONDITION_DISPLAY_NAMES: Mapping[str, str] = {
    "NaturalClean": "Clean",
    "NaturalNeutral": "Neutral",
    "NaturalContinueSearch": "ContinueSearch",
    "Natural_epistemic_confirmation": "Epistemic confirmation",
    "Natural_search_match_cue": "Search-match cue",
    "Natural_source_coverage": "Source coverage",
    "Natural_epistemic_confirmation_source": "Epistemic confirmation + source",
    "Natural_search_match_source": "Search-match + source",
    "Natural_PreRegisteredSufficiency": "PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation": "PreRegisteredSufficiencySaturation",
    "OracleComplete_Neutral": "Oracle Neutral",
    "OracleComplete_PreRegisteredSufficiencySaturation": "Oracle PreRegisteredSufficiencySaturation",
}

MAIN_PAPER_CDF_CONDITIONS: tuple[str, ...] = (
    "NaturalClean",
    "NaturalNeutral",
    "NaturalContinueSearch",
    "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)

# ---------------------------------------------------------------------------
# Terminal route vocabulary (Section 3)
# ---------------------------------------------------------------------------

ROUTE_RESEARCH_COMPLETE = "research_complete"
ROUTE_NO_TOOL_EXIT = "researcher_no_tool_exit"  # raw label DIRECT_ANSWER upstream; never surfaced publicly
ROUTE_OTHER = "other_terminal_route"
ROUTE_ERROR = "technical_error"

PUBLIC_ROUTE_LABELS: Mapping[str, str] = {
    ROUTE_RESEARCH_COMPLETE: "ResearchComplete",
    ROUTE_NO_TOOL_EXIT: "Researcher no-tool exit",
    ROUTE_OTHER: "Other terminal route",
    ROUTE_ERROR: "Technical/error",
}

RAW_LABEL_NO_TOOL_EXIT = "DIRECT_ANSWER"  # preserved only as an internal raw label, never shown publicly

# ---------------------------------------------------------------------------
# Tool names (as they appear in ODR's tool schema; shared between the
# researcher decision parser and episode-level branch detection so both
# agree on what counts as "a search")
# ---------------------------------------------------------------------------

SEARCH_TOOL_NAME = "tavily_search"  # ODR's tool label even when routed to benchmark_local
THINK_TOOL_NAME = "think_tool"
RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

DETERMINISTIC_SEED = 20260910
PCR_CHECKPOINTS: tuple[int, ...] = (1, 3, 5)
BOOTSTRAP_ITERATIONS = 10000

# ---------------------------------------------------------------------------
# Interpretation text required verbatim in the final report (Section 1)
# ---------------------------------------------------------------------------

GEMMA_GPT_EXCLUSION_STATEMENT = (
    "Gemma completed its production matrix (432/432 shards, 2026-09-13) and "
    "is now included above. GPT-OSS-20B's production matrix remains blocked "
    "(dead SLURM dependency, see PUBLISHING.md) and is excluded at the "
    "model-family level. No GPT-OSS-20B result, trace, configuration, job, "
    "or evaluation output was modified."
)
