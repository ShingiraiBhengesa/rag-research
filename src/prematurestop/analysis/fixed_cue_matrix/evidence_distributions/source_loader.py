"""Loads a unified, per-row evidence-coverage-at-completion population for
all three models, applying the GPT-OSS fix documented in SOURCE_AUDIT.md
section 3 (join against pcr_run_level.csv to restrict to genuine PCR
completions; derive retrieved/missing/full-coverage from
required_dependency_count x local_completion_coverage rather than using
evidence_run_level.csv's own missing_hop_count/full_hop_indicator columns,
which are at the wrong -- final-report-input -- boundary).

Read-only against all canonical sources. Writes nothing.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from prematurestop.paths import RESULTS_E1_GEMMA_GPTOSS

from prematurestop.paths import REPO_ROOT  # noqa: E402

QWEN_LLAMA_EVIDENCE_HOPS = REPO_ROOT / "results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/run_level_evidence_hops.csv"
GPTOSS_EVIDENCE_RUN_LEVEL = RESULTS_E1_GEMMA_GPTOSS / "data" / "evidence_run_level.csv"
GPTOSS_PCR_RUN_LEVEL = RESULTS_E1_GEMMA_GPTOSS / "data" / "pcr_run_level.csv"

PCR_NUMERATOR_ROUTES = ("RESEARCH_COMPLETE", "NO_TOOL_DIRECT_EXIT")

MODELS = ("qwen14b", "llama8b", "gpt_oss_20b")
MODEL_DISPLAY = {
    "qwen14b": "Qwen2.5-14B-Instruct",
    "llama8b": "Llama-3.1-8B-Instruct",
    "gpt_oss_20b": "GPT-OSS-20B",
}
DATASETS = ("hotpot", "2wiki", "musique", "multihoprag")
DATASET_DISPLAY = {"hotpot": "HotpotQA", "2wiki": "2WikiMultiHopQA", "musique": "MuSiQue", "multihoprag": "MultiHopRAG"}
# GPT-OSS/2wiki: 0/108 valid shards as of this audit (2026-10-06) -- a rerun
# is in progress but incomplete. Never derived from a macro average; the
# dataset is simply absent from GPT-OSS's population below.
MODEL_DATASETS = {
    "qwen14b": DATASETS,
    "llama8b": DATASETS,
    "gpt_oss_20b": ("hotpot", "musique", "multihoprag"),
}

CONDITIONS_8 = (
    "NaturalClean", "Natural_epistemic_confirmation", "Natural_search_match_cue",
    "Natural_source_coverage", "Natural_epistemic_confirmation_source",
    "Natural_search_match_source", "Natural_PreRegisteredSufficiency",
    "Natural_PreRegisteredSufficiencySaturation",
)
ORACLE_CONDITIONS = ("OracleComplete_Neutral", "OracleComplete_PreRegisteredSufficiencySaturation")
ALL_CONDITIONS = CONDITIONS_8 + ORACLE_CONDITIONS

CONDITION_DISPLAY = {
    "NaturalClean": "Clean",
    "NaturalNeutral": "Natural Neutral",
    "Natural_epistemic_confirmation": "Epistemic Confirmation",
    "Natural_search_match_cue": "Search Match",
    "Natural_source_coverage": "Source Coverage",
    "Natural_epistemic_confirmation_source": "Epistemic Confirmation + Source Coverage",
    "Natural_search_match_source": "Search Match + Source Coverage",
    "Natural_PreRegisteredSufficiency": "Fixed Sufficiency",
    "Natural_PreRegisteredSufficiencySaturation": "Fixed Sufficiency Saturation",
    "OracleComplete_Neutral": "Oracle Complete--Neutral",
    "OracleComplete_PreRegisteredSufficiencySaturation": "Oracle Complete--Saturation",
}

COVERAGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CoverageRow:
    model: str
    dataset: str
    condition: str
    example_id: str
    required: int
    retrieved: int
    missing: int
    coverage: float
    full_coverage: bool
    terminal_route: str


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_qwen_llama_rows(model: str) -> list[CoverageRow]:
    assert model in ("qwen14b", "llama8b")
    rows = _read_csv(QWEN_LLAMA_EVIDENCE_HOPS)
    out = []
    for r in rows:
        if r["model"] != model:
            continue
        if r.get("evidence_eligible") != "True":
            continue
        if r["local_coverage_fraction"] in ("", "None"):
            continue  # not a genuine PCR-qualifying completion (completed_within_budget == False)
        required = int(r["required_evidence_count"])
        retrieved = int(r["local_evidence_count_retrieved"])
        missing = int(r["local_evidence_count_missing"])
        coverage = float(r["local_coverage_fraction"])
        full = r["local_full_evidence"] == "True"
        out.append(CoverageRow(
            model=model, dataset=r["dataset"], condition=r["condition"], example_id=r["example_id"],
            required=required, retrieved=retrieved, missing=missing, coverage=coverage,
            full_coverage=full, terminal_route=r["terminal_route"],
        ))
    return out


def load_gptoss_rows() -> list[CoverageRow]:
    """Applies the SOURCE_AUDIT.md section 3 fix: join against
    pcr_run_level.csv to restrict to genuine PCR completions, and derive
    retrieved/missing/full-coverage from required x coverage rather than
    using evidence_run_level.csv's wrong-boundary missing_hop_count /
    full_hop_indicator columns."""
    pcr_rows = _read_csv(GPTOSS_PCR_RUN_LEVEL)
    qualifying_route = {}
    for r in pcr_rows:
        if r["model"] != "gpt_oss_20b":
            continue
        key = (r["dataset"], r["condition"], r["example_id"])
        qualifying_route[key] = r["terminal_route"]

    ev_rows = _read_csv(GPTOSS_EVIDENCE_RUN_LEVEL)
    out = []
    for r in ev_rows:
        if r["model"] != "gpt_oss_20b":
            continue
        if r["local_completion_coverage"] in ("", "None"):
            continue
        key = (r["dataset"], r["condition"], r["example_id"])
        route = qualifying_route.get(key)
        if route not in PCR_NUMERATOR_ROUTES:
            continue  # excluded per SOURCE_AUDIT.md section 3b
        required = int(r["required_dependency_count"])
        coverage = float(r["local_completion_coverage"])
        retrieved_f = coverage * required
        retrieved = round(retrieved_f)
        if abs(retrieved_f - retrieved) > COVERAGE_TOLERANCE * max(required, 1):
            # Should never happen post-filter (validated in validation.py);
            # skip defensively rather than fabricate an integer.
            continue
        missing = required - retrieved
        out.append(CoverageRow(
            model="gpt_oss_20b", dataset=r["dataset"], condition=r["condition"], example_id=r["example_id"],
            required=required, retrieved=retrieved, missing=missing, coverage=coverage,
            full_coverage=(missing == 0), terminal_route=route,
        ))
    return out


def load_all_rows() -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    rows.extend(load_qwen_llama_rows("qwen14b"))
    rows.extend(load_qwen_llama_rows("llama8b"))
    rows.extend(load_gptoss_rows())
    return rows


def rows_for(rows: list[CoverageRow], model: str, dataset: str | None = None, condition: str | None = None) -> list[CoverageRow]:
    out = [r for r in rows if r.model == model]
    if dataset is not None:
        out = [r for r in out if r.dataset == dataset]
    if condition is not None:
        out = [r for r in out if r.condition == condition]
    return out
