# Advisor Results Deck — 2026-09-29 — Read This First

## What this ZIP contains

A self-contained slide-building package for a research-advisor meeting,
covering the completed cue-injection PCR (Premature Completion Rate) study
across three models, four datasets, and 12 experimental conditions. Every
figure/table is either copied verbatim from an already-validated canonical
source, or derived from that source's own run-level/summary CSVs (never
from raw traces, never re-run).

## Models included

| Model | Datasets covered | Status |
|---|---|---|
| Qwen2.5-14B-Instruct | HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG | Complete (432/432 shards) |
| Llama-3.1-8B-Instruct | HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG | Complete (432/432 shards) |
| GPT-OSS-20B | HotpotQA, MuSiQue, MultiHopRAG only | **2WikiMultiHopQA incomplete (0/108 shards)** — a fix was deployed and a rerun (SLURM job 166537) was queued at export time but had not yet started. Every GPT-OSS table/figure in this deck covers only 3 of 4 datasets; 2WikiMultiHopQA is never imputed or silently dropped — it is simply absent by construction. |

## Model pending / excluded

**Gemma-3-27B-IT is not included anywhere in this deck.** Its production
data is currently invalid (a discovered generation-time bug caused it to
imitate its own conversation history instead of making real tool calls,
corrupting ~96% of its rows) and a corrected full rerun is in progress. It
was excluded per explicit instruction, not merely because it was
inconvenient.

## Authoritative source paths (read-only; nothing in them was modified)

- `qwen_llama_results_bundle/` — canonical Qwen/Llama results, frozen at
  commit `9c327466` (see `RESULTS_CANONICAL_INDEX.md`), with additive,
  uncommitted extensions on top (a 5th PCR figure family, `CUE_EFFECT_ANALYSIS.md`)
  that do not change any underlying number.
- `gemma_gptoss_results_bundle/` — the canonical GPT-OSS equivalent
  (uncommitted, built at HEAD `959b6315`; also currently holds Gemma27B's
  invalid data alongside GPT-OSS's valid data in the same directory —
  **only the gpt_oss_20b rows were ever read for this export**).
- `analysis/evidence_hop_coverage/` — the canonical local-completion
  evidence-coverage package (Qwen/Llama only; no GPT-OSS equivalent exists
  — see `03_evidence/evidence_results_notes.md`).
- `analysis/canonical/canonical_completion.py` — the sole PCR implementation
  both bundles use.
- `RESULTS_CANONICAL_INDEX.md` — the repo's own canonical-source declaration.

## How PCR is defined

See `02_pcr/PCR_DEFINITION.md` for the full definition, denominator,
terminal-route list, and this export's independent validation results
(all passed). One-line version: PCR@k = the fraction of eligible runs whose
researcher had already stopped (via `RESEARCH_COMPLETE` or a no-tool exit)
by decision step k.

## What belongs in the main slides vs. the appendix

- **Main slides**: `01_conditions/`, `02_pcr/{qwen,llama,gpt_oss}/` (4
  figures each) + `02_pcr/mini_tables/`, `03_evidence/*_mini_table.*` +
  `cross_model_evidence_summary.*`, `04_answers/*_mini_table.*` +
  `cross_model_answer_summary.*`.
- **Appendix only**: everything under `05_appendix/` — full per-dataset
  stepwise tables, route decomposition, hop-stratified evidence detail,
  full (non-macro-averaged) answer tables, and this bundle's own
  validation-gate CSVs.
- See `SLIDE_ASSET_INDEX.md` for an exact slide-by-slide file mapping.

## Unresolved limitations (see `CLAIMS_AND_CAVEATS.md` and `MISSING_OR_INCOMPATIBLE_RESULTS.md` for full detail)

1. GPT-OSS/2WikiMultiHopQA is entirely absent (rerun queued, not started).
2. All semantic-accuracy numbers, for all three models, are **pre-adjudication**
   automatic labels — no human review has been completed for anyone.
3. GPT-OSS has no dedicated hop-stratified/bootstrap-CI evidence package;
   its evidence-coverage numbers were derived fresh for this export from
   the same underlying field, with a simpler (still real, still correct)
   computation — see `03_evidence/evidence_results_notes.md`.
4. This deck's "model-level" PCR curves/mini-tables use a macro-average
   across datasets — a simplification introduced for slide readability,
   not a redefinition of the canonical per-dataset numbers (which remain
   in the appendix unaveraged). See `02_pcr/PCR_DEFINITION.md`'s
   aggregation-method section.
