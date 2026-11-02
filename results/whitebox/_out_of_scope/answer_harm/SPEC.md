# Answer-harm analysis v2 — specification

Written 2026-10-19, **before** the analysis code was run. Any deviation is
recorded at the bottom under "Changes after the first run".

## Why a v2

`evidence_hop_coverage*/EVIDENCE_VS_ANSWER_HARM_NOTES.md` (v1) has four
problems:
1. It pools all 14 comparisons (attacks *and* controls) and all models and
   datasets into one table.
2. It stratifies on the **cue run's** evidence state alone. That state is an
   outcome of the cue, and is confounded with question difficulty.
3. It uses the Researcher's local stop, not the evidence the final-report
   writer actually received.
4. It has no grounding stratum, no adjudication sensitivity and no
   multiple-comparison control.

## Data (no traces are re-read)

| Input | Qwen-14B, Llama-8B | Gemma-27B, gpt-oss-20b |
|---|---|---|
| Per-run evidence | `evidence_hop_coverage/run_level_evidence_hops.csv` | `evidence_hop_coverage_gemma_gptoss/run_level_evidence_hops.csv` |
| Per-run answers | `local_data/evaluation/v3_final_completed_qwen_llama_20260910/answers/final_answer_predictions.jsonl` — reproduces the published `qwen_llama/data/answer_summary.csv` correct counts in 96/96 cells | `gemma_gptoss/data/answer_run_level.csv` |

**Answer outcome.** `correct` = automatic semantic label `correct`. Labels
`incorrect` and `needs_adjudication` both count as not correct, as in the
published accuracy tables. Runs absent from the answer file (technical
failures) leave their pair unformed; the count is reported per cell.

## Units and comparisons

- **Pair** = the same question (`example_id`) run under a comparison's control
  and cue conditions, within one model × dataset.
- **Comparisons** come from `results_bundle.comparisons.COMPARISONS`, grouped
  into families:
  - **Primary attack:** Fixed Sufficiency vs Neutral; Fixed Sufficiency +
    Saturation vs Neutral.
  - **Secondary attack:** Epistemic Confirmation, Search Match and Source
    Coverage vs Neutral; the two stacking ablations; the four "vs Clean"
    total-effect comparisons.
  - **Negative / reverse controls:** Neutral vs Clean (any appended text);
    Oracle Saturation vs Oracle Neutral (evidence complete by construction);
    Continue Search vs Neutral (opposite direction).

## Estimands and tests

Per pair, b = control correct and cue wrong; c = control wrong and cue correct.

1. **Net harm** = (b − c) / n_pairs, in percentage points. Positive = the cue
   made answers worse.
   - Per model × dataset × comparison: exact two-sided McNemar test on (b, c),
     and a 95% paired bootstrap CI (2,000 resamples of pairs, seed 20260910).
   - Pooled per model and over all models, **stratified by model × dataset**:
     pooled net harm = Σ(b − c) / Σn, exact McNemar on Σb vs Σc (valid for
     matched pairs across strata), and a bootstrap that resamples pairs within
     each stratum.
   - Benjamini–Hochberg FDR at q = 0.05 within each family over the
     model × dataset cells.
2. **Mechanism: paired evidence transition.** Each pair is classified by
   whether control and cue each had *full* evidence at the **final-report
   input** (`final_full_evidence`):

   | control | cue | class |
   |---|---|---|
   | full | full | both_complete |
   | full | not full | **evidence_lost** |
   | not full | full | evidence_gained |
   | not full | not full | both_incomplete |

   Net harm and McNemar are computed within each class, pooled as in item 1.
   The hypothesis: harm concentrates in `evidence_lost`. The classification
   uses both runs of the same question, which addresses the difficulty
   confound of v1. The same analysis on the Researcher's local stop
   (`local_full_evidence`) is reported as secondary.
3. **Grounding stratum.** Is the evidence already complete in d₁ for the
   control run (`anchor_full_evidence`)? Prediction: no harm when d₁ is already
   complete.
4. **Hop stratum.** Annotated hop/evidence count (`dataset_stratum_value`).

## Sensitivity analyses

- S1: drop pairs in which either side is `needs_adjudication`.
- S2: drop gpt-oss HotpotQA, MuSiQue and MultiHop-RAG. These were served
  without the lenient Harmony parser; gpt-oss 2Wiki was served with it.

## Outputs (this folder)

- `cells.csv`: item 1 per model × dataset × comparison (with BH q-values).
- `pooled.csv`: item 1 pooled per model and overall.
- `mechanism.csv`: item 2, final and local boundaries.
- `strata.csv`: items 3 and 4.
- `sensitivity.csv`: S1 and S2 for the primary family.
- `figures/`: forest plot of net harm for the primary comparisons, and net
  harm by evidence-transition class.
- `ANSWER_HARM_RESULTS.md`: results with every number citing its CSV.

Code: `src/prematurestop/analysis/fixed_cue_matrix/answer_harm.py`. Tests:
`tests/analysis/test_answer_harm.py`.

## Changes after the first run

- **Added (post-hoc):** `evidence_shift.csv`. For each comparison, counts of
  pairs where the cue lost vs gained full evidence, with an exact sign test.
  The first run showed harm concentrated in `evidence_lost`. Whether the cue
  shifts the *balance* of lost vs gained beyond the Neutral-vs-Clean noise
  baseline is what links section 2 to the average effect, so it was added.
  Not pre-specified; reported as post-hoc.
- **Added:** figures and `ANSWER_HARM_RESULTS.md`, as listed under Outputs.
- No pre-specified estimand, stratum or sensitivity analysis was changed or
  dropped.
