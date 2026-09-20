# EVIDENCE_HOP_RESULTS.md

## 1. Scope and locked relationship to PCR

This is the first supporting analysis after PCR. PCR (T_i, terminal route, eligibility) is read verbatim from `analysis.canonical.canonical_completion` and is not redefined, recomputed, or reinterpreted here. This report covers evidence/hop coverage only -- no thesis figures, no no-tool response content audit, no final-answer harm claims (a later, separate task).

Commit at generation time: `9c327466ab8e730ba29a71db375ea5aa7556981d` (equals the required frozen commit: True).

## 2. Data and eligibility counts

Total canonical run rows: **43200** (2 models x 4 datasets x 12 conditions).

See `evidence_hop_summary.csv` for the full model x dataset x stratum x condition breakdown (`n_valid`, `n_evidence_eligible`, `n_completed_and_evidence_eligible`, `n_max_budget_exit` columns). Evidence-eligible rows by model/dataset:

| Model | Dataset | n_evidence_eligible |
|---|---|---|
| llama8b | 2wiki | 5313 |
| llama8b | hotpot | 5261 |
| llama8b | multihoprag | 5181 |
| llama8b | musique | 5204 |
| qwen14b | 2wiki | 5397 |
| qwen14b | hotpot | 5396 |
| qwen14b | multihoprag | 5399 |
| qwen14b | musique | 5384 |

## 3. Evidence coverage at local completion

See `run_level_evidence_hops.csv` (`local_coverage_fraction`, `local_full_evidence`, `local_evidence_count_missing`) for every row, and `evidence_hop_summary.csv`'s `mean_local_completion_coverage` / `full_evidence_rate_at_local_completion` / `evidence_incomplete_completion_rate_among_completed_eligible` columns for the aggregate view, grouped by model x dataset x evidence/hop stratum x condition.

**Not the same boundary as `qwen_llama_results_bundle`'s `evidence_summary.csv`** (that one measures final-report-input coverage, downstream of any redelegation/compression -- this measures upstream coverage at the researcher's own completion decision). See `README.md`'s reconciliation section for the full explanation and a per-row cross-check between the two.

## 4. Results by hop/evidence-count stratum

`evidence_hop_summary.csv`'s `evidence_hop_stratum` column carries the per-dataset stratum value (hop_count for HotpotQA/2WikiMultiHopQA/MuSiQue, required_evidence_count for MultiHopRAG -- see README.md). Compact view (mean local-completion coverage, main-paper conditions only, all strata for that dataset):

| Model | Dataset | Stratum | Condition | n_completed_eligible | mean_local_coverage | full_evidence_rate |
|---|---|---|---|---|---|---|
| llama8b | 2wiki |  | NaturalClean | 0 | NA | NA |
| llama8b | 2wiki |  | NaturalContinueSearch | 0 | NA | NA |
| llama8b | 2wiki |  | NaturalNeutral | 0 | NA | NA |
| llama8b | 2wiki |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| llama8b | 2wiki |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| llama8b | 2wiki | 2 | NaturalClean | 329 | 0.734 | 54.4% |
| llama8b | 2wiki | 2 | NaturalContinueSearch | 322 | 0.707 | 49.4% |
| llama8b | 2wiki | 2 | NaturalNeutral | 332 | 0.709 | 49.7% |
| llama8b | 2wiki | 2 | Natural_PreRegisteredSufficiency | 337 | 0.740 | 55.2% |
| llama8b | 2wiki | 2 | Natural_PreRegisteredSufficiencySaturation | 337 | 0.737 | 54.6% |
| llama8b | 2wiki | 4 | NaturalClean | 103 | 0.602 | 15.5% |
| llama8b | 2wiki | 4 | NaturalContinueSearch | 103 | 0.629 | 25.2% |
| llama8b | 2wiki | 4 | NaturalNeutral | 105 | 0.617 | 22.9% |
| llama8b | 2wiki | 4 | Natural_PreRegisteredSufficiency | 107 | 0.568 | 12.1% |
| llama8b | 2wiki | 4 | Natural_PreRegisteredSufficiencySaturation | 107 | 0.565 | 12.1% |
| llama8b | hotpot |  | NaturalClean | 0 | NA | NA |
| llama8b | hotpot |  | NaturalContinueSearch | 0 | NA | NA |
| llama8b | hotpot |  | NaturalNeutral | 0 | NA | NA |
| llama8b | hotpot |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| llama8b | hotpot |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| llama8b | hotpot | 2 | NaturalClean | 427 | 0.741 | 56.7% |
| llama8b | hotpot | 2 | NaturalContinueSearch | 416 | 0.744 | 56.2% |
| llama8b | hotpot | 2 | NaturalNeutral | 427 | 0.737 | 55.3% |
| llama8b | hotpot | 2 | Natural_PreRegisteredSufficiency | 438 | 0.703 | 49.8% |
| llama8b | hotpot | 2 | Natural_PreRegisteredSufficiencySaturation | 438 | 0.697 | 49.1% |
| llama8b | multihoprag |  | NaturalClean | 0 | NA | NA |
| llama8b | multihoprag |  | NaturalContinueSearch | 0 | NA | NA |
| llama8b | multihoprag |  | NaturalNeutral | 0 | NA | NA |
| llama8b | multihoprag |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| llama8b | multihoprag |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| llama8b | multihoprag | 2 | NaturalClean | 198 | 1.000 | 100.0% |
| llama8b | multihoprag | 2 | NaturalContinueSearch | 200 | 1.000 | 100.0% |
| llama8b | multihoprag | 2 | NaturalNeutral | 199 | 1.000 | 100.0% |
| llama8b | multihoprag | 2 | Natural_PreRegisteredSufficiency | 201 | 1.000 | 100.0% |
| llama8b | multihoprag | 2 | Natural_PreRegisteredSufficiencySaturation | 201 | 1.000 | 100.0% |
| llama8b | multihoprag | 3 | NaturalClean | 163 | 1.000 | 100.0% |
| llama8b | multihoprag | 3 | NaturalContinueSearch | 161 | 1.000 | 100.0% |
| llama8b | multihoprag | 3 | NaturalNeutral | 163 | 1.000 | 100.0% |
| llama8b | multihoprag | 3 | Natural_PreRegisteredSufficiency | 165 | 1.000 | 100.0% |
| llama8b | multihoprag | 3 | Natural_PreRegisteredSufficiencySaturation | 165 | 1.000 | 100.0% |
| llama8b | multihoprag | 4 | NaturalClean | 64 | 0.906 | 62.5% |
| llama8b | multihoprag | 4 | NaturalContinueSearch | 62 | 0.931 | 72.6% |
| llama8b | multihoprag | 4 | NaturalNeutral | 66 | 0.917 | 66.7% |
| llama8b | multihoprag | 4 | Natural_PreRegisteredSufficiency | 69 | 0.844 | 37.7% |
| llama8b | multihoprag | 4 | Natural_PreRegisteredSufficiencySaturation | 69 | 0.830 | 31.9% |
| llama8b | musique |  | NaturalClean | 0 | NA | NA |
| llama8b | musique |  | NaturalContinueSearch | 0 | NA | NA |
| llama8b | musique |  | NaturalNeutral | 0 | NA | NA |
| llama8b | musique |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| llama8b | musique |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| llama8b | musique | 2 | NaturalClean | 224 | 0.632 | 39.7% |
| llama8b | musique | 2 | NaturalContinueSearch | 218 | 0.619 | 38.5% |
| llama8b | musique | 2 | NaturalNeutral | 223 | 0.608 | 35.9% |
| llama8b | musique | 2 | Natural_PreRegisteredSufficiency | 226 | 0.551 | 28.8% |
| llama8b | musique | 2 | Natural_PreRegisteredSufficiencySaturation | 226 | 0.542 | 27.4% |
| llama8b | musique | 3 | NaturalClean | 134 | 0.639 | 28.4% |
| llama8b | musique | 3 | NaturalContinueSearch | 134 | 0.654 | 30.6% |
| llama8b | musique | 3 | NaturalNeutral | 133 | 0.669 | 33.1% |
| llama8b | musique | 3 | Natural_PreRegisteredSufficiency | 135 | 0.617 | 24.4% |
| llama8b | musique | 3 | Natural_PreRegisteredSufficiencySaturation | 135 | 0.610 | 23.7% |
| llama8b | musique | 4 | NaturalClean | 67 | 0.556 | 13.4% |
| llama8b | musique | 4 | NaturalContinueSearch | 67 | 0.563 | 14.9% |
| llama8b | musique | 4 | NaturalNeutral | 65 | 0.558 | 15.4% |
| llama8b | musique | 4 | Natural_PreRegisteredSufficiency | 69 | 0.500 | 11.6% |
| llama8b | musique | 4 | Natural_PreRegisteredSufficiencySaturation | 69 | 0.489 | 11.6% |
| qwen14b | 2wiki |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| qwen14b | 2wiki |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| qwen14b | 2wiki | 2 | NaturalClean | 335 | 0.781 | 56.4% |
| qwen14b | 2wiki | 2 | NaturalContinueSearch | 335 | 0.819 | 64.2% |
| qwen14b | 2wiki | 2 | NaturalNeutral | 340 | 0.778 | 56.2% |
| qwen14b | 2wiki | 2 | Natural_PreRegisteredSufficiency | 340 | 0.746 | 50.3% |
| qwen14b | 2wiki | 2 | Natural_PreRegisteredSufficiencySaturation | 340 | 0.747 | 50.6% |
| qwen14b | 2wiki | 4 | NaturalClean | 109 | 0.633 | 8.3% |
| qwen14b | 2wiki | 4 | NaturalContinueSearch | 109 | 0.656 | 11.0% |
| qwen14b | 2wiki | 4 | NaturalNeutral | 108 | 0.641 | 10.2% |
| qwen14b | 2wiki | 4 | Natural_PreRegisteredSufficiency | 109 | 0.631 | 7.3% |
| qwen14b | 2wiki | 4 | Natural_PreRegisteredSufficiencySaturation | 109 | 0.628 | 8.3% |
| qwen14b | hotpot | 2 | NaturalClean | 443 | 0.827 | 67.3% |
| qwen14b | hotpot | 2 | NaturalContinueSearch | 444 | 0.843 | 71.2% |
| qwen14b | hotpot | 2 | NaturalNeutral | 444 | 0.827 | 67.8% |
| qwen14b | hotpot | 2 | Natural_PreRegisteredSufficiency | 446 | 0.795 | 60.1% |
| qwen14b | hotpot | 2 | Natural_PreRegisteredSufficiencySaturation | 447 | 0.800 | 60.9% |
| qwen14b | multihoprag | 2 | NaturalClean | 211 | 1.000 | 100.0% |
| qwen14b | multihoprag | 2 | NaturalContinueSearch | 211 | 1.000 | 100.0% |
| qwen14b | multihoprag | 2 | NaturalNeutral | 211 | 1.000 | 100.0% |
| qwen14b | multihoprag | 2 | Natural_PreRegisteredSufficiency | 211 | 1.000 | 100.0% |
| qwen14b | multihoprag | 2 | Natural_PreRegisteredSufficiencySaturation | 211 | 1.000 | 100.0% |
| qwen14b | multihoprag | 3 | NaturalClean | 169 | 1.000 | 100.0% |
| qwen14b | multihoprag | 3 | NaturalContinueSearch | 170 | 1.000 | 100.0% |
| qwen14b | multihoprag | 3 | NaturalNeutral | 170 | 1.000 | 100.0% |
| qwen14b | multihoprag | 3 | Natural_PreRegisteredSufficiency | 170 | 1.000 | 100.0% |
| qwen14b | multihoprag | 3 | Natural_PreRegisteredSufficiencySaturation | 170 | 1.000 | 100.0% |
| qwen14b | multihoprag | 4 | NaturalClean | 69 | 0.895 | 58.0% |
| qwen14b | multihoprag | 4 | NaturalContinueSearch | 69 | 0.931 | 72.5% |
| qwen14b | multihoprag | 4 | NaturalNeutral | 69 | 0.906 | 62.3% |
| qwen14b | multihoprag | 4 | Natural_PreRegisteredSufficiency | 69 | 0.891 | 56.5% |
| qwen14b | multihoprag | 4 | Natural_PreRegisteredSufficiencySaturation | 69 | 0.870 | 47.8% |
| qwen14b | musique |  | NaturalContinueSearch | 0 | NA | NA |
| qwen14b | musique |  | NaturalNeutral | 0 | NA | NA |
| qwen14b | musique |  | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| qwen14b | musique |  | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| qwen14b | musique | 2 | NaturalClean | 231 | 0.736 | 53.2% |
| qwen14b | musique | 2 | NaturalContinueSearch | 233 | 0.762 | 57.5% |
| qwen14b | musique | 2 | NaturalNeutral | 229 | 0.716 | 49.3% |
| qwen14b | musique | 2 | Natural_PreRegisteredSufficiency | 230 | 0.700 | 45.2% |
| qwen14b | musique | 2 | Natural_PreRegisteredSufficiencySaturation | 231 | 0.703 | 46.3% |
| qwen14b | musique | 3 | NaturalClean | 137 | 0.601 | 16.1% |
| qwen14b | musique | 3 | NaturalContinueSearch | 138 | 0.582 | 13.8% |
| qwen14b | musique | 3 | NaturalNeutral | 138 | 0.609 | 18.1% |
| qwen14b | musique | 3 | Natural_PreRegisteredSufficiency | 139 | 0.556 | 18.7% |
| qwen14b | musique | 3 | Natural_PreRegisteredSufficiencySaturation | 140 | 0.552 | 16.4% |
| qwen14b | musique | 4 | NaturalClean | 73 | 0.473 | 6.8% |
| qwen14b | musique | 4 | NaturalContinueSearch | 72 | 0.455 | 2.8% |
| qwen14b | musique | 4 | NaturalNeutral | 74 | 0.470 | 6.8% |
| qwen14b | musique | 4 | Natural_PreRegisteredSufficiency | 75 | 0.417 | 1.3% |
| qwen14b | musique | 4 | Natural_PreRegisteredSufficiencySaturation | 71 | 0.437 | 1.4% |

## 5. Paired cue-versus-control evidence loss

See `paired_evidence_deltas.csv` for every comparison in `analysis.canonical.results_bundle.comparisons.COMPARISONS` (the single reconciled comparison manifest -- no second list introduced here). `paired_evidence_loss` = control mean coverage minus cue mean coverage; positive means the cue condition recovered LESS required evidence at local completion than its paired control.

| Comparison | Model | Dataset | n_paired | paired_evidence_loss | pct_cue_lower | full_evidence_rate_loss |
|---|---|---|---|---|---|---|
| Natural Neutral vs Natural Clean | llama8b | 2wiki | 427 | 0.014 | 9.1% | 1.6% |
| Epistemic Confirmation vs Natural Neutral | llama8b | 2wiki | 435 | 0.005 | 2.5% | 0.9% |
| Search Match vs Natural Neutral | llama8b | 2wiki | 434 | 0.009 | 2.8% | 1.8% |
| Fixed Sufficiency vs Natural Neutral | llama8b | 2wiki | 433 | -0.013 | 12.9% | -1.4% |
| Fixed Sufficiency Saturation vs Natural Neutral | llama8b | 2wiki | 433 | -0.010 | 13.2% | -0.9% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | llama8b | 2wiki | 427 | 0.016 | 8.9% | 2.1% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | llama8b | 2wiki | 427 | 0.020 | 9.1% | 3.0% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | llama8b | 2wiki | 427 | -0.002 | 14.8% | -0.5% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | llama8b | 2wiki | 427 | 0.001 | 15.2% | 0.0% |
| Continue Search vs Natural Neutral | llama8b | 2wiki | 423 | -0.008 | 1.7% | -1.4% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | llama8b | 2wiki | 436 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | llama8b | 2wiki | 437 | 0.005 | 3.4% | 1.1% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | llama8b | 2wiki | 433 | -0.018 | 11.8% | -2.5% |
| Search Match + Source Coverage vs Search Match | llama8b | 2wiki | 433 | -0.023 | 11.5% | -3.9% |
| Natural Neutral vs Natural Clean | llama8b | hotpot | 419 | 0.004 | 6.0% | 1.2% |
| Epistemic Confirmation vs Natural Neutral | llama8b | hotpot | 425 | 0.018 | 4.5% | 2.8% |
| Search Match vs Natural Neutral | llama8b | hotpot | 425 | 0.020 | 4.7% | 3.5% |
| Fixed Sufficiency vs Natural Neutral | llama8b | hotpot | 420 | 0.037 | 21.7% | 6.0% |
| Fixed Sufficiency Saturation vs Natural Neutral | llama8b | hotpot | 420 | 0.040 | 21.9% | 6.4% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | llama8b | hotpot | 421 | 0.024 | 9.0% | 4.3% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | llama8b | hotpot | 421 | 0.026 | 9.3% | 5.0% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | llama8b | hotpot | 419 | 0.044 | 21.7% | 7.9% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | llama8b | hotpot | 419 | 0.048 | 22.0% | 8.4% |
| Continue Search vs Natural Neutral | llama8b | hotpot | 411 | -0.012 | 2.4% | -1.7% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | llama8b | hotpot | 431 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | llama8b | hotpot | 419 | -0.006 | 11.0% | 0.0% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | llama8b | hotpot | 424 | -0.002 | 17.7% | 0.0% |
| Search Match + Source Coverage vs Search Match | llama8b | hotpot | 426 | 0.005 | 17.8% | 1.2% |
| Natural Neutral vs Natural Clean | llama8b | multihoprag | 421 | -0.001 | 1.2% | -0.5% |
| Epistemic Confirmation vs Natural Neutral | llama8b | multihoprag | 428 | 0.004 | 2.1% | 1.6% |
| Search Match vs Natural Neutral | llama8b | multihoprag | 428 | 0.012 | 4.7% | 4.7% |
| Fixed Sufficiency vs Natural Neutral | llama8b | multihoprag | 418 | 0.011 | 6.0% | 4.5% |
| Fixed Sufficiency Saturation vs Natural Neutral | llama8b | multihoprag | 418 | 0.014 | 6.9% | 5.5% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | llama8b | multihoprag | 422 | 0.003 | 2.8% | 1.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | llama8b | multihoprag | 422 | 0.011 | 5.2% | 4.3% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | llama8b | multihoprag | 417 | 0.010 | 5.8% | 4.1% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | llama8b | multihoprag | 417 | 0.013 | 6.7% | 5.0% |
| Continue Search vs Natural Neutral | llama8b | multihoprag | 422 | -0.003 | 0.2% | -1.2% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | llama8b | multihoprag | 421 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | llama8b | multihoprag | 426 | -0.001 | 1.2% | -0.2% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | llama8b | multihoprag | 418 | 0.004 | 4.1% | 1.7% |
| Search Match + Source Coverage vs Search Match | llama8b | multihoprag | 419 | 0.001 | 2.6% | 0.5% |
| Natural Neutral vs Natural Clean | llama8b | musique | 414 | 0.003 | 8.7% | 0.5% |
| Epistemic Confirmation vs Natural Neutral | llama8b | musique | 417 | 0.010 | 7.0% | 1.0% |
| Search Match vs Natural Neutral | llama8b | musique | 417 | 0.012 | 6.7% | 1.7% |
| Fixed Sufficiency vs Natural Neutral | llama8b | musique | 405 | 0.050 | 27.9% | 6.7% |
| Fixed Sufficiency Saturation vs Natural Neutral | llama8b | musique | 405 | 0.059 | 28.9% | 7.7% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | llama8b | musique | 420 | 0.012 | 10.5% | 1.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | llama8b | musique | 420 | 0.015 | 11.0% | 1.7% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | llama8b | musique | 408 | 0.051 | 26.5% | 7.1% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | llama8b | musique | 408 | 0.060 | 27.7% | 8.1% |
| Continue Search vs Natural Neutral | llama8b | musique | 406 | -0.011 | 3.9% | -2.0% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | llama8b | musique | 409 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | llama8b | musique | 397 | 0.016 | 20.7% | 2.3% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | llama8b | musique | 412 | 0.014 | 24.8% | 1.0% |
| Search Match + Source Coverage vs Search Match | llama8b | musique | 412 | 0.018 | 23.8% | 1.9% |
| Natural Neutral vs Natural Clean | qwen14b | 2wiki | 443 | 0.001 | 4.7% | 0.0% |
| Epistemic Confirmation vs Natural Neutral | qwen14b | 2wiki | 447 | -0.002 | 2.7% | -0.2% |
| Search Match vs Natural Neutral | qwen14b | 2wiki | 445 | 0.002 | 11.7% | 0.9% |
| Fixed Sufficiency vs Natural Neutral | qwen14b | 2wiki | 447 | 0.027 | 14.3% | 5.1% |
| Fixed Sufficiency Saturation vs Natural Neutral | qwen14b | 2wiki | 447 | 0.027 | 14.1% | 4.7% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | qwen14b | 2wiki | 444 | -0.001 | 4.3% | -0.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | qwen14b | 2wiki | 442 | 0.003 | 11.3% | 0.9% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | qwen14b | 2wiki | 443 | 0.027 | 14.9% | 5.0% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | qwen14b | 2wiki | 443 | 0.026 | 14.7% | 4.5% |
| Continue Search vs Natural Neutral | qwen14b | 2wiki | 442 | -0.035 | 1.8% | -6.3% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | qwen14b | 2wiki | 449 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | qwen14b | 2wiki | 444 | -0.008 | 11.0% | -1.6% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | qwen14b | 2wiki | 445 | 0.006 | 12.8% | 1.1% |
| Search Match + Source Coverage vs Search Match | qwen14b | 2wiki | 446 | 0.001 | 3.6% | 0.0% |
| Natural Neutral vs Natural Clean | qwen14b | hotpot | 439 | 0.001 | 4.1% | -0.5% |
| Epistemic Confirmation vs Natural Neutral | qwen14b | hotpot | 442 | 0.021 | 8.8% | 3.4% |
| Search Match vs Natural Neutral | qwen14b | hotpot | 441 | 0.026 | 13.6% | 5.2% |
| Fixed Sufficiency vs Natural Neutral | qwen14b | hotpot | 441 | 0.031 | 16.1% | 7.5% |
| Fixed Sufficiency Saturation vs Natural Neutral | qwen14b | hotpot | 441 | 0.025 | 15.0% | 6.6% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | qwen14b | hotpot | 441 | 0.023 | 11.8% | 2.9% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | qwen14b | hotpot | 441 | 0.028 | 14.5% | 5.0% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | qwen14b | hotpot | 439 | 0.033 | 15.7% | 7.3% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | qwen14b | hotpot | 440 | 0.027 | 14.8% | 6.4% |
| Continue Search vs Natural Neutral | qwen14b | hotpot | 441 | -0.018 | 2.3% | -3.6% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | qwen14b | hotpot | 441 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | qwen14b | hotpot | 438 | 0.015 | 12.6% | 3.4% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | qwen14b | hotpot | 443 | -0.005 | 7.9% | -0.5% |
| Search Match + Source Coverage vs Search Match | qwen14b | hotpot | 444 | 0.000 | 10.4% | 1.1% |
| Natural Neutral vs Natural Clean | qwen14b | multihoprag | 449 | -0.002 | 1.3% | -0.7% |
| Epistemic Confirmation vs Natural Neutral | qwen14b | multihoprag | 450 | 0.001 | 2.0% | 0.2% |
| Search Match vs Natural Neutral | qwen14b | multihoprag | 449 | 0.001 | 2.2% | 0.4% |
| Fixed Sufficiency vs Natural Neutral | qwen14b | multihoprag | 450 | 0.002 | 4.2% | 0.9% |
| Fixed Sufficiency Saturation vs Natural Neutral | qwen14b | multihoprag | 450 | 0.006 | 4.9% | 2.2% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | qwen14b | multihoprag | 449 | -0.001 | 2.2% | -0.4% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | qwen14b | multihoprag | 448 | -0.001 | 2.5% | -0.2% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | qwen14b | multihoprag | 449 | 0.001 | 3.3% | 0.2% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | qwen14b | multihoprag | 449 | 0.004 | 4.0% | 1.6% |
| Continue Search vs Natural Neutral | qwen14b | multihoprag | 450 | -0.004 | 1.3% | -1.6% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | qwen14b | multihoprag | 450 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | qwen14b | multihoprag | 450 | 0.000 | 3.3% | 0.0% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | qwen14b | multihoprag | 450 | -0.003 | 2.2% | -1.3% |
| Search Match + Source Coverage vs Search Match | qwen14b | multihoprag | 449 | -0.003 | 1.8% | -1.1% |
| Natural Neutral vs Natural Clean | qwen14b | musique | 434 | 0.007 | 6.7% | 1.4% |
| Epistemic Confirmation vs Natural Neutral | qwen14b | musique | 438 | 0.040 | 23.7% | 5.3% |
| Search Match vs Natural Neutral | qwen14b | musique | 436 | 0.031 | 22.2% | 4.8% |
| Fixed Sufficiency vs Natural Neutral | qwen14b | musique | 437 | 0.030 | 23.8% | 2.5% |
| Fixed Sufficiency Saturation vs Natural Neutral | qwen14b | musique | 434 | 0.027 | 22.6% | 2.3% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | qwen14b | musique | 437 | 0.046 | 24.3% | 6.6% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | qwen14b | musique | 435 | 0.037 | 22.8% | 6.2% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | qwen14b | musique | 436 | 0.036 | 23.6% | 3.9% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | qwen14b | musique | 435 | 0.034 | 21.6% | 3.9% |
| Continue Search vs Natural Neutral | qwen14b | musique | 434 | -0.015 | 13.1% | -2.5% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | qwen14b | musique | 434 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | qwen14b | musique | 434 | 0.024 | 22.4% | 5.3% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | qwen14b | musique | 434 | -0.020 | 17.3% | -0.7% |
| Search Match + Source Coverage vs Search Match | qwen14b | musique | 439 | -0.013 | 19.8% | -2.1% |

This table may say whether cue conditions are associated with lower evidence coverage at completion. It must NOT be read as an answer-harm claim -- that requires the separate, later paired final-answer analysis.

## 6. Supervisor recovery

See `evidence_hop_summary.csv`'s `mean_recovery_gain` and `incomplete_at_completion_full_at_final_rate` columns, and `run_level_evidence_hops.csv`'s `recovery_gain` / `incomplete_at_completion_full_at_final` columns per row. `recovery_gain` = final coverage minus local-completion coverage; positive means the workflow recovered evidence after local completion (via supervisor redelegation or otherwise).

## 7. Validation results

**14/14 gates passed.** See `validation_gates.csv` for the full detail; failing gates preserve every row (nothing is silently discarded) and record the likely cause in the `detail` column.

| Gate | Description | Status |
|---|---|---|
| 1 | Run IDs / example IDs unique within model, dataset, condition | PASS |
| 2 | Valid/eligible counts match latest PCR results | PASS |
| 3 | Completion step/route match canonical PCR run-level data | PASS |
| 4 | No completion-specific evidence value assigned to MAX_BUDGET_EXIT | PASS |
| 5 | Coverage values in [0,1] | PASS |
| 6 | Missing count in [0, required_evidence_count] | PASS |
| 7 | Full-evidence true exactly when missing count is zero | PASS |
| 8 | Local-completion evidence excludes post-completion retrievals | PASS |
| 9 | Final coverage is not lower than local coverage | PASS |
| 10 | Every matched evidence identifier has traceable provenance | PASS |
| 11 | Every paired comparison contains identical example IDs on both sides | PASS |
| 12 | Hop/evidence-count strata agree with dataset annotations | PASS |
| 13 | Oracle Complete conditions have full anchor coverage | PASS |
| 14 | Manual validation sample covers >=5 runs/dataset, both models, both routes where available | PASS |

## 8. Important limitations

See `README.md`'s "Known limitations" section (sentence-level supporting-fact recall, MuSiQue reasoning-path text, and the compressed-notes/final-report-input boundaries are not part of this table). This report does not make any final-answer-harm claim -- a cue that lowers evidence coverage at completion is reported as a process-level association only.

## 9. The smallest justified next analysis

Join `run_level_evidence_hops.csv` (or its `example_id`/`condition` keys) against `analysis.answer_evaluation`'s per-row answer records to test whether evidence-incomplete completions are ALSO the runs with a control-correct/cue-wrong paired answer flip -- i.e. whether the evidence-loss association found here actually predicts downstream answer harm, rather than assuming it. That is the paired final-answer-harm analysis this task's stop-list explicitly deferred.
