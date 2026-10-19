# EVIDENCE_HOP_RESULTS.md

## 1. Scope and locked relationship to PCR

This is the first supporting analysis after PCR. PCR (T_i, terminal route, eligibility) is read verbatim from `prematurestop.evaluation.completion` and is not redefined, recomputed, or reinterpreted here. This report covers evidence/hop coverage only -- no thesis figures, no no-tool response content audit, no final-answer harm claims (a later, separate task).

Commit at generation time: `bca0b16cf8cf17f4e6ed1197f727351d79a99e79` (equals the required frozen commit: False).

## 2. Data and eligibility counts

Total canonical run rows: **42926** (2 models x 4 datasets x 12 conditions).

See `evidence_hop_summary.csv` for the full model x dataset x stratum x condition breakdown (`n_valid`, `n_evidence_eligible`, `n_completed_and_evidence_eligible`, `n_max_budget_exit` columns). Evidence-eligible rows by model/dataset:

| Model | Dataset | n_evidence_eligible |
|---|---|---|
| gemma27b | 2wiki | 5300 |
| gemma27b | hotpot | 5400 |
| gemma27b | multihoprag | 5400 |
| gemma27b | musique | 5350 |
| gpt_oss_20b | 2wiki | 5350 |
| gpt_oss_20b | hotpot | 5015 |
| gpt_oss_20b | multihoprag | 5017 |
| gpt_oss_20b | musique | 5129 |

## 3. Evidence coverage at local completion

See `run_level_evidence_hops.csv` (`local_coverage_fraction`, `local_full_evidence`, `local_evidence_count_missing`) for every row, and `evidence_hop_summary.csv`'s `mean_local_completion_coverage` / `full_evidence_rate_at_local_completion` / `evidence_incomplete_completion_rate_among_completed_eligible` columns for the aggregate view, grouped by model x dataset x evidence/hop stratum x condition.

**Not the same boundary as `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s `evidence_summary.csv`** (that one measures final-report-input coverage, downstream of any redelegation/compression -- this measures upstream coverage at the researcher's own completion decision). See `README.md`'s reconciliation section for the full explanation and a per-row cross-check between the two.

## 4. Results by hop/evidence-count stratum

`evidence_hop_summary.csv`'s `evidence_hop_stratum` column carries the per-dataset stratum value (hop_count for HotpotQA/2WikiMultiHopQA/MuSiQue, required_evidence_count for MultiHopRAG -- see README.md). Compact view (mean local-completion coverage, main-paper conditions only, all strata for that dataset):

| Model | Dataset | Stratum | Condition | n_completed_eligible | mean_local_coverage | full_evidence_rate |
|---|---|---|---|---|---|---|
| gemma27b | 2wiki | 2 | NaturalClean | 341 | 0.868 | 76.2% |
| gemma27b | 2wiki | 2 | NaturalContinueSearch | 341 | 0.861 | 75.4% |
| gemma27b | 2wiki | 2 | NaturalNeutral | 341 | 0.865 | 75.7% |
| gemma27b | 2wiki | 2 | Natural_PreRegisteredSufficiency | 341 | 0.870 | 75.1% |
| gemma27b | 2wiki | 2 | Natural_PreRegisteredSufficiencySaturation | 270 | 0.867 | 75.9% |
| gemma27b | 2wiki | 4 | NaturalClean | 109 | 0.807 | 37.6% |
| gemma27b | 2wiki | 4 | NaturalContinueSearch | 109 | 0.780 | 34.9% |
| gemma27b | 2wiki | 4 | NaturalNeutral | 109 | 0.814 | 39.4% |
| gemma27b | 2wiki | 4 | Natural_PreRegisteredSufficiency | 109 | 0.745 | 22.9% |
| gemma27b | 2wiki | 4 | Natural_PreRegisteredSufficiencySaturation | 80 | 0.781 | 37.5% |
| gemma27b | hotpot | 2 | NaturalClean | 450 | 0.837 | 69.1% |
| gemma27b | hotpot | 2 | NaturalContinueSearch | 450 | 0.833 | 69.3% |
| gemma27b | hotpot | 2 | NaturalNeutral | 450 | 0.821 | 66.7% |
| gemma27b | hotpot | 2 | Natural_PreRegisteredSufficiency | 450 | 0.790 | 62.4% |
| gemma27b | hotpot | 2 | Natural_PreRegisteredSufficiencySaturation | 450 | 0.813 | 65.1% |
| gemma27b | multihoprag | 2 | NaturalClean | 211 | 1.000 | 100.0% |
| gemma27b | multihoprag | 2 | NaturalContinueSearch | 211 | 1.000 | 100.0% |
| gemma27b | multihoprag | 2 | NaturalNeutral | 211 | 1.000 | 100.0% |
| gemma27b | multihoprag | 2 | Natural_PreRegisteredSufficiency | 211 | 1.000 | 100.0% |
| gemma27b | multihoprag | 2 | Natural_PreRegisteredSufficiencySaturation | 211 | 1.000 | 100.0% |
| gemma27b | multihoprag | 3 | NaturalClean | 170 | 1.000 | 100.0% |
| gemma27b | multihoprag | 3 | NaturalContinueSearch | 170 | 1.000 | 100.0% |
| gemma27b | multihoprag | 3 | NaturalNeutral | 170 | 1.000 | 100.0% |
| gemma27b | multihoprag | 3 | Natural_PreRegisteredSufficiency | 170 | 1.000 | 100.0% |
| gemma27b | multihoprag | 3 | Natural_PreRegisteredSufficiencySaturation | 170 | 1.000 | 100.0% |
| gemma27b | multihoprag | 4 | NaturalClean | 69 | 0.848 | 39.1% |
| gemma27b | multihoprag | 4 | NaturalContinueSearch | 69 | 0.862 | 44.9% |
| gemma27b | multihoprag | 4 | NaturalNeutral | 69 | 0.859 | 43.5% |
| gemma27b | multihoprag | 4 | Natural_PreRegisteredSufficiency | 69 | 0.808 | 23.2% |
| gemma27b | multihoprag | 4 | Natural_PreRegisteredSufficiencySaturation | 69 | 0.801 | 20.3% |
| gemma27b | musique | 2 | NaturalClean | 233 | 0.736 | 54.9% |
| gemma27b | musique | 2 | NaturalContinueSearch | 233 | 0.758 | 58.4% |
| gemma27b | musique | 2 | NaturalNeutral | 207 | 0.720 | 51.2% |
| gemma27b | musique | 2 | Natural_PreRegisteredSufficiency | 233 | 0.740 | 55.4% |
| gemma27b | musique | 2 | Natural_PreRegisteredSufficiencySaturation | 233 | 0.723 | 50.2% |
| gemma27b | musique | 3 | NaturalClean | 142 | 0.526 | 14.8% |
| gemma27b | musique | 3 | NaturalContinueSearch | 142 | 0.521 | 14.1% |
| gemma27b | musique | 3 | NaturalNeutral | 126 | 0.524 | 11.9% |
| gemma27b | musique | 3 | Natural_PreRegisteredSufficiency | 142 | 0.521 | 14.1% |
| gemma27b | musique | 3 | Natural_PreRegisteredSufficiencySaturation | 142 | 0.523 | 12.0% |
| gemma27b | musique | 4 | NaturalClean | 74 | 0.446 | 4.1% |
| gemma27b | musique | 4 | NaturalContinueSearch | 75 | 0.450 | 5.3% |
| gemma27b | musique | 4 | NaturalNeutral | 67 | 0.485 | 7.5% |
| gemma27b | musique | 4 | Natural_PreRegisteredSufficiency | 75 | 0.443 | 1.3% |
| gemma27b | musique | 4 | Natural_PreRegisteredSufficiencySaturation | 75 | 0.460 | 5.3% |
| gpt_oss_20b | 2wiki | 2 | NaturalClean | 338 | 0.834 | 67.8% |
| gpt_oss_20b | 2wiki | 2 | NaturalContinueSearch | 338 | 0.840 | 69.5% |
| gpt_oss_20b | 2wiki | 2 | NaturalNeutral | 340 | 0.841 | 69.4% |
| gpt_oss_20b | 2wiki | 2 | Natural_PreRegisteredSufficiency | 339 | 0.823 | 64.9% |
| gpt_oss_20b | 2wiki | 2 | Natural_PreRegisteredSufficiencySaturation | 340 | 0.812 | 62.9% |
| gpt_oss_20b | 2wiki | 4 | NaturalClean | 109 | 0.617 | 14.7% |
| gpt_oss_20b | 2wiki | 4 | NaturalContinueSearch | 109 | 0.617 | 11.0% |
| gpt_oss_20b | 2wiki | 4 | NaturalNeutral | 109 | 0.633 | 14.7% |
| gpt_oss_20b | 2wiki | 4 | Natural_PreRegisteredSufficiency | 109 | 0.619 | 13.8% |
| gpt_oss_20b | 2wiki | 4 | Natural_PreRegisteredSufficiencySaturation | 109 | 0.601 | 7.3% |
| gpt_oss_20b | hotpot | None | NaturalClean | 0 | NA | NA |
| gpt_oss_20b | hotpot | None | NaturalContinueSearch | 0 | NA | NA |
| gpt_oss_20b | hotpot | None | NaturalNeutral | 0 | NA | NA |
| gpt_oss_20b | hotpot | None | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| gpt_oss_20b | hotpot | None | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| gpt_oss_20b | hotpot | 2 | NaturalClean | 429 | 0.851 | 71.6% |
| gpt_oss_20b | hotpot | 2 | NaturalContinueSearch | 427 | 0.851 | 71.2% |
| gpt_oss_20b | hotpot | 2 | NaturalNeutral | 389 | 0.843 | 70.7% |
| gpt_oss_20b | hotpot | 2 | Natural_PreRegisteredSufficiency | 417 | 0.839 | 70.0% |
| gpt_oss_20b | hotpot | 2 | Natural_PreRegisteredSufficiencySaturation | 416 | 0.835 | 68.3% |
| gpt_oss_20b | multihoprag | None | NaturalClean | 0 | NA | NA |
| gpt_oss_20b | multihoprag | None | NaturalContinueSearch | 0 | NA | NA |
| gpt_oss_20b | multihoprag | None | NaturalNeutral | 0 | NA | NA |
| gpt_oss_20b | multihoprag | None | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| gpt_oss_20b | multihoprag | None | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| gpt_oss_20b | multihoprag | 2 | NaturalClean | 193 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 2 | NaturalContinueSearch | 186 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 2 | NaturalNeutral | 192 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 2 | Natural_PreRegisteredSufficiency | 201 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 2 | Natural_PreRegisteredSufficiencySaturation | 197 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 3 | NaturalClean | 149 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 3 | NaturalContinueSearch | 161 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 3 | NaturalNeutral | 158 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 3 | Natural_PreRegisteredSufficiency | 158 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 3 | Natural_PreRegisteredSufficiencySaturation | 161 | 1.000 | 100.0% |
| gpt_oss_20b | multihoprag | 4 | NaturalClean | 68 | 0.864 | 45.6% |
| gpt_oss_20b | multihoprag | 4 | NaturalContinueSearch | 69 | 0.870 | 47.8% |
| gpt_oss_20b | multihoprag | 4 | NaturalNeutral | 68 | 0.875 | 50.0% |
| gpt_oss_20b | multihoprag | 4 | Natural_PreRegisteredSufficiency | 65 | 0.854 | 41.5% |
| gpt_oss_20b | multihoprag | 4 | Natural_PreRegisteredSufficiencySaturation | 64 | 0.855 | 42.2% |
| gpt_oss_20b | musique | None | NaturalClean | 0 | NA | NA |
| gpt_oss_20b | musique | None | NaturalContinueSearch | 0 | NA | NA |
| gpt_oss_20b | musique | None | NaturalNeutral | 0 | NA | NA |
| gpt_oss_20b | musique | None | Natural_PreRegisteredSufficiency | 0 | NA | NA |
| gpt_oss_20b | musique | None | Natural_PreRegisteredSufficiencySaturation | 0 | NA | NA |
| gpt_oss_20b | musique | 2 | NaturalClean | 220 | 0.768 | 57.7% |
| gpt_oss_20b | musique | 2 | NaturalContinueSearch | 214 | 0.769 | 57.5% |
| gpt_oss_20b | musique | 2 | NaturalNeutral | 217 | 0.749 | 53.0% |
| gpt_oss_20b | musique | 2 | Natural_PreRegisteredSufficiency | 224 | 0.717 | 47.8% |
| gpt_oss_20b | musique | 2 | Natural_PreRegisteredSufficiencySaturation | 221 | 0.724 | 51.6% |
| gpt_oss_20b | musique | 3 | NaturalClean | 133 | 0.581 | 19.5% |
| gpt_oss_20b | musique | 3 | NaturalContinueSearch | 133 | 0.629 | 23.3% |
| gpt_oss_20b | musique | 3 | NaturalNeutral | 134 | 0.567 | 19.4% |
| gpt_oss_20b | musique | 3 | Natural_PreRegisteredSufficiency | 133 | 0.574 | 19.5% |
| gpt_oss_20b | musique | 3 | Natural_PreRegisteredSufficiencySaturation | 136 | 0.583 | 22.1% |
| gpt_oss_20b | musique | 4 | NaturalClean | 69 | 0.453 | 4.3% |
| gpt_oss_20b | musique | 4 | NaturalContinueSearch | 73 | 0.476 | 8.2% |
| gpt_oss_20b | musique | 4 | NaturalNeutral | 72 | 0.458 | 6.9% |
| gpt_oss_20b | musique | 4 | Natural_PreRegisteredSufficiency | 73 | 0.414 | 4.1% |
| gpt_oss_20b | musique | 4 | Natural_PreRegisteredSufficiencySaturation | 74 | 0.470 | 5.4% |

## 5. Paired cue-versus-control evidence loss

See `paired_evidence_deltas.csv` for every comparison in `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS` (the single reconciled comparison manifest -- no second list introduced here). `paired_evidence_loss` = control mean coverage minus cue mean coverage; positive means the cue condition recovered LESS required evidence at local completion than its paired control.

| Comparison | Model | Dataset | n_paired | paired_evidence_loss | pct_cue_lower | full_evidence_rate_loss |
|---|---|---|---|---|---|---|
| Natural Neutral vs Natural Clean | gemma27b | 2wiki | 450 | 0.001 | 8.4% | 0.0% |
| Epistemic Confirmation vs Natural Neutral | gemma27b | 2wiki | 450 | 0.031 | 14.7% | 4.7% |
| Search Match vs Natural Neutral | gemma27b | 2wiki | 450 | 0.014 | 12.2% | 1.1% |
| Fixed Sufficiency vs Natural Neutral | gemma27b | 2wiki | 450 | 0.013 | 15.3% | 4.4% |
| Fixed Sufficiency Saturation vs Natural Neutral | gemma27b | 2wiki | 350 | 0.009 | 12.3% | 1.7% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gemma27b | 2wiki | 450 | 0.032 | 14.0% | 4.7% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gemma27b | 2wiki | 450 | 0.015 | 11.8% | 1.1% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gemma27b | 2wiki | 450 | 0.014 | 14.0% | 4.4% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gemma27b | 2wiki | 350 | 0.004 | 10.6% | -0.3% |
| Continue Search vs Natural Neutral | gemma27b | 2wiki | 450 | 0.012 | 10.9% | 1.3% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gemma27b | 2wiki | 450 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gemma27b | 2wiki | 450 | 0.003 | 10.7% | 0.7% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gemma27b | 2wiki | 449 | -0.021 | 7.8% | -3.3% |
| Search Match + Source Coverage vs Search Match | gemma27b | 2wiki | 450 | -0.017 | 8.0% | -1.3% |
| Natural Neutral vs Natural Clean | gemma27b | hotpot | 450 | 0.016 | 10.0% | 2.4% |
| Epistemic Confirmation vs Natural Neutral | gemma27b | hotpot | 450 | 0.007 | 9.6% | 0.2% |
| Search Match vs Natural Neutral | gemma27b | hotpot | 450 | 0.007 | 10.7% | 0.7% |
| Fixed Sufficiency vs Natural Neutral | gemma27b | hotpot | 450 | 0.031 | 13.1% | 4.2% |
| Fixed Sufficiency Saturation vs Natural Neutral | gemma27b | hotpot | 450 | 0.008 | 10.7% | 1.6% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gemma27b | hotpot | 450 | 0.022 | 11.8% | 2.7% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gemma27b | hotpot | 450 | 0.022 | 12.4% | 3.1% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gemma27b | hotpot | 450 | 0.047 | 14.9% | 6.7% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gemma27b | hotpot | 450 | 0.023 | 13.3% | 4.0% |
| Continue Search vs Natural Neutral | gemma27b | hotpot | 450 | -0.012 | 7.8% | -2.7% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gemma27b | hotpot | 450 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gemma27b | hotpot | 450 | 0.013 | 9.8% | 2.4% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gemma27b | hotpot | 450 | 0.010 | 9.8% | 3.1% |
| Search Match + Source Coverage vs Search Match | gemma27b | hotpot | 450 | -0.018 | 6.4% | -2.9% |
| Natural Neutral vs Natural Clean | gemma27b | multihoprag | 450 | -0.002 | 2.0% | -0.7% |
| Epistemic Confirmation vs Natural Neutral | gemma27b | multihoprag | 450 | 0.000 | 2.4% | 0.0% |
| Search Match vs Natural Neutral | gemma27b | multihoprag | 450 | 0.003 | 3.1% | 1.1% |
| Fixed Sufficiency vs Natural Neutral | gemma27b | multihoprag | 450 | 0.008 | 5.1% | 3.1% |
| Fixed Sufficiency Saturation vs Natural Neutral | gemma27b | multihoprag | 450 | 0.009 | 4.9% | 3.6% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gemma27b | multihoprag | 450 | -0.002 | 2.9% | -0.7% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gemma27b | multihoprag | 450 | 0.001 | 2.2% | 0.4% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gemma27b | multihoprag | 450 | 0.006 | 4.2% | 2.4% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gemma27b | multihoprag | 450 | 0.007 | 4.2% | 2.9% |
| Continue Search vs Natural Neutral | gemma27b | multihoprag | 450 | -0.001 | 2.7% | -0.2% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gemma27b | multihoprag | 450 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gemma27b | multihoprag | 450 | 0.003 | 3.8% | 1.1% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gemma27b | multihoprag | 450 | 0.001 | 2.9% | 0.4% |
| Search Match + Source Coverage vs Search Match | gemma27b | multihoprag | 450 | -0.003 | 3.1% | -1.1% |
| Natural Neutral vs Natural Clean | gemma27b | musique | 399 | 0.002 | 13.3% | 2.0% |
| Epistemic Confirmation vs Natural Neutral | gemma27b | musique | 400 | -0.006 | 16.8% | -2.0% |
| Search Match vs Natural Neutral | gemma27b | musique | 400 | -0.013 | 16.5% | -4.0% |
| Fixed Sufficiency vs Natural Neutral | gemma27b | musique | 400 | 0.004 | 17.2% | -0.8% |
| Fixed Sufficiency Saturation vs Natural Neutral | gemma27b | musique | 400 | -0.001 | 17.5% | 0.5% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gemma27b | musique | 449 | -0.007 | 14.7% | 0.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gemma27b | musique | 449 | -0.010 | 15.6% | -1.6% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gemma27b | musique | 449 | -0.001 | 17.1% | 0.4% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gemma27b | musique | 449 | 0.004 | 17.4% | 3.1% |
| Continue Search vs Natural Neutral | gemma27b | musique | 400 | -0.015 | 15.0% | -4.5% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gemma27b | musique | 450 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gemma27b | musique | 400 | -0.009 | 16.2% | -1.8% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gemma27b | musique | 450 | -0.006 | 12.9% | 0.9% |
| Search Match + Source Coverage vs Search Match | gemma27b | musique | 450 | 0.001 | 12.9% | 1.6% |
| Natural Neutral vs Natural Clean | gpt_oss_20b | 2wiki | 446 | -0.010 | 9.4% | -1.3% |
| Epistemic Confirmation vs Natural Neutral | gpt_oss_20b | 2wiki | 449 | 0.005 | 13.6% | 1.3% |
| Search Match vs Natural Neutral | gpt_oss_20b | 2wiki | 448 | 0.013 | 14.3% | 3.3% |
| Fixed Sufficiency vs Natural Neutral | gpt_oss_20b | 2wiki | 447 | 0.017 | 15.9% | 3.6% |
| Fixed Sufficiency Saturation vs Natural Neutral | gpt_oss_20b | 2wiki | 448 | 0.030 | 15.2% | 6.7% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gpt_oss_20b | 2wiki | 447 | -0.006 | 12.8% | -0.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gpt_oss_20b | 2wiki | 446 | 0.004 | 12.6% | 2.0% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gpt_oss_20b | 2wiki | 445 | 0.008 | 14.6% | 2.5% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gpt_oss_20b | 2wiki | 446 | 0.021 | 14.6% | 5.4% |
| Continue Search vs Natural Neutral | gpt_oss_20b | 2wiki | 446 | 0.005 | 13.0% | 0.9% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gpt_oss_20b | 2wiki | 449 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gpt_oss_20b | 2wiki | 396 | 0.004 | 12.6% | 1.8% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gpt_oss_20b | 2wiki | 446 | 0.001 | 13.5% | 0.9% |
| Search Match + Source Coverage vs Search Match | gpt_oss_20b | 2wiki | 446 | -0.013 | 11.0% | -2.7% |
| Natural Neutral vs Natural Clean | gpt_oss_20b | hotpot | 375 | 0.001 | 10.4% | -0.3% |
| Epistemic Confirmation vs Natural Neutral | gpt_oss_20b | hotpot | 367 | -0.030 | 8.7% | -4.6% |
| Search Match vs Natural Neutral | gpt_oss_20b | hotpot | 350 | -0.006 | 11.1% | -0.6% |
| Fixed Sufficiency vs Natural Neutral | gpt_oss_20b | hotpot | 363 | 0.006 | 13.5% | 0.6% |
| Fixed Sufficiency Saturation vs Natural Neutral | gpt_oss_20b | hotpot | 359 | 0.011 | 15.9% | 3.3% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gpt_oss_20b | hotpot | 405 | -0.017 | 11.4% | -3.0% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gpt_oss_20b | hotpot | 384 | 0.003 | 13.3% | 0.3% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gpt_oss_20b | hotpot | 403 | 0.011 | 14.9% | 1.2% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gpt_oss_20b | hotpot | 398 | 0.008 | 12.6% | 1.8% |
| Continue Search vs Natural Neutral | gpt_oss_20b | hotpot | 370 | -0.005 | 10.8% | -0.5% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gpt_oss_20b | hotpot | 405 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gpt_oss_20b | hotpot | 366 | 0.007 | 12.8% | 0.5% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gpt_oss_20b | hotpot | 402 | 0.020 | 13.4% | 2.2% |
| Search Match + Source Coverage vs Search Match | gpt_oss_20b | hotpot | 375 | 0.009 | 12.8% | 1.9% |
| Natural Neutral vs Natural Clean | gpt_oss_20b | multihoprag | 380 | -0.002 | 2.1% | -0.8% |
| Epistemic Confirmation vs Natural Neutral | gpt_oss_20b | multihoprag | 383 | 0.001 | 3.1% | 0.5% |
| Search Match vs Natural Neutral | gpt_oss_20b | multihoprag | 392 | 0.003 | 3.6% | 1.3% |
| Fixed Sufficiency vs Natural Neutral | gpt_oss_20b | multihoprag | 394 | 0.005 | 4.1% | 2.0% |
| Fixed Sufficiency Saturation vs Natural Neutral | gpt_oss_20b | multihoprag | 392 | 0.002 | 3.6% | 0.8% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gpt_oss_20b | multihoprag | 376 | -0.001 | 3.5% | -0.3% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gpt_oss_20b | multihoprag | 382 | 0.001 | 3.7% | 0.3% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gpt_oss_20b | multihoprag | 387 | 0.003 | 3.1% | 1.0% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gpt_oss_20b | multihoprag | 387 | 0.001 | 3.4% | 0.5% |
| Continue Search vs Natural Neutral | gpt_oss_20b | multihoprag | 389 | 0.001 | 3.3% | 0.3% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gpt_oss_20b | multihoprag | 403 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gpt_oss_20b | multihoprag | 389 | 0.001 | 3.6% | 0.5% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gpt_oss_20b | multihoprag | 381 | 0.001 | 3.1% | 0.5% |
| Search Match + Source Coverage vs Search Match | gpt_oss_20b | multihoprag | 388 | 0.000 | 2.1% | 0.0% |
| Natural Neutral vs Natural Clean | gpt_oss_20b | musique | 402 | 0.015 | 19.2% | 2.2% |
| Epistemic Confirmation vs Natural Neutral | gpt_oss_20b | musique | 403 | -0.010 | 17.6% | -0.7% |
| Search Match vs Natural Neutral | gpt_oss_20b | musique | 400 | -0.014 | 17.8% | -0.8% |
| Fixed Sufficiency vs Natural Neutral | gpt_oss_20b | musique | 408 | 0.023 | 24.5% | 2.9% |
| Fixed Sufficiency Saturation vs Natural Neutral | gpt_oss_20b | musique | 406 | 0.006 | 20.7% | 0.0% |
| Natural_epistemic_confirmation vs Natural Clean (total end-to-end change) | gpt_oss_20b | musique | 403 | 0.008 | 20.6% | 2.2% |
| Natural_search_match_cue vs Natural Clean (total end-to-end change) | gpt_oss_20b | musique | 398 | 0.010 | 22.1% | 2.5% |
| Natural_PreRegisteredSufficiency vs Natural Clean (total end-to-end change) | gpt_oss_20b | musique | 404 | 0.033 | 22.8% | 5.2% |
| Natural_PreRegisteredSufficiencySaturation vs Natural Clean (total end-to-end change) | gpt_oss_20b | musique | 407 | 0.019 | 22.6% | 2.2% |
| Continue Search vs Natural Neutral | gpt_oss_20b | musique | 400 | -0.037 | 16.2% | -4.2% |
| Oracle Complete-Saturation vs Oracle Complete-Neutral | gpt_oss_20b | musique | 407 | 0.000 | 0.0% | 0.0% |
| Source Coverage vs Natural Neutral | gpt_oss_20b | musique | 398 | -0.019 | 16.6% | -2.3% |
| Epistemic Confirmation + Source Coverage vs Epistemic Confirmation | gpt_oss_20b | musique | 404 | -0.014 | 18.1% | -2.5% |
| Search Match + Source Coverage vs Search Match | gpt_oss_20b | musique | 408 | -0.007 | 21.3% | -2.7% |

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

Join `run_level_evidence_hops.csv` (or its `example_id`/`condition` keys) against `prematurestop.evaluation.answer_evaluation`'s per-row answer records to test whether evidence-incomplete completions are ALSO the runs with a control-correct/cue-wrong paired answer flip -- i.e. whether the evidence-loss association found here actually predicts downstream answer harm, rather than assuming it. That is the paired final-answer-harm analysis this task's stop-list explicitly deferred.
