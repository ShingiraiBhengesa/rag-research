# EVIDENCE_BOUNDARY_AND_HOP_PCR_RESULTS.md

Integrates three pieces that were previously either scattered across two packages or missing entirely: supervisor recovery (one summary instead of two disconnected fields), the two evidence boundaries (side by side instead of two separate files), and hop-stratified PCR (new -- PCR had never been cut by hop/evidence-count stratum before this). Locked PCR, evidence matching rules, and comparison manifest are unchanged from `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/README.md` and `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` -- this report only combines already-validated numbers, it does not recompute them.

## 1. Hop-stratified PCR

See `hop_stratified_pcr_stepwise.csv` / `hop_stratified_pcr_paired_deltas.csv` -- PCR curves and paired deltas, grouped by `(model, dataset, hop_stratum, condition)` instead of just `(model, dataset, condition)`. Reuses `canonical_completion.compute_route_specific_curves`/`.build_strict_pairs` unchanged; no new PCR logic. Stratum values per dataset: HotpotQA={2} (fixed, not a real stratification axis), 2WikiMultiHopQA={2,4}, MuSiQue={2,3,4}, MultiHopRAG={2,3,4} (required-evidence-count, not ordered hops).

## 2. Evidence boundaries, side by side

See `evidence_boundary_paired_results.csv` -- one row per `(model, dataset, comparison_name)`, with both `final_report_input_*` columns (from `results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/evidence_summary.csv`) and `local_completion_*` columns (from `paired_evidence_deltas.csv`) present together. **These two boundaries are not expected to agree** (see `README.md`'s reconciliation section) -- this file exists so a reader can see both numbers in one place without cross-referencing two files by hand.

## 3. Supervisor recovery, unified

See `supervisor_recovery_summary.csv` -- one row per `(model, dataset, condition)`, built from `process_metrics.csv::later_evidence_recovery`'s categorical label (`src/prematurestop/analysis/fixed_cue_matrix/paper_evaluation/recovery.py`'s `RECOVERY_*` constants). Percentages are restricted to `n_redelegated` rows (redelegation alone is never counted as recovery), matching `figures.py::plot_supervisor_recovery`'s existing convention.

Pooled across all 96 (model,dataset,condition) cells: 18307 total redelegated rows, 4874 fully restored, 829 partially restored, 12602 did not restore.

## 4. Validation

**4/4 new gates passed** (appended to `validation_gates.csv` as gates 15-18, alongside the original 14):

| Gate | Description | Status |
|---|---|---|
| 15 | Every hop-stratified PCR curve is monotonic nondecreasing | PASS |
| 16 | Hop-stratified paired N never exceeds the unstratified paired N for the same cell | PASS |
| 17 | Recovery category counts sum to n_total, n_redelegated = n_total - n_no_redelegation - n_unrecognized_label | PASS |
| 18 | Every (model,dataset,comparison) cell has both evidence boundaries present | PASS |

## 5. What this is NOT

This is a consolidation of already-validated numbers, not a new causal analysis and not Chapter 5. It does not resolve the final-report-input vs local-completion boundary difference (both are reported, on purpose, side by side) and it does not make an answer-harm claim (see `EVIDENCE_VS_ANSWER_HARM_NOTES.md` for that, still preliminary). `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` sections 7.4-7.6 remain the actual Chapter-5-facing narrative, still `_TODO(human)`.
