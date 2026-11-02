# Thesis → repository results map

Maps the results reported in the thesis to the files in this repository that
hold them. Thesis: [Sufficient by Design (SLU, 2026)](https://example.com/thesis-link-placeholder)
*(placeholder link, to be replaced)*.

Table and figure numbers follow the thesis draft; check them against the final
PDF. Paths are relative to the repository root. `E1` is
`results/whitebox/e1_fixed_cue_matrix/`, `E2` is
`results/whitebox/e2_bridge_mechanism/`, and `*` stands for both E1 model
bundles (`qwen_llama`, `gemma_gptoss`).

**Figures:** the six figures printed in the thesis are in
[`results/thesis_figures/`](../results/thesis_figures/), named by thesis
number, with SHA-256 checksums against the thesis PDF:

| Thesis | File |
|---|---|
| Figure 3.1 | `results/thesis_figures/fig3_1_agentic_rag_loop.png` |
| Figure 4.1 | `results/thesis_figures/fig4_1_routes.pdf` |
| Figure 5.1 | `results/thesis_figures/fig5_1_pcr_curves.pdf` |
| Figure 5.2 | `results/thesis_figures/fig5_2_step_boxplot.png` |
| Figure 5.3 | `results/thesis_figures/fig5_3_e2_pcr_cdf.pdf` |
| Figure 5.4 | `results/thesis_figures/fig5_4_e2_forest.png` |

**Tables:** the CSVs behind the thesis tables (ASR, step reduction, EICR,
coverage loss and the E2 tables) are in
[`results/thesis_tables/`](../results/thesis_tables/); its README maps each file
to its table. The tables below also list the underlying run-level data.

## E1 — fixed-cue matrix

| Thesis | Result | Repository file |
|---|---|---|
| Table 3.4 | The twelve conditions and exact cue text (code ID ↔ thesis name) | `E1/canonical/condition_metadata.csv` (`thesis_display_label`; README condition table) |
| §5.1 | Run inventory: 86,400 planned, 84,098 PCR-eligible | `E1/*/data/run_inventory.csv`, `E1/canonical/final_sample_counts.csv` |
| Table 5.1, 5.2 | ASR and step reduction D_q (cue vs reference, per model × dataset) | `results/thesis_tables/e1/asr_table_all.csv`, `e1/timing_reconciled.csv` ← per-run stopping steps in `E1/evidence_hop_coverage*/run_level_evidence_hops.csv` |
| Figure 5.1 | PCR curves by model, mean of 4 datasets | `results/thesis_figures/fig5_1_pcr_curves.pdf` ← `E1/*/data/pcr_stepwise.csv` |
| Figure 4.1 | How Neutral episodes end (routes) | `results/thesis_figures/fig4_1_routes.pdf` ← `E1/*/data/route_decomposition_stepwise.csv` |
| Figure 5.2 | Paired step reduction D_q, Fixed Sufficiency vs Neutral | `results/thesis_figures/fig5_2_step_boxplot.png` ← `E1/*/data/pcr_run_level.csv` |
| Table 5.3 | EICR, Fixed Sufficiency vs Neutral, by model × dataset | `results/thesis_tables/e1/eicr_reconciled.csv` ← `E1/evidence_hop_coverage*/run_level_evidence_hops.csv` |
| Table 5.4 | Paired coverage loss at the local stop | `results/thesis_tables/coverage/e1_coverage_loss.csv` |
| §5 (sensitivity) | Identical-d₁ subset; technical failures by condition | `E1/sensitivity/` |

## E2 — bridge-anchored study (v2b full run)

| Thesis | Result | Repository file |
|---|---|---|
| §5.6 | Bridge and cue coverage per cell (anchors, valid cues) | `E2/v2b_full/diagnostics/bridge_and_cue_coverage.csv`, `E2/v2b_full/bridge_selection/bridge_provenance.csv` |
| §5.6 | Generated-cue audit (gates V1–V5, scores) | `E2/v2b_full/generated_cues/cue_audit.csv` |
| Table 5.5 | Each condition vs Neutral (ASR) | `results/thesis_tables/e2/e2_asr_all.csv` ← `E2/v2b_full/trajectories/run_labels.csv` |
| Table 5.6, Figure 5.4 | Structure, truth, selection and bridge-vs-fixed contrasts (forest plot) | `results/thesis_figures/fig5_4_e2_forest.png` ← `results/thesis_figures/data/fig5_4_e2_contrasts_k50.csv` (from `E2/v2b_full/trajectories/run_labels.csv`) |
| Figure 5.3 | PCR@k, Neutral / Fixed Sufficiency / Bridge K3, common anchor-matched questions | `results/thesis_figures/fig5_3_e2_pcr_cdf.pdf` ← `results/thesis_figures/data/fig5_3_e2_pcr_curves_common.csv` (from `E2/v2b_full/trajectories/run_labels.csv`) |
| Appendix A, Eq. A.1 | Whole-run premature stop (pre-specified primary) | `E2/v2b_full/pcr/primary_by_cell.csv`, `E2/v2b_full/pcr/premature_stop_contrasts.csv`, `E2/v2b_full/pcr/stopped_by_step_2_contrasts.csv` (Llama gate fallback) |
| Appendix A | Baseline gate | `E2/v2b_full/diagnostics/baseline_gate.csv` |

## Not thesis results

* `results/whitebox/_out_of_scope/answer_harm/`: final-answer analysis, out of scope.
* `E2/v2b_full/diagnostics/answer_accuracy_by_arm.csv` and `paired_answer_and_evidence.csv`: answer accuracy, recorded by the pipeline but not reported.
