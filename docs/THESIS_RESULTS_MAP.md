# Thesis → repository results map

Maps the results reported in the thesis to the files in this repository that
hold them. Thesis: [Sufficient by Design (SLU, 2026)](https://example.com/thesis-link-placeholder)
*(placeholder link, to be replaced)*.

Table and figure numbers follow the thesis draft; check them against the final
PDF. Paths are relative to the repository root. `E1` is
`results/whitebox/e1_fixed_cue_matrix/`, `E2` is
`results/whitebox/e2_bridge_mechanism/`, and `*` stands for both E1 model
bundles (`qwen_llama`, `gemma_gptoss`).

## E1 — fixed-cue matrix

| Thesis | Result | Repository file |
|---|---|---|
| Table 3.4 | The twelve conditions and exact cue text (code ID ↔ thesis name) | `E1/canonical/condition_metadata.csv` (`thesis_display_label`; README condition table) |
| §5.1 | Run inventory: 86,400 planned, 84,098 PCR-eligible | `E1/*/data/run_inventory.csv`, `E1/canonical/final_sample_counts.csv` |
| Table 5.1, 5.2 | ASR and step reduction D_q (cue vs reference, per model × dataset) | **Summary only.** Derived from per-run stopping steps in `E1/*/data/pcr_run_level.csv` (`first_terminal_post_observation_step`, `pcr_eligible`); no published ASR table |
| Figure 5.1 | PCR curves by model | `E1/thesis_figures/fig5_1_pcr_curves.*` ← `E1/*/data/pcr_stepwise.csv` |
| Figure 5.2 | How Neutral episodes end (routes) | `E1/thesis_figures/fig5_2_routes.*` ← `E1/*/data/route_decomposition_stepwise.csv` |
| Figure 5.3 | Cue-ranking heatmap, ΔPCR at the characteristic step | `E1/thesis_figures/fig5_3_heatmap.*` ← `E1/*/data/pcr_paired_deltas_stepwise.csv` |
| Table 5.3 | EICR by model × dataset | `E1/evidence_hop_coverage/evidence_hop_summary.csv`, `E1/evidence_hop_coverage_gemma_gptoss/evidence_hop_summary.csv` (n-weighted over hop strata) |
| Table 5.4, Figure 5.4 | Evidence coverage at the local stop; paired coverage loss | `E1/thesis_figures/fig5_4_coverage.*`; `E1/evidence_hop_coverage*/paired_evidence_deltas.csv`, `E1/*/data/evidence_summary.csv` |
| §5 (sensitivity) | Identical-d₁ subset; technical failures by condition | `E1/sensitivity/` |
| Trace raster | Per-step actions, Qwen-14B HotpotQA | `E1/thesis_figures/trace_raster.*` ← `E1/thesis_figures/data/trace_raster_actions_qwen14b_hotpot.csv` |

## E2 — bridge-anchored study (v2b full run)

| Thesis | Result | Repository file |
|---|---|---|
| §5.6 | Bridge and cue coverage per cell (anchors, valid cues) | `E2/v2b_full/diagnostics/bridge_and_cue_coverage.csv`, `E2/v2b_full/bridge_selection/bridge_provenance.csv` |
| §5.6 | Generated-cue audit (gates V1–V5, scores) | `E2/v2b_full/generated_cues/cue_audit.csv` |
| Table 5.5 | Each condition vs Neutral (PCR / ASR) | `E2/v2b_full/pcr/pcr_paired_contrasts.csv`; ASR derived from `t_i` in `E2/v2b_full/trajectories/run_labels.csv` |
| Table 5.6, contrast forest | Structure, truth and selection contrasts | `E2/thesis_figures/e2_contrast_forest.*` ← `E2/v2b_full/pcr/pcr_paired_contrasts.csv` (`all_paired`) |
| E2 PCR figure | PCR@k by arm, matched questions | `E2/thesis_figures/e2_pcr_curves.*` ← `E2/v2b_full/trajectories/run_labels.csv` |
| E2 EICR figure | EICR by arm, matched questions | `E2/thesis_figures/e2_eicr.*` ← `run_labels.csv` + `E2/v2b_full/diagnostics/evidence_coverage.csv` |
| Appendix A, Eq. A.1 | Whole-run premature stop (pre-specified primary) | `E2/v2b_full/pcr/primary_by_cell.csv`, `E2/v2b_full/pcr/premature_stop_contrasts.csv`, `E2/v2b_full/pcr/stopped_by_step_2_contrasts.csv` (Llama gate fallback) |
| Appendix A | Baseline gate | `E2/v2b_full/diagnostics/baseline_gate.csv` |
| Figure A.1 | Bridge cue in context (2Wiki `c076e821…`) | `E2/thesis_figures/figA1_bridge_cue_in_context.*` |

## Not thesis results

* `results/whitebox/_out_of_scope/answer_harm/`: final-answer analysis, out of scope.
* `E2/v1_replay_full/`: superseded first E2 run.
* `E2/v2b_full/diagnostics/answer_accuracy_by_arm.csv` and `paired_answer_and_evidence.csv`: answer accuracy, recorded by the pipeline but not reported.
