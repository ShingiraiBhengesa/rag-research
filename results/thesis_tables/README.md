# Thesis tables

The CSVs behind the thesis's results tables, written by
`scripts/thesis_figures/` from the published run-level data. Values are
unrounded. Table numbers follow the thesis draft; check them against the final
PDF.

| Thesis | Result | File |
|---|---|---|
| Table 5.1 | ASR for every cue vs its reference, per model × dataset (sign test, BH) | `e1/asr_table_all.csv`; per-model summary `e1/asr_macro.csv` |
| Table 5.2, Figure 5.2 | Step reduction D_q, Fixed Sufficiency vs Neutral | `e1/timing_reconciled.csv`, `e1/timing_summary.csv`; per-pair values `e1/step_pairs_all.csv`; Wilcoxon check `e2/e1_asr_wilcoxon.csv` |
| Table 5.3 | EICR, Fixed Sufficiency vs Neutral (unpaired and paired, McNemar, BH) | `e1/eicr_reconciled.csv`; reconciliation of EICR definitions `e1/eicr_reconciliation.csv` |
| Table 5.4 | Paired coverage loss at the Researcher's stop | `coverage/e1_coverage_loss.csv` |
| §5 sensitivity | ASR on the identical-d₁ subset | `e1/d1_sensitivity_asr.csv` |
| §5.6 | E2 attrition from 450 questions to valid anchors and cues | `e2/e2_attrition.csv` |
| Table 5.5 | E2 ASR, each condition vs Neutral | `e2/e2_asr_all.csv` |
| Table 5.6, Figure 5.4 | E2 head-to-head contrasts at each cell's k50 | `e2/e2_contrasts_k50.csv` (k50 per cell: `e2/e2_k50.csv`); step-level contrasts `e1/e2_contrasts.csv` |
| §5.6 | E2 EICR (McNemar) and coverage loss | `e2/e2_eicr_mcnemar.csv`, `coverage/e2_coverage_loss.csv` |
| Figure 5.3 | E2 PCR curves (all valid runs; common questions) | `e2/e2_pcr_curves.csv`, `coverage/e2_pcr_curves_common.csv` |

## Regenerate

From the repository root, in this order:

```bash
python scripts/thesis_figures/thesis_restructure_v2.py
python scripts/thesis_figures/thesis_restructure_v2.py --cov
python scripts/thesis_figures/eicr_timing_reconciled.py
python scripts/thesis_figures/fig5_2_step_boxplot.py
python scripts/thesis_figures/thesis_restructure_e2.py
python scripts/thesis_figures/thesis_restructure_e2.py --cov
```

These need only tracked files. One of them is under Git LFS
(`results/whitebox/e1_fixed_cue_matrix/qwen_llama/data/answer_run_level.csv`),
so run `git lfs pull` after cloning.

The scripts print `VERIFY … PASS/FAIL` cross-checks against earlier outputs
kept in `scripts/thesis_figures/reference/`. On a clean clone (2026-11-02),
all 18 checks pass and every CSV here is reproduced byte-identical.

Answer-accuracy and harm outputs (`e2_accuracy`, `e2_mechanism`, `*harm*`) are
also written but are gitignored, because they are outside the thesis scope.
