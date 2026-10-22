# E2 thesis figures (v2b full run)

Print-ready figures for the E2 results chapter, at 300 DPI, as PDF and PNG.
Every plotted value is read from the v2b full-run outputs; none are typed in
by hand. Regenerate with:

    python -m prematurestop.analysis.bridge_mechanism.thesis_figures [--run-dir DIR]

The source run directory (`local_data/e2_bridge_mechanism/run_results/full/`)
is not yet published under `results/`.

| File | Figure | Data source |
|---|---|---|
| `e2_pcr_curves` | PCR@k for Neutral, Fixed Sufficiency, Bridge K3, Entity-only, Unsupported, per model × dataset | `trajectories/run_labels.csv` (t_i), matched questions |
| `e2_eicr` | Evidence-incomplete completion rate: Neutral, Fixed Sufficiency, Bridge K3, with 95% Wilson CIs | `trajectories/run_labels.csv` + `diagnostics/evidence_coverage.csv`, matched questions |
| `e2_contrast_forest` | ΔPCR at the characteristic step (Qwen 1, Llama 3) for the 3 pre-registered and 2 secondary contrasts, HotpotQA and 2Wiki | `pcr/pcr_paired_contrasts.csv`, `all_paired` |
| `figA1_bridge_cue_in_context` | Appendix Figure A.1: chain H, d₁ with the bridge span and the appended Bridge K3 cue, and the two real Qwen-14B paths (Neutral recovers; Bridge K3 stops with 1/2 evidence) for 2Wiki `c076e821…` | `python -m prematurestop.analysis.bridge_mechanism.figure_a1`; texts copied verbatim from the decomposition, trace, cue audit and run labels (listed in the module docstring) |

## Notes for captions

- **Matched questions.** Each figure compares arms on the same questions only.
  The Unsupported cue exists for fewer questions, so the PCR-curve panels have
  n = 54 / 123 / 59 / 103 (Qwen HotpotQA / Qwen 2Wiki / Llama HotpotQA /
  Llama 2Wiki). Those are fewer than the paired contrasts, which match only
  the two arms being compared.
- **EICR (%):**

  | Cell | n | Neutral | Fixed Suff. | Bridge K3 |
  |---|---|---|---|---|
  | Qwen HotpotQA | 124 | 32.3 | 39.5 | 37.9 |
  | Qwen 2Wiki | 177 | 62.7 | 67.8 | 66.1 |
  | Llama HotpotQA | 145 | 34.5 | 40.7 | 39.3 |
  | Llama 2Wiki | 163 | 42.3 | 46.0 | 45.4 |

- **Forest plot.** Filled marker = 95% bootstrap CI excludes 0. These are
  PCR-timing contrasts. They are not the pre-registered primary outcome
  (premature stop without hop-2 evidence), which was not significant in any
  cell. For Llama, the step-3 view matches E1; E2's pre-registered Llama
  outcome after the failed baseline gate is stopped-by-step-2.
