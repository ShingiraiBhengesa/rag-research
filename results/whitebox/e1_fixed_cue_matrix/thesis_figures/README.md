# E1 thesis figures

Print-ready figures for the E1 results chapter, at 300 DPI, as PDF and PNG.
Every plotted value is read from the published CSVs; none are typed in by
hand. Regenerate all six with:

    python -m prematurestop.analysis.fixed_cue_matrix.thesis_figures

| File | Figure | Data source |
|---|---|---|
| `fig5_1_pcr_curves` | PCR curves for each model: Neutral, Fixed Sufficiency, Suff. + Saturation, Continue Search; dotted line at the characteristic step | `*/data/pcr_stepwise.csv`, mean of 4 datasets |
| `fig5_2_routes` | How Neutral-condition episodes end, per model | `*/data/route_decomposition_stepwise.csv`, Neutral, pooled over datasets |
| `fig5_3_heatmap` | Cue-ranking heatmap: Δ PCR at each model's characteristic step; ● = CI excludes 0 in ≥ 3/4 datasets | `*/data/pcr_paired_deltas_stepwise.csv` |
| `fig5_4_coverage` | Evidence coverage at the Researcher's stop vs at the final report; arrows = Suff. + Saturation recovery | `evidence_hop_coverage*/evidence_hop_summary.csv`, n-weighted |
| `fig5_6_avg_harm` | Average net answer harm by model, with 95% CIs and pooled values | `answer_harm/pooled.csv` |
| `fig5_7_mechanism` | Net answer harm by paired evidence transition (key mechanism figure) | `answer_harm/mechanism.csv`, final-report input, all models |

## Notes for captions

- **Smoothing (5.1).** Curves are drawn with monotone (PCHIP) interpolation
  through the 10 measured steps. They can't overshoot or dip, so each remains
  a valid cumulative curve.
- **Routes (5.2).** These are measured shares of Neutral-condition runs:

  | Model | Explicit completion | No-tool exit | Budget | Technical failure |
  |---|---|---|---|---|
  | Qwen | 62% | 37% | 0% | 1% |
  | Llama | 87% | 8% | 0% | 5% |
  | Gemma | 0% | 100% | 0% | 0% |
  | gpt-oss | 21% | 74% | 0% | 5% |

  Earlier draft values (30/68, 72/20, 0/98, 15/76) did not come from the data
  and should not be used.
- **Recovery arrows (5.4)** are Suff. + Saturation final − local coverage:
  Qwen +12.7, Llama +2.4, Gemma +1.2, gpt-oss +8.8 pp. Neutral recovery is
  +11.9 / +2.0 / +1.5 / +8.2 pp.
- **Net harm (5.6, 5.7).** Positive values mean answers got worse; this is
  the negative of the accuracy change.
- **Significance stars (5.7)** come from exact McNemar p-values:
  *** p < 0.001, ** p < 0.01, * p < 0.05.
- **Shading (5.7).** Each evidence class has its own colour pair. The lighter
  (left) bar is Fixed Sufficiency; the darker (right) bar is
  Suff. + Saturation.
