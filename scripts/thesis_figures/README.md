# Thesis figure scripts

These scripts generated Figures 5.2–5.4 and the results tables of the final
thesis, copied here from the analysis notes where they were developed. Final
outputs: `results/thesis_figures/` (figures) and `results/thesis_tables/`
(tables). They need only tracked files: run `git lfs pull` after cloning,
then run them in the order below.

Figures 4.1 and 5.1 come from `python -m prematurestop.analysis.fixed_cue_matrix.thesis_figures`.

Run from the repository root with the analysis environment. Each script puts
its own directory on `sys.path`, so the modules import each other directly:

```bash
python scripts/thesis_figures/thesis_restructure_v2.py        # E1 tables (ASR, step reduction, ...)
python scripts/thesis_figures/thesis_restructure_v2.py --cov  # E1 coverage loss
python scripts/thesis_figures/eicr_timing_reconciled.py       # E1 EICR and timing tables
python scripts/thesis_figures/fig5_2_step_boxplot.py          # Figure 5.2
python scripts/thesis_figures/thesis_restructure_e2.py        # E2 tables, Figure 5.4 (fig_e2_forest)
python scripts/thesis_figures/thesis_restructure_e2.py --cov  # E2 coverage, Figure 5.3 (fig_e2_pcr_cdf_common)
```

| Module | Role |
|---|---|
| `thesis_restructure.py` | Shared style, constants and E1 paths (base module) |
| `thesis_restructure_v2.py` | E1 statistics and helpers; `E2_DIR` points at `results/whitebox/e2_bridge_mechanism/v2b_full` |
| `thesis_restructure_e2.py` | E2 analyses: forest plot and common-question PCR curves |
| `eicr_timing_reconciled.py` | E1 EICR and timing reconciliation (Table 5.3) |
| `fig5_2_step_boxplot.py`, `fig_e2_pcr_cdf.py` | Single-figure scripts |
| `reference/` | Earlier outputs, read only by the `VERIFY` cross-checks |

Outputs:

* Figures go to `results/figures/thesis_restructure_v2/` and
  `results/figures/thesis_restructure_e2/`, which are gitignored scratch.
* CSVs go to `results/thesis_tables/{e1,e2,coverage}/` (tracked). Answer and
  harm outputs there are gitignored, because they are outside the thesis
  scope.

Copy a figure into `results/thesis_figures/` only when it replaces a figure in
the thesis, and update the checksums in that folder's README.

Changes from the notes copies:

* Every input is a tracked file. E2 reads the published v2b run, and the
  Qwen/Llama answers come from the published `answer_run_level.csv`, which has
  labels identical to the local predictions file. Nothing reads `notes/` or
  `local_data/`.
* CSV outputs go to `results/thesis_tables/` instead of `notes/reviewer_checks/`.
* `task_a_common` draws the 50% reference line and the "n = … common
  questions" labels again. With those restored it reproduces the thesis's
  Figure 5.3 pixel for pixel.
