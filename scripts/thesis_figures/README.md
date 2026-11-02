# Thesis figure scripts

These scripts generated Figures 5.1–5.4 of the final thesis, copied here from
the analysis notes where they were developed. Final outputs:
`results/thesis_figures/`.

Run from the repository root with the analysis environment. Each script puts
its own directory on `sys.path`, so the modules import each other directly:

```bash
python scripts/thesis_figures/fig5_1_pcr_curves.py          # Figure 5.1
python scripts/thesis_figures/fig5_2_step_boxplot.py        # Figure 5.2
python scripts/thesis_figures/thesis_restructure_e2.py      # Figure 5.4 (fig_e2_forest); run before --cov
python scripts/thesis_figures/thesis_restructure_e2.py --cov  # Figure 5.3 (fig_e2_pcr_cdf_common)
```

| Module | Role |
|---|---|
| `thesis_restructure.py` | Shared style, constants and E1 paths (base module) |
| `thesis_restructure_v2.py` | E1 statistics and helpers; `E2_DIR` points at `results/whitebox/e2_bridge_mechanism/v2b_full` |
| `thesis_restructure_e2.py` | E2 analyses: forest plot and common-question PCR curves |
| `fig5_1_pcr_curves.py`, `fig5_2_step_boxplot.py`, `fig_e2_pcr_cdf.py` | Single-figure scripts |

Outputs:

* Figures go to `results/figures/thesis_restructure_v2/` and
  `results/figures/thesis_restructure_e2/`, which are gitignored scratch.
* CSVs go to `notes/reviewer_checks/{e2,cov}/`, also gitignored.

Copy a figure into `results/thesis_figures/` only when it replaces a figure in
the thesis, and update the checksums in that folder's README.

Changes from the notes copies:

* `E2_DIR` reads the published v2b run instead of `local_data/`.
* `task_a_common` draws the 50% reference line and the "n = … common
  questions" labels again. With those restored it reproduces the thesis's
  Figure 5.3 pixel for pixel.
