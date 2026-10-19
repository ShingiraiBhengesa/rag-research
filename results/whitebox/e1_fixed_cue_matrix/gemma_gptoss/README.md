# results/whitebox/e1_fixed_cue_matrix/gemma_gptoss

Frozen, verified results bundle for the completed `gemma27b/gpt_oss_20b`
production runs (12 conditions x 4 datasets x 2 models). Built from
`results/traces/{gemma27b,gpt_oss_20b}/` and `results/runs/{gemma27b,gpt_oss_20b}/`
via `src/prematurestop/evaluation/completion.py` (the sole authoritative
PCR/route/eligibility source -- see
`results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md`).

## Regenerating everything

    conda activate whitebox
    python -m prematurestop.analysis.fixed_cue_matrix.results_bundle.build_bundle \
        --bundle-root results/whitebox/e1_fixed_cue_matrix/gemma_gptoss --models gemma27b,gpt_oss_20b --stage all

`--stage data` (default) runs steps 1-11 (freeze through validation gates)
and writes everything under `data/`. `--stage figures`/`--stage tables`/
`--stage workbook` consume `data/*.csv` only and can be re-run independently
once `data/` exists. `--stage all` runs all four in order.

No file in this bundle is hand-edited except the narrative sections marked
`_TODO(human)_` in `MASTER_RESULTS.md` -- everything else, including every
number in every `.md`/`.tex` file and every figure, is generated from
`data/*.csv` and will be silently overwritten by a re-run. If you edit a
number by hand, the next `--stage all` run erases the edit; fix the
generator in `src/prematurestop/analysis/fixed_cue_matrix/results_bundle/`, not the output file.

## Directory guide

- `data/` -- the 13 canonical CSVs (spec section 5) plus 2 supporting files
  (`mechanism_run_level.csv`/`mechanism_summary.csv`, feeding the trace-raster
  figures only -- not part of the spec's required 13).
- `tables/` -- Markdown tables generated from `data/`. Start at
  `tables/TABLE_INDEX.md` for sign conventions and file-join keys before
  citing any number.
- `figures/` -- PNG+PDF+caption triples, one per figure, organized by type.
- `appendix/` -- the same numbers as LaTeX booktabs tables, for direct
  inclusion in the thesis appendix.
- `gemma_gptoss_results_workbook.xlsx` -- one sheet per `data/*.csv`.
- `analysis_manifest.json` -- the data freeze: git commit, input file
  hashes, shard completion, active-job check (must be empty for the
  included models).
- `data/VALIDATION_GATES.csv` -- the 13 validation gates (spec section 10);
  the bundle is not final until every row here is PASS.

## Locked scope and definitions

Models: gemma27b,gpt_oss_20b only. Datasets: HotpotQA, 2WikiMultiHopQA,
MuSiQue, MultiHopRAG. All 12 production conditions. PCR = **Post-observation
Completion Rate**, never "Premature" -- see `MASTER_RESULTS.md` section 1
for the full locked-definitions text. The other production models have
their own sibling bundle under `results/whitebox/e1_fixed_cue_matrix/`;
legacy Attack-A/WGAR/Cue-Only/Bridge-Fact results live
under `archive/legacy_results/`, never merged here (gate 13 checks this).

## Scope notes

- `gemma27b`: Served with the custom `gemma3_tool_code` parser (`src/prematurestop/system/tool_parsers/gemma3_tool_code.py`) for every dataset; an earlier run with the wrong `functiongemma` parser is quarantined and excluded [table: results/whitebox/e1_fixed_cue_matrix/canonical/gemma27b_results_synthesis.md].
- `gpt_oss_20b`: **Parser difference across datasets.** gpt_oss_20b/2wiki was served with the lenient Harmony tool-call parser plugin (`src/prematurestop/system/tool_parsers/gptoss_lenient_harmony.py`, submitted 2026-09-28, runs 2026-10-03/04); hotpot, musique and multihoprag were served without it (2026-09-14/15). All 12 conditions within a dataset share one parser, so within-dataset paired contrasts are valid; cross-dataset comparisons for gpt_oss_20b mix two serving configurations, and its technical-failure rates are not comparable across datasets (per-cell failures: [table: data/run_inventory.csv]).
