# qwen_llama_results_bundle

Frozen, verified results bundle for the completed `qwen14b`/`llama8b`
production runs (12 conditions x 4 datasets x 2 models). Built from
`results/traces/{qwen14b,llama8b}/` and `results/runs/{qwen14b,llama8b}/`
via `analysis/canonical/canonical_completion.py` (the sole authoritative
PCR/route/eligibility source -- see
`analysis/canonical/chapter3_final_source_of_truth.md`).

## Regenerating everything

    conda activate whitebox
    python -m analysis.canonical.results_bundle.build_bundle \
        --bundle-root qwen_llama_results_bundle --stage all

`--stage data` (default) runs steps 1-11 (freeze through validation gates)
and writes everything under `data/`. `--stage figures`/`--stage tables`/
`--stage workbook` consume `data/*.csv` only and can be re-run independently
once `data/` exists. `--stage all` runs all four in order.

No file in this bundle is hand-edited except the narrative sections marked
`_TODO(human)_` in `MASTER_RESULTS.md` -- everything else, including every
number in every `.md`/`.tex` file and every figure, is generated from
`data/*.csv` and will be silently overwritten by a re-run. If you edit a
number by hand, the next `--stage all` run erases the edit; fix the
generator in `analysis/canonical/results_bundle/`, not the output file.

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
- `qwen_llama_results_workbook.xlsx` -- one sheet per `data/*.csv`.
- `analysis_manifest.json` -- the data freeze: git commit, input file
  hashes, shard completion, active-job check (must be empty for these two
  models).
- `data/VALIDATION_GATES.csv` -- the 13 validation gates (spec section 10);
  the bundle is not final until every row here is PASS. (Currently 13/13.)
- `adjudication/` -- the blinded semantic-adjudication packet for the 543
  rows `data/answer_run_level.csv` itself flags as `needs_adjudication_flag
  == True`. See `adjudication/README.md`. Not yet reviewed by a human; the
  bundle's answer-accuracy numbers remain pre-adjudication until it is.

## Locked scope and definitions

Models: qwen14b, llama8b only. Datasets: HotpotQA, 2WikiMultiHopQA,
MuSiQue, MultiHopRAG. All 12 production conditions. PCR = **Post-observation
Completion Rate**, never "Premature" -- see `MASTER_RESULTS.md` section 1
for the full locked-definitions text. Gemma27b and GPT-OSS-20B are excluded
from this bundle until their run matrices pass the same completion/
eligibility checks; legacy Attack-A/WGAR/Cue-Only/Bridge-Fact results live
under `archive/legacy_results/`, never merged here (gate 13 checks this).
