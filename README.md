# White-Box RAG Control-Flow Evaluation

This repository contains the code for a white-box evaluation of control-flow
failure modes in Open Deep Research (ODR), an agentic RAG system that iterates
between search, reasoning, and `ResearchComplete`.

The publishable experiment is the benchmark-local production matrix,
implemented in `attack/` and analyzed in `analysis/`. Only the search
provider is replaced with deterministic local retrieval; the ODR graph,
prompts, supervisor, researcher subgraph, `think_tool`, `ResearchComplete`,
compression, and final report generation all execute through
`open_deep_research/`.

This is the only version of the experiment. Earlier iteration used an
internal "v3" label and a `whitebox_attack/` wrapper directory; both have
been dropped from every path, module, and import (2026-09-14) since there
is nothing else to disambiguate from. Pre-final code and superseded result
directories are kept, not deleted, under `archive/` for provenance.

## Repository layout

```text
rag-research/
├── attack/                 # production experiment code (see attack/README.md) -- only the
│   │                          # current attack: nothing superseded or one-off lives here
│   ├── runtime/                 # frozen system imported live by SLURM jobs
│   ├── launch/                    # matrix/sbatch/validation/submission generator
│   ├── answer_scoring/             # validated semantic-accuracy procedure, used by analysis/
│   └── tests/                        # 70 tests
├── analysis/                # canonical offline evaluation (see analysis/README.md)
│   ├── canonical/                # canonical PCR/completion-route implementation + audit trail
│   └── tests/                      # 41 tests (+ 21 in canonical/)
├── open_deep_research/       # the ODR agent itself (graph, prompts, state, config, tools)
├── results/                  # production data: matrix, traces, evaluation outputs (git-ignored
│   │                          # except matrix/prelaunch/evaluation summaries -- see Results Policy)
├── archive/                  # superseded code and result directories -- kept on disk for
│   │                          # provenance, entirely git-ignored (see .gitignore)
├── requirements-whitebox.txt  # pinned deps for the "whitebox" environment (see Environment setup)
└── PUBLISHING.md              # active-job lock, publication checklist, deferred cleanup log
```

## Environment setup

Reproducing a full production run needs three separate conda/virtualenv
environments (Python 3.11 in all three here), matching the three roles a
production SLURM job plays -- serve the model, run the ODR graph, and
drive/validate/analyze everything else:

| Env (example name) | Role | How to build it |
|---|---|---|
| `whitebox` | Runs `attack/tests`, `analysis/`, `attack/launch/generate_production_arrays.py`, `attack/runtime/runtime_gate.py` and `production_manifest.py` (the launch/validation/analysis harness -- never the ODR graph itself) | `pip install -r requirements-whitebox.txt` |
| `odr` | Runs the actual ODR graph during a production shard (`attack/runtime/runner.py`, `benchmark_search.py`, `production_runner.py`) | `pip install -e open_deep_research/` (its `pyproject.toml` declares langgraph/langchain and everything else needed) |
| `vllm` | Serves the model being evaluated over an OpenAI-compatible endpoint | Install `vllm` (tested against `vllm==0.23.0`, `torch==2.11.0`, `triton==3.6.0`, `xgrammar==0.2.2`) per the [vLLM docs](https://docs.vllm.ai/) for your hardware |

The generated sbatch scripts (`attack/launch/generate_production_arrays.py`)
hardcode absolute interpreter paths for these three envs
(`WHITEBOX_PYTHON`/`ODR_PYTHON`/`VLLM_CONDA_ENV` near the top of that file) --
update those constants to point at your own environments before generating
new scripts. Everything else in this repository (paths, model IDs, dataset
names) is already repo-relative and machine-independent.

No GPU is required to run the test suite, `analysis.cli`, or `--mode
validate`/`--mode generate` (`--local-files-only` avoids any network calls) --
only actually executing a production shard or the runtime gate needs a GPU
(via the `vllm`/`odr` environments).

## Final production scope

Final models:

- `qwen14b`: `Qwen/Qwen2.5-14B-Instruct` -- **complete** (432/432 shards)
- `llama8b`: `meta-llama/Llama-3.1-8B-Instruct` -- **complete** (432/432 shards)
- `gemma27b`: `google/gemma-3-27b-it` -- **complete** (432/432 shards, finished 2026-09-13)
- `gpt_oss_20b`: `openai/gpt-oss-20b` -- **blocked** (0/432 shards; dead SLURM
  dependency, see `PUBLISHING.md`; being fixed and resubmitted)

Final datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiHopRAG.

Final conditions are defined in `attack/runtime/cues.py` (12 total; see
`analysis/canonical/condition_metadata.csv` for the full verbatim cue-text
table). The final production matrix is recorded at:

```text
results/matrix/v3_final_production_matrix.json
```

`qwen7b` is not part of the final production matrix; its artifacts are
quarantined under `results/quarantine/` and excluded from all final
aggregate claims.

## Notable finding: Gemma-3-27B-IT's tool-calling does not function in this pipeline

Gemma-3-27B-IT, served via vLLM with the `functiongemma` tool-call parser,
writes its intended tool invocation as plain markdown text (a
` ```tool_code\ntavily_search(...)``` ` block) rather than a real structured
tool call, in 98.9% of its 21,600 production rows (deterministically
confirmed, not a guess). This means its episodes are driven almost entirely
by ODR's `force_benchmark_search` safety net rather than the model's own
tool-use decisions -- a real, reportable finding about model/serving-stack
compatibility, not a bug in the experiment. Discovering this also exposed
and fixed a genuine gap in the offline trace-analysis parser (it never
recognized that safety net as a valid PCR branch point), which materially
corrected `llama8b`'s already-reported completion statistics too (~5% of
rows). Full writeup:
`analysis/canonical/gemma27b_analysis/gemma27b_results_synthesis.md`;
fix details in `analysis/CHANGELOG.md`.

## Primary code paths

- `attack/runtime/production_models.py` -- final model allowlist.
- `attack/runtime/production_manifest.py` -- frozen dataset manifests.
- `attack/launch/generate_production_arrays.py` -- matrix, sbatch,
  prelaunch validation, and duplicate-safe submission.
- `attack/runtime/runtime_gate.py` -- per-model runtime validation before
  production shards execute.
- `attack/runtime/production_runner.py` -- array shard execution.
- `analysis/` -- canonical offline evaluation pipeline (PCR/completion
  routes, evidence coverage, answer accuracy, pairing).
- `open_deep_research/src/open_deep_research/` -- the ODR graph, prompts,
  state, configuration, and tool plumbing.

## Reproducibility commands

Validate the frozen matrix and scripts:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.launch.generate_production_arrays \
  --mode validate \
  --production-root results \
  --local-files-only
```

Submit missing final arrays without duplicating active jobs:

```bash
/home/sbhengesa/.conda/envs/whitebox/bin/python -B \
  -m attack.launch.generate_production_arrays \
  --mode submit \
  --production-root results
```

Refresh canonical aggregate outputs after jobs complete:

```bash
python -m analysis.cli --stage all \
  --production-root results \
  --output-root results/evaluation/v3_final_completed_qwen_llama_gemma_20260913
```

Run the full test suite (131 tests: 70 in `attack/tests` + 41 in
`analysis/tests` + 20 in `analysis/canonical`; a further 47 tests for
archived/superseded evaluators live at `archive/legacy_evaluators`, kept on
disk for provenance but not run as part of routine verification):

```bash
python -m unittest discover -s attack/tests
python -m unittest discover -s analysis/tests
python -m unittest analysis.canonical.test_canonical_completion
```

## Current publication rule

While SLURM jobs from the final matrix are queued or running (currently:
`gpt_oss_20b`), do not rename, delete, or edit the runtime-imported files
listed in `attack/README.md`'s "Active-Job Lock" section (everything under
`attack/runtime/`, `attack/launch/generate_production_arrays.py`, and the
ODR modules). Queued jobs import those files from this working tree when
they start.

Safe cleanup during active jobs is limited to documentation, the
`analysis/` package, ignored validation artifacts, and quarantine/archive
moves that do not change final-model run directories.

## Results policy

Raw traces, logs, caches, and per-shard runtime files are intentionally
ignored by git. Publishable outputs are the final matrix, prelaunch report,
submission records, and aggregate evaluation tables/reports under:

```text
results/evaluation/v3_final_completed_qwen_llama_gemma_20260913/
results/evaluation/final_production/
```

Within `v3_final_completed_qwen_llama_gemma_20260913/`, only the small
aggregate tables are tracked (`pcr/`, `statistics/`, `by_model_dataset/`,
`answers/final_answer_summary.csv`). The large raw/per-row artifacts in the
same directory -- `answers/final_answer_predictions.jsonl` (~99M),
`evidence/*.csv` (per-example rows, ~320k/~64k lines), and all of `audits/`
(row/violation-level audit CSVs) -- are git-ignored: fully reproducible from
traces + code via `analysis.cli`, not needed to verify or cite the paper's
claims, and too large to publish.

The aggregate report must be regenerated after `gpt_oss_20b`'s final jobs
finish before all-four-model paper numbers are cited.

## Reorganization history

2026-09-13/14: the codebase was restructured twice for publication
readiness -- first an internal `whitebox_attack/v3/` subpackage split, then
a full top-level flattening dropping the `whitebox_`/`v3` naming entirely
(`whitebox_attack/v3/*` → `attack/*`, `whitebox_attack/v3/analysis/` →
`analysis/`, `whitebox_paper_results/e2e_full_system_v3/production/` →
`results/`). All pre-final code and superseded result directories were
moved to `archive/`, never deleted. See `analysis/canonical/reorg_prep/`
for the exact scripts that performed both moves, and `attack/README.md` /
`analysis/README.md` for the resulting package layouts.
