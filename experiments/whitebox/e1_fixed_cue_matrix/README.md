# E1 — fixed-cue matrix (white-box)

**Status: complete.** Results: `results/whitebox/e1_fixed_cue_matrix/`.

## Design

E1 crosses 4 victim models, 4 datasets and 12 conditions, with 450 questions
per dataset. Each question runs once per condition with the victim at
temperature 0.

| Factor | Levels |
|---|---|
| Victims | `qwen14b` Qwen/Qwen2.5-14B-Instruct · `llama8b` meta-llama/Llama-3.1-8B-Instruct · `gemma27b` google/gemma-3-27b-it · `gpt_oss_20b` openai/gpt-oss-20b |
| Datasets | HotpotQA · 2WikiMultihopQA · MuSiQue · MultiHop-RAG |
| Conditions | see below; exact cue text in `results/whitebox/e1_fixed_cue_matrix/canonical/condition_metadata.csv` |

There are 12 conditions:

* **`NaturalClean`**: no injection. This is the control.
* **`NaturalNeutral`**: a neutral sentence of matched length is appended to
  d1. This controls for "any appended text".
* **`NaturalContinueSearch`**: a cue that tells the agent to keep searching.
  This tests the opposite direction.
* **Seven fixed premature-completion cues:**
  * Epistemic Confirmation
  * Search Match
  * Source Coverage
  * Epistemic Confirmation + Source Coverage
  * Search Match + Source Coverage
  * Fixed Sufficiency
  * Fixed Sufficiency Saturation
* **Two oracle conditions:** all gold evidence is supplied at once, with
  either the neutral or the saturation cue. These separate the cue's effect
  from the effect of evidence that is actually missing.

The condition registry is in `src/prematurestop/cues/fixed/registry.py`. The
cue families are documented in `src/prematurestop/cues/fixed/families.py`.

## How a run works

1. `pipelines/fixed_cue_matrix/launch.py --mode generate` writes the frozen
   matrix and one SLURM array per model × dataset.
2. Each array task does the following:
   * serves the victim with vLLM;
   * passes `pipelines/fixed_cue_matrix/runtime_gate.py`, a live tool-calling
     check;
   * runs `pipelines/fixed_cue_matrix/production_runner.py` over its shard.
3. The runner (`system/runner.py`) executes the full ODR graph per question.
   Search goes to `system/benchmark_search.py`, a local BM25 index over the
   dataset corpus. The cue is appended to the first retrieved document.
4. `analysis/fixed_cue_matrix/cli.py` parses the traces and computes PCR,
   evidence coverage and answer accuracy. It also writes the result bundles.

## Commands

```bash
# Validate the frozen matrix and generated scripts (no GPU, no network)
python -B -m prematurestop.pipelines.fixed_cue_matrix.launch \
  --mode validate --production-root "$PREMATURESTOP_LOCAL_DATA" --local-files-only

# Submit missing arrays (skips any already queued/running)
python -B -m prematurestop.pipelines.fixed_cue_matrix.launch \
  --mode submit --production-root "$PREMATURESTOP_LOCAL_DATA"

# Rebuild the canonical analysis from traces
python -m prematurestop.analysis.fixed_cue_matrix.cli --stage all \
  --production-root "$PREMATURESTOP_LOCAL_DATA" --output-root <out_dir>
```

Provenance records from the actual run are in
`results/whitebox/e1_fixed_cue_matrix/production_records/`:
* `matrix/` holds the frozen matrix;
* `prelaunch/` holds the validation report;
* `submissions/` holds the SLURM submission log.

These records list file paths from the pre-2026-10-16 layout
(`attack/runtime/...`). `docs/MIGRATION_2026-10-16.md` maps them to the
current paths.

## Findings that shaped the run (keep in mind when reading results)

* **Gemma-3 tool parser.** Gemma-27B's first run used vLLM's `functiongemma`
  parser, which never recognised Gemma-3's ` ```tool_code``` ` tool calls. That
  run is quarantined. The reported Gemma results come from the rerun with
  `system/tool_parsers/gemma3_tool_code.py`. Write-up:
  `results/whitebox/e1_fixed_cue_matrix/canonical/gemma27b_results_synthesis.md`.
* **Trace-parser fix.** ODR's forced-search safety net logs its tool call on
  the *next* event, so the parser originally missed it. The fix changed
  Llama-8B's completion statistics on about 5% of rows. See
  `docs/CHANGELOG_analysis.md`.
* **gpt-oss Harmony parser.** There is a lenient parser for gpt-oss-20b's
  Harmony format, `system/tool_parsers/gptoss_lenient_harmony.py`. It was
  built and tested, but it is not used in the reported gpt-oss run.
  `results/whitebox/e1_fixed_cue_matrix/gemma_gptoss/RESULTS.md` states the
  scope.
