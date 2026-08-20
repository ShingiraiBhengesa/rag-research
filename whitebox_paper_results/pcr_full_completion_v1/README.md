# White-Box PCR Paper Rerun

This folder is reserved for the full-completion PCR rerun used for paper-ready
results. It is intentionally separate from `whitebox_results/`.

## Layout

- `samples/`: frozen per-dataset raw sample manifests.
- `baselines/{model}/{dataset}/`: clean baseline JSONLs used to compute eligibility.
- `manifests/`: global eligible sample manifests, one per dataset.
- `jsonl/{model}/{dataset}/`: per-model, per-dataset output JSONL files.
- `logs/{model}/{dataset}/`: SLURM stdout/stderr for the matching jobs.
- `ddg_cache/`: DuckDuckGo cache for this rerun only.
- `sbatch/`: generated per-model, per-dataset SLURM scripts.
- `submit_order.md`: generated submit guide.
- `submit_pipeline.py`: generated helper that submits jobs with SLURM dependencies.

## Experiment Rules

The attack scripts and metrics are unchanged. The rerun changes batching,
output paths, and sample control:

- Raw examples are frozen once per dataset before any model is run.
- Clean baselines are rerun for every model/dataset on those frozen examples.
- The final attack sample set for each dataset is the intersection of examples
  that are clean-eligible for every configured model.
- WGA writes clean rows plus the five Wrong Gap Assessment cue conditions.
- FactOnly and CueOnly must be run after the matching WGA job, because they use
  its clean rows as the manifest through `WB_WGA_PATH`.
- `WB_MAX_STEPS` defaults to 40 in the generator, matching the final-run safety
  cap and avoiding the interrupted 15-step rerun cap.
- PCR rows still include `step_trace_*`, `tool_trace_*`, completion steps, and
  search counts.

Generate or refresh the sbatch files with:

```bash
python whitebox_attack/generate_paper_rerun_sbatch.py
```

Submit the full dependency pipeline with:

```bash
python whitebox_paper_results/pcr_full_completion_v1/submit_pipeline.py
```
