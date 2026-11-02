# E2 — bridge-anchored study: results

Design, arms and outcome definitions: `experiments/whitebox/e2_bridge_mechanism/README.md`.

## `v2b_full/` — thesis run

The v2b full run (array 176494, code `4c94fd0a`). Every arm runs fresh, with
vLLM's prefix cache reset before each run. Layout below, plus
`fidelity_check.csv`. This is the run the thesis reports. Its
main evaluation uses local measures on matched-d₁ questions; the whole-run
premature stop (`pcr/primary_by_cell.csv`) is reported in Appendix A only.
Arms whose fresh d₁ differs from the anchor's are `d1_mismatch` and excluded;
see the v2b section of the experiment README. The thesis figures are in
`results/thesis_figures/`. The superseded v1 run (replayed first observation)
was removed from the repository on 2026-11-02 and is still in git history.

| Folder | Contents |
|---|---|
| `summaries/PILOT_SUMMARY.md` | Human-readable summary of the run |
| `pcr/` | PCR@k curves, paired contrasts, primary outcome per cell |
| `diagnostics/` | Outcomes, evidence coverage and answer accuracy by arm; baseline gate; replication check |
| `bridge_selection/` | Which bridge fact was selected as the anchor for each question (provenance) |
| `generated_cues/` | Audit of every generated cue (gates V1–V5 and score) |
| `trajectories/` | One row per run (`run_labels.csv`) and per agent step (`trajectory_steps.csv`) |
| `tool_traces/` | Every search call issued by the victim |
| `figures/` | PCR curves per model × dataset |

The raw victim traces that `trace_path` points to are not in git. They live
under `local_data/` (see `src/prematurestop/paths.py`).
