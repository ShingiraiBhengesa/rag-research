# E2 — bridge-mechanism study: results

Design, arms and outcome definitions: `experiments/whitebox/e2_bridge_mechanism/README.md`.

## `v1_replay_full/` — first full run (superseded; read the caveat)

Full canonical split (450 questions per dataset), Qwen2.5-14B and Llama-3.1-8B,
HotpotQA and 2Wiki, eight arms.

**Caveat — do not cite these contrasts as final.** In v1, each cue arm resumed
the victim from a *replayed* copy of the neutral run's first search observation.
The replay dropped the agent's `think_tool` call and the supervisor's planning
message. As a result, the neutral arm itself did not reproduce the canonical E1
behaviour (`diagnostics/replication_vs_canonical.csv`). v2 runs every arm
fresh, with the anchor's prompt date pinned. The v2 results will be added
beside this folder as `v2_fresh/` when that run completes.

Headline v1 observations (descriptive only):
- None of the pre-registered primary contrasts was significant.
- Qwen-14B PCR@1 rose under bridge-structured cues (+16.6 points on 2Wiki,
  +10.4 points on HotpotQA).
- The fixed sufficiency cue was stronger than the dynamic one.

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
