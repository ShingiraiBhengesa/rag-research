# Final model status (Phase 5)

Read-only `squeue`/shard-file snapshot taken 2026-09-11, same session as the
rest of this final gap audit.

| Model | Status | Planned shards | Existing shards | Valid rows | PCR-eligible | Answer-evaluated |
|---|---|---|---|---|---|---|
| `qwen14b` (Qwen2.5-14B-Instruct) | `INCLUDED_COMPLETE` | 432 | 432 | 21,576 | 21,408 | 21,523 |
| `llama8b` (Llama-3.1-8B-Instruct) | `INCLUDED_COMPLETE` | 432 | 432 | 20,971 | 19,696 | 20,902 |
| `gemma27b` (Gemma-3-27B-IT) | `PARTIAL_ACTIVE` | 432 | 414 | not computed | not computed | not computed |
| `gpt_oss_20b` (GPT-OSS-20B) | `BLOCKED` | 432 | 0 | 0 | 0 | 0 |
| `qwen7b` (Qwen2.5-7B-Instruct) | `VALIDATION_ONLY` | n/a | 30 | not recomputed | n/a | n/a |

**`qwen14b`, `llama8b` — `INCLUDED_COMPLETE`.** Both have all 432/432 shard
files, zero active/queued jobs in `squeue`, and no permanent blocker. These
are the only two models included in every canonical PCR/route/direct-exit
computation in this task and the prior canonical-completion task.

**`gemma27b` — `PARTIAL_ACTIVE`.** 414/432 shards present; `squeue` shows
job `147621` actively `RUNNING` (task 89, on node `ml01`) with 18 more tasks
`PENDING` under a `JobArrayTaskLimit` throttle — i.e. it is genuinely still
making progress, not stalled. Per this task's own read-only-safety
constraints and the pre-existing analysis package's own scope
(`analysis/config.py::INCLUDED_MODELS = ("qwen14b",
"llama8b")`), no per-row valid/eligible/answer-evaluated count was computed
for gemma27b in this audit — doing so would require touching its
actively-written output directory, which is unnecessary I/O against a live
job and was avoided. **Not included in the final canonical thesis analysis
as currently constituted**; may become includable once its matrix completes
and a supervised, deliberate decision is made to extend `INCLUDED_MODELS`.

**`gpt_oss_20b` — `BLOCKED`.** 0/432 shards; all 4 production arrays are
`PENDING` in `squeue` with reason `DependencyNeverSatisfied` — each carries
an `afterok:149854_*` dependency on `v3_recovery_gate_gpt_oss_20b`, which
**FAILED** 2026-09-08 (`sacct -j 149854`: exit code 20:0, per `PUBLISHING.md`,
not re-verified via a fresh `sacct` call in this read-only-safety-conscious
audit since the historical fact of the failure is already documented and
`squeue`'s live `DependencyNeverSatisfied` reason independently confirms the
practical consequence). An `afterok` dependency on a permanently-failed job
can never be satisfied — this is a structural, not transient, block that
requires a human decision (resubmit without the dependency, or fix the gate)
per `PUBLISHING.md`'s own recommendation. **Excluded from the final thesis
analysis; zero rows exist to include.**

**`qwen7b` — `VALIDATION_ONLY`.** Never part of `FINAL_PRODUCTION_MODELS`
(`attack/production_models.py:8`); its 30-shard pipeline-
validation run is quarantined at
`production/quarantine/qwen7b_pipeline_validation_20260828_174947/`, outside
the live `production/runs/` tree. **Excluded from the final thesis analysis
by original design, not by data availability.**

## Answer to "is the model set final?"

**No, not in the sense of "all four intended models are ready."** The
model SET intended by the production design is fixed
(`FINAL_PRODUCTION_MODELS = ("qwen14b", "gemma27b", "gpt_oss_20b",
"llama8b")`), but only 2 of those 4 currently have usable data
(`INCLUDED_COMPLETE`). The thesis's final included-model set, as of this
audit, is **`qwen14b` and `llama8b` only** — this must be stated plainly,
not implied as "the full model set was studied."
