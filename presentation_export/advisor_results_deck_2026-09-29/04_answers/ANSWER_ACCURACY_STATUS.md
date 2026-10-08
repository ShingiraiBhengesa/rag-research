# Final-Answer Accuracy — Adjudication Status

## Three distinct things (do not conflate — see `RESULTS_CANONICAL_INDEX.md` §6)

1. **Current automatic semantic accuracy** — fully computed today, for every
   valid answer row, by `attack/answer_scoring/answer_evaluator_v2.py::semantic_accuracy()`.
   This is what every number in `04_answers/*_answer_mini_table.csv` and the
   figures in this deck actually shows. It is real, computed, present data —
   **not** missing.
2. **Flagged cases requiring adjudication** — rows where the automatic rule
   itself set `needs_adjudication_flag == True` (ambiguous extraction,
   multi-candidate containment, high token overlap without a safe match).
   Counts per condition are in the `n_needs_adjudication` column of each
   mini table. For Qwen/Llama, these are the same 543 rows already queued in
   `qwen_llama_results_bundle/adjudication/` (blinded review packet, not yet
   reviewed). GPT-OSS has its own flagged rows (visible in its mini table)
   that have not been queued for review at all yet.
3. **Adjudicated (human-reviewed) semantic accuracy** — **does not exist for
   any of the three models**, and will not until a human completes the
   review and `qwen_llama_results_bundle/adjudication/apply_semantic_adjudication.py`
   (or an equivalent script for GPT-OSS) is run.

## Status by model

| Model | Evaluator | Adjudication queue built? | Adjudication complete? |
|---|---|---|---|
| Qwen2.5-14B-Instruct | `answer_evaluator_v2.py::semantic_accuracy()` | Yes (543-row blinded packet, `qwen_llama_results_bundle/adjudication/`) | **No** |
| Llama-3.1-8B-Instruct | Same evaluator | Same packet (shared with Qwen's queue) | **No** |
| GPT-OSS-20B | Same evaluator (identical code path — no separate evaluator was written for this model) | **No** — its flagged rows have not been queued | **No** |

**Every semantic-accuracy number in this entire deck is pre-adjudication.**
Do not present any of it to the advisor as final, human-validated accuracy.
The `pct_semantic_correct_macro_avg` / `Sem. correct %` columns are the
current best automatic estimate, with a disclosed, per-condition count of
how many rows are flagged as uncertain (`n_needs_adjudication`).

## Why GPT-OSS is not force-fit into the Qwen/Llama adjudication narrative

GPT-OSS used the identical automatic evaluator, so its semantic-accuracy
numbers are methodologically comparable to Qwen/Llama's *pre-adjudication*
numbers. But it has no adjudication queue of its own yet — that is real,
separate future work, not an oversight in this export.
