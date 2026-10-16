# Semantic adjudication

Blinded human-review packet for the 543 rows where
`qwen_llama_results_bundle/data/answer_run_level.csv::needs_adjudication_flag
== True` -- i.e. where automatic semantic accuracy
(`attack/answer_scoring/answer_evaluator_v2.py::semantic_accuracy`) itself
could not confidently resolve `Correct`/`Incorrect`. Automatic semantic
accuracy is not missing and is not recomputed here; this directory exists
to adjudicate specifically the subset the automatic rule flagged as
ambiguous.

## Files

- `semantic_adjudication_queue.csv` -- **the file to review.** 543 rows,
  blinded (no `model`, `condition`, run ID, or cue/control identity), rows
  shuffled with the repository's own deterministic seed (`20260910`). No
  label is prefilled; every row starts `review_status=pending`.
- `semantic_adjudication_key.csv` -- the de-anonymization key
  (`review_id` -> `model`/`dataset`/`condition`/`example_id`). Keep this
  separate from whoever is doing the blind review; only
  `apply_semantic_adjudication.py` needs it, to fold completed labels back
  into the pipeline's own identifiers.
- `semantic_adjudication_protocol.md` -- the labeling rules (Correct/
  Incorrect/Unclear), how to handle aliases/lists/yes-no/contradictions/
  multiple candidates, and the second-reviewer recommendation.
- `build_semantic_adjudication_packet.py` -- regenerates the two CSVs
  above from `data/answer_run_level.csv` + `results/manifests/*.jsonl`.
  Already run once to produce the current queue/key; re-running it would
  reshuffle and reset any in-progress review, so do not re-run until you
  intend to discard the current review state.
- `apply_semantic_adjudication.py` -- **not yet run.** Folds a completed
  review back into three new files (`data/answer_run_level_adjudicated.csv`,
  `answer_summary_adjudicated.csv`, `answer_paired_flips_adjudicated.csv`).
  Refuses to run until every one of the 543 rows has a final
  `Correct`/`Incorrect`/resolved-`Unclear` state.

## What "blinded" means here, precisely

The queue exposes `dataset`, `question`, `gold_answer`, `gold_aliases`,
`extracted_short_answer`, `raw_final_report`, and `automatic_semantic_label`
-- never `model`, `condition`, the original run ID, or which side of a
cue/control pair a row belongs to. Note that `raw_final_report` is the
verbatim generated report and can itself contain `benchmark://` citation
URLs whose IDs embed the example's own hash -- this is unavoidable (it is
the content being reviewed) and does not reveal model or condition.

## Known limitation carried into `apply_semantic_adjudication.py`

`answer_run_level.csv` does not preserve `AnswerRecord.extractable` (only
`extracted_short_answer`, `extraction_reason`, and the derived labels).
`apply_semantic_adjudication.py`'s adjudicated summary therefore omits
`n_valid_answer`/`pct_valid_answer` rather than approximate them --
documented in that script's own docstring.

## Next step

Have a human reviewer work through `semantic_adjudication_queue.csv`
following `semantic_adjudication_protocol.md`. Only after every row reaches
a final state should `apply_semantic_adjudication.py` be run.
