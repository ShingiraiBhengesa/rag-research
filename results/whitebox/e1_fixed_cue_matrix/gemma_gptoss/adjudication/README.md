# Semantic adjudication — Gemma-3-27B / gpt-oss-20b

This is the blinded human-review packet for the **648 rows** of
`data/answer_run_level.csv` with `needs_adjudication_flag == True`. In those
rows the automatic semantic-accuracy rule
(`src/prematurestop/evaluation/answer_scoring.py`) could not confidently decide
`Correct` or `Incorrect`.

Automatic labels exist for every row and are not recomputed here. This
folder exists to review only the ambiguous subset. It is the same procedure,
with the same seed, as the Qwen/Llama packet in
`../../qwen_llama/adjudication/`.

## Files

- `semantic_adjudication_queue.csv` is **the file to review**. It has 648
  rows, blinded: there is no model, condition, run ID or cue identity. The
  rows were shuffled with the repository's deterministic seed (`20260910`).
  No label is prefilled, and every row starts as `review_status=pending`.
- `semantic_adjudication_key.csv` is the de-anonymisation key
  (`review_id` → model, dataset, condition, example_id). Keep it away from the
  reviewer.
- `flagged_counts.csv` gives the number of flagged rows per model × dataset.
- `semantic_adjudication_protocol.md` contains the labelling rules. They are
  identical to the Qwen/Llama protocol.

## Regenerating

    python -m prematurestop.analysis.fixed_cue_matrix.results_bundle.adjudication_packet \
        --bundle-root results/whitebox/e1_fixed_cue_matrix/gemma_gptoss

Re-running reshuffles the queue and resets any review in progress. Do not
re-run it once reviewing has started.

## Status

Review has **not started**. Until it is complete, every accuracy number in
this bundle is an automatic semantic label, not adjudicated accuracy.

To fold completed labels back in, use the logic of
`../../qwen_llama/adjudication/apply_semantic_adjudication.py`, pointed at this
bundle's `data/` folder and its 648-row total.
