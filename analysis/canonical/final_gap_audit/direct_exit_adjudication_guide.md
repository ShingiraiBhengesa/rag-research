# Human adjudication guide for `direct_exit_human_review_sample.csv`

This sample contains 552 rows drawn via a reproducible, seed-fixed
(`config.DETERMINISTIC_SEED = 20260910`) stratified sample — up to 3 rows
per (model, dataset, condition, early/late-step) stratum — from the 9,580
`NO_TOOL_DIRECT_EXIT` rows that deterministic structural rules could NOT
confidently classify (95.8% of all such rows). Columns prefixed `_` (model,
dataset, condition, example_id, step) are the de-anonymization key; hide
them if you want to review fully blind, then reveal them afterward to check
for systematic patterns.

## What you are looking at

`response_text_preview` is the first 300 characters of the actual researcher
response text that caused this row to be classified `NO_TOOL_DIRECT_EXIT`
(i.e. the model produced this text with an empty tool-call list, which is
what ends the researcher episode without another search). `preliminary_auto_category`
is this audit's PRELIMINARY, length-based guess — treat it as a null
hypothesis to confirm or overturn, not as ground truth.

## Categories (assign exactly one to `human_category`)

- **`SUBSTANTIVE_DIRECT_ANSWER`** — a concise, direct answer to the research
  question (a short factual claim, name, date, yes/no, etc.), not padded
  with extensive research narration.
- **`SUBSTANTIVE_RESEARCH_SUMMARY`** — a longer-form synthesis of findings
  (multiple paragraphs, headers, "Sources" section, etc.) that reads like a
  mini-report rather than a short answer, even if a correct answer is
  embedded somewhere inside it.
- **`REFUSAL`** — the model declines to answer or claims it cannot/does not
  have enough information, without a substantive attempt.
- **`EMPTY_OR_WHITESPACE`** — should not appear in this sample (these are
  auto-classified), flag if you see one as a bug.
- **`ERROR_MESSAGE`** — should not appear in this sample (auto-classified),
  flag if you see one as a bug.
- **`PROCESS_COMMENTARY`** — a short statement about the research process
  itself ("I have gathered sufficient information to proceed.") with no
  actual answer content.
- **`MALFORMED_TOOL_ATTEMPT`** — the text is an attempt to invoke a tool
  (JSON-like syntax, a tool name, function-call-looking text) that failed to
  register as a real tool call. Should mostly be auto-classified already,
  but borderline/partial cases may appear here.
- **`UNCLEAR`** — none of the above fit confidently, or the text is
  ambiguous/truncated/garbled beyond confident classification.

## Additional judgment notes

- Judge the response ON ITS OWN — do not use the gold answer or the eventual
  final report to decide the category. Content classification and
  correctness are different questions (this audit never uses gold-answer
  containment as evidence of correctness — see `direct_exit_content_audit.csv`'s
  `downstream_final_report_agrees_prefix_match` column, which is a
  diagnostic only, not a correctness signal).
- A response that is both a short answer AND ends with 1-2 sentences of
  supporting rationale is still `SUBSTANTIVE_DIRECT_ANSWER` — reserve
  `SUBSTANTIVE_RESEARCH_SUMMARY` for genuinely report-like, multi-paragraph
  outputs.
- If the 300-character preview is truncated mid-sentence and you cannot
  tell which category applies, mark `UNCLEAR` and note "needs full text" in
  `human_notes` rather than guessing from a truncated fragment.

## After adjudication

Once `human_category` is filled in for all 552 rows, compute agreement
between `preliminary_auto_category` and `human_category` (a simple
percent-agreement or Cohen's kappa) and report it alongside the corrected
overall proportions in a follow-up note — this audit deliberately stops
short of doing that itself, since the whole point of this sample is that a
human, not this script, makes the final call.
