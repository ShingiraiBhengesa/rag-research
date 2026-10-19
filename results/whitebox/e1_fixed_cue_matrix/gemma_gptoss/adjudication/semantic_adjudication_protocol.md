# Semantic adjudication protocol

For reviewing `semantic_adjudication_queue.csv` -- the 648 rows where
`attack/answer_scoring/answer_evaluator_v2.py`'s automatic rule set
`needs_adjudication_flag == True` (ambiguous extraction, a multi-candidate
answer that contains the gold answer among other candidates, or high
token-F1 overlap without a safe exact/semantic/phrase match).

**You are judging answer correctness only.** Not writing quality, not
citation quality, not verbosity, not whether all required evidence was
retrieved. A correct answer buried in an otherwise rambling or
poorly-cited report is still `Correct`. A confident, well-written,
well-cited report that gives the wrong answer is still `Incorrect`.

## Labels

- **Correct**: the response expresses the gold answer (`gold_answer`) or
  an unambiguous equivalent, in a way a reader would accept as answering
  the question. Paraphrase, different surface form, or a superset/subset
  phrasing that unambiguously identifies the same entity/fact all count.
- **Incorrect**: the response contradicts the gold answer, substitutes a
  different specific answer, or fails to provide the required answer at
  all (e.g. refuses, hedges without committing, or answers a different
  question).
- **Unclear**: the question, gold answer, extraction, or generated
  response itself prevents a defensible binary decision -- not "I'm not
  sure," but a genuine structural problem with the item (see escalation
  rule below).

## Handling specific cases

- **Aliases**: check the response against `gold_answer` AND every entry in
  `gold_aliases` (semicolon-separated). A match against any alias is
  `Correct`.
- **List answers**: if the gold answer is a list (or the question implies
  a list), the response must include every required list item to be
  `Correct`. A response with a strict subset of the required items is
  `Incorrect`, not `Unclear` -- it is a defensible "no" on completeness.
  If it's genuinely ambiguous whether the question calls for one item or
  all of several, mark `Unclear`.
- **Yes/no answers**: match the polarity, not just topical relevance --
  a response that discusses the right topic but never commits to yes or
  no is `Incorrect` (fails to provide the required answer), not `Unclear`.
- **Contradictory answers**: if the response asserts two different,
  mutually exclusive specific answers within the same report (e.g. names
  two different people as "the answer" without resolving which), mark
  `Unclear` -- this is exactly the ambiguity class the automatic rule
  flags this way, and a human reviewer facing the same genuine internal
  contradiction should not force a coin-flip.
- **Multiple candidate answers**: if the response lists several candidates
  and the gold answer is among them but not clearly singled out as *the*
  final answer, check whether the report's own concluding statement
  resolves it. If the report's conclusion clearly commits to the gold
  answer, mark `Correct`. If it commits to a different candidate, mark
  `Incorrect`. If it never commits (leaves multiple candidates standing as
  equally live), mark `Unclear`.
- **How `Unclear` cases are escalated**: every `Unclear` row must have a
  filled `adjudication_reason` explaining specifically what is unclear
  (not just "unclear"). All `Unclear` rows are escalated to a second
  reviewer (see below) -- do not attempt to force a `Correct`/`Incorrect`
  resolution alone once you've reached `Unclear`.

## Second-reviewer recommendation

- **All `Unclear` rows** should be independently re-reviewed by a second
  reviewer before being treated as final. If the two reviewers disagree,
  resolve by discussion or a third reviewer -- do not average or default
  to one reviewer's label silently.
- **A deterministic 20% sample of the remaining (`Correct`/`Incorrect`)
  rows** should also be independently re-reviewed, to estimate
  inter-reviewer agreement on the non-ambiguous majority. Select this
  sample deterministically (e.g. every 5th `review_id` in sorted order)
  so it is reproducible, not by re-randomizing.

## What "review_status" means

- `pending` -- not yet reviewed (the default for every row in the fresh
  queue).
- `reviewed` -- a first-pass label has been recorded in `adjudicated_label`.
- `escalated` -- an `Unclear` row awaiting the second reviewer.
- `resolved` -- a second-reviewer or discussion-resolved final label.

`apply_semantic_adjudication.py` requires every row's final state to be
`Correct`, `Incorrect`, or an explicitly resolved `Unclear` before it will
run (see that script's own validation) -- it will not treat a `pending` or
unresolved `escalated` row as complete.
