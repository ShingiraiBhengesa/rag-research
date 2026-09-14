# Chapter 2 required corrections (Phase 8)

**Blocked: no thesis project (and therefore no Chapter 2 file) was found on
this machine — see `thesis_project_inventory.md`.** No sentence could be
located, quoted, or corrected because there is no `content/chapters/chapter2_related_work.tex`
(or any `.tex` file resembling it) anywhere under `/home/sbhengesa`. This
file could not fulfill its literal purpose (finding and quoting exact
current sentences with file:line); nothing below is invented content
attributed to a real file.

## What this audit CAN provide instead: the exact template to apply once Chapter 2 is located

Search Chapter 2 for sentences matching these patterns (case-insensitive):
`"identical retrieval"`, `"same evidence"`, `"same top-k"`, `"fixed passage"`,
`"guaranteed same"`, `"controls retrieval"`, `"same retrieved"`, `"identical
passage"`, `"same document"` combined with words like `"clean"`/`"attack"`/
`"paired"`/`"condition"`. Any sentence asserting that clean and attack
conditions receive **identical** retrieval results, that the retrieved
document remains **fixed** across paired runs, that the same retrieved
evidence is **guaranteed** across conditions, or that retrieval **ranking**
is identical across conditions, conflicts directly with this repo's own
measured evidence:

- `analysis/canonical/retrieval_equivalence_summary.md`:
  only 42.5% of 63,314 readable paired traces share an identical pre-suffix
  observation hash; only 50.7% used an identical first search query; 33.7%
  retrieved genuinely different evidence.
- The pre-existing `strict_pairing_audit.csv`: only 34.8% of 77,411
  condition-pair-example rows pass the stricter combined
  `validate_paired_traces` invariant.
- `chapter3_source_of_truth.md` §7/§14 and
  `chapter3_final_source_of_truth.md` §11: production calls
  `run_single_condition` independently per condition, never
  `run_paired_replay` — there is no structural mechanism in the actual
  production entry point that would force identical retrieval across
  conditions.

## Recommended replacement sentence template

Use wording of this shape (adapt to the surrounding paragraph's voice, do
not paste verbatim without checking it fits):

> "Within a single run, the attack appends a short cue to the retrieved
> passage after retrieval has already occurred; it does not alter which
> passage was selected. Because each experimental condition for a given
> example is executed as an independent run, the model's own search query
> can differ across conditions, and empirically the retrieved evidence
> differs in roughly a third of paired cases (Section 3.X). The underlying
> corpus, BM25 index, and retriever configuration are held fixed across all
> conditions and models; retrieval-content equivalence between paired
> conditions is measured directly from trace data rather than assumed by
> design."

This preserves the four required distinctions:
1. **Within a run**: the cue is appended after retrieval, never altering the
   selected passage (`attack/benchmark_search.py:309`,
   confirmed in every prior audit pass).
2. **Across conditions**: model-generated queries may differ, producing
   different retrieved evidence (measured, not assumed — see the exact
   percentages above).
3. **Corpus/index/config fixed**: BM25 index hash and retriever config hash
   are 100% identical across all 63,314 readable pairs measured
   (`retrieval_equivalence_summary.md`) — this part of the prior claim IS
   fully supported and should be kept.
4. **Measured, not assumed**: cite the specific percentage(s) rather than
   asserting equivalence categorically.

## Action required

Once the actual Chapter 2 file is available, re-run this same search
against it and produce the exact file:line-quoted corrections this phase
originally asked for. Nothing here should be treated as evidence that
Chapter 2 does or does not currently contain a problematic sentence — that
remains genuinely unknown until the file is located.
