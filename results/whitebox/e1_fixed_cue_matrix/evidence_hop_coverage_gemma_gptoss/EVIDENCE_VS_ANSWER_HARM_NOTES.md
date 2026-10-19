# Evidence-incompleteness vs paired answer harm -- notes

Smallest justified next analysis after `EVIDENCE_HOP_RESULTS.md` (section 9). Tests whether pairs where the CUE condition was evidence-INCOMPLETE at local completion (`local_full_evidence == False` in `run_level_evidence_hops.csv`) show a higher control-correct/cue-wrong ("harmed") rate than pairs where the cue condition WAS evidence-complete, across every comparison in `prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons.COMPARISONS`, pooled over all datasets/models/comparisons.

**This is an association check only, not a causal or thesis-ready claim** -- pairs are pooled across comparisons/datasets for this top-line number; see `evidence_incomplete_vs_answer_harm.csv` for the full per-comparison, per-model, per-dataset breakdown (each row also carries its own McNemar p-value) before citing anything.

## Pooled top-line comparison

Within-stratum McNemar's test asks: given this evidence-completeness state, is the cue's effect on answer correctness (control-correct/cue-wrong vs control-wrong/cue-correct) distinguishable from chance? The two strata are disjoint pair sets (not paired against each other), so comparing them directly uses non-overlap of the Wilson 95% CI on the harmed rate, not a paired test.

| Cue evidence at local completion | n_pairs | control_correct_cue_wrong | control_wrong_cue_correct | harmed rate (95% CI) | McNemar p (n_discordant) |
|---|---|---|---|---|---|
| Complete | 31086 | 3371 | 3520 | 10.84% (10.50%-11.19%) | 0.0746 (n=6891) |
| Incomplete | 16231 | 1486 | 1190 | 9.16% (8.72%-9.61%) | 0.0000 (n=2676) |

The pooled harmed rate among evidence-incomplete cue completions is **lower than** the rate among evidence-complete cue completions (9.16% vs 10.84%). Their 95% Wilson CIs **do not overlap** -- the two strata's harmed rates are distinguishable at the pooled level, though this still pools across heterogeneous comparisons/datasets/models and is not a per-comparison claim.


Within the evidence-incomplete stratum, the cue's effect on paired answer correctness is itself significant by McNemar's test (p=0.0000, n_discordant=2676).

## Known limitations

- Pairs are pooled across all comparisons/datasets/models for the top-line table above; the CSV has the disaggregated view, each with its own McNemar p-value (small per-cell n means most individual rows will not reach significance -- use the pooled numbers or aggregate deliberately, not a single row's p-value, for any claim).
- Answer correctness is pre-adjudication (automatic semantic label), same caveat as `results/whitebox/e1_fixed_cue_matrix/qwen_llama`'s answer tables.
- Technical-failure and no-evidence-data pairs are excluded, not imputed (counts in the CSV's `n_excluded_*` columns).
- This does not establish that evidence loss CAUSES the answer flip -- only whether the two co-occur more than chance-pooled would suggest, and whether that co-occurrence itself looks like chance within each stratum.
