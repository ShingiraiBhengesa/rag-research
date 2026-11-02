# Evidence-incompleteness vs paired answer harm -- notes

> **Superseded 2026-10-19** by `results/whitebox/_out_of_scope/answer_harm/ANSWER_HARM_RESULTS.md`. This v1 check pools attack and control comparisons and stratifies on the cue run's evidence alone, which is confounded with question difficulty. Keep for provenance; cite v2.


Smallest justified next analysis after `EVIDENCE_HOP_RESULTS.md` (section 9). Tests whether pairs where the CUE condition was evidence-INCOMPLETE at local completion (`local_full_evidence == False` in `run_level_evidence_hops.csv`) show a higher control-correct/cue-wrong ("harmed") rate than pairs where the cue condition WAS evidence-complete, across every comparison in `analysis.canonical.results_bundle.comparisons.COMPARISONS`, pooled over all datasets/models/comparisons.

**This is an association check only, not a causal or thesis-ready claim** -- pairs are pooled across comparisons/datasets for this top-line number; see `evidence_incomplete_vs_answer_harm.csv` for the full per-comparison, per-model, per-dataset breakdown (each row also carries its own McNemar p-value) before citing anything.

## Pooled top-line comparison

Within-stratum McNemar's test asks: given this evidence-completeness state, is the cue's effect on answer correctness (control-correct/cue-wrong vs control-wrong/cue-correct) distinguishable from chance? The two strata are disjoint pair sets (not paired against each other), so comparing them directly uses non-overlap of the Wilson 95% CI on the harmed rate, not a paired test.

| Cue evidence at local completion | n_pairs | control_correct_cue_wrong | control_wrong_cue_correct | harmed rate (95% CI) | McNemar p (n_discordant) |
|---|---|---|---|---|---|
| Complete | 28871 | 2099 | 2370 | 7.27% (6.98%-7.58%) | 0.0001 (n=4469) |
| Incomplete | 20077 | 1480 | 1204 | 7.37% (7.02%-7.74%) | 0.0000 (n=2684) |

The pooled harmed rate among evidence-incomplete cue completions is **higher than** the rate among evidence-complete cue completions (7.37% vs 7.27%). Their 95% Wilson CIs **overlap** -- this pooled difference is not distinguishable from noise at this level of aggregation.


Within the evidence-complete stratum, the cue's effect on paired answer correctness is itself significant by McNemar's test (p=0.0001, n_discordant=4469).


Within the evidence-incomplete stratum, the cue's effect on paired answer correctness is itself significant by McNemar's test (p=0.0000, n_discordant=2684).

## Known limitations

- Pairs are pooled across all comparisons/datasets/models for the top-line table above; the CSV has the disaggregated view, each with its own McNemar p-value (small per-cell n means most individual rows will not reach significance -- use the pooled numbers or aggregate deliberately, not a single row's p-value, for any claim).
- Answer correctness is pre-adjudication (automatic semantic label), same caveat as `qwen_llama_results_bundle`'s answer tables.
- Technical-failure and no-evidence-data pairs are excluded, not imputed (counts in the CSV's `n_excluded_*` columns).
- This does not establish that evidence loss CAUSES the answer flip -- only whether the two co-occur more than chance-pooled would suggest, and whether that co-occurrence itself looks like chance within each stratum.
