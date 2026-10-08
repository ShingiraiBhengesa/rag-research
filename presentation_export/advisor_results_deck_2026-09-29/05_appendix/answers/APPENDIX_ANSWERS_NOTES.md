# Answers Appendix — scope note

Full (per-model, per-dataset, per-condition — not macro-averaged) `answer_summary.csv` and `answer_paired_flips.csv` for both bundles, plus each bundle's `condition_comparison_manifest.csv` (confirms which control each cue condition is paired against).

**Not included**: `answer_run_level.csv` (per-example rows). For Qwen/Llama this file is a git-LFS pointer, not materialized in this working tree; for GPT-OSS it is a real ~75MB file. Neither was necessary for this export -- the summary/paired-flip tables above already give full (model, dataset, condition) detail. The in-progress human-adjudication packet (`qwen_llama_results_bundle/adjudication/`) was also not copied here -- see `04_answers/ANSWER_ACCURACY_STATUS.md` for its status and location; it is working review material, not a results table.
