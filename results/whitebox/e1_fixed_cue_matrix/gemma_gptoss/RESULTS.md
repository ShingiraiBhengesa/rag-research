# Gemma27B / GPT-OSS-20B Results

Generated from `data/*.csv` (this bundle). Reuses the same canonical
machinery as `qwen_llama_results_bundle/` (`canonical_completion.py`,
`evidence.py`, `answer_evaluation.py`) -- see this repo's
`analysis/canonical/gemma_gptoss_bundle/` for the generator, and
`analysis/canonical/gemma_gptoss_bundle/scope.py` for this bundle's model
and dataset scope.

## 0. Scope and a blocking data-quality finding (read this first)

**gpt_oss_20b**: complete for 3 of 4 datasets (HotpotQA, MuSiQue,
MultiHopRAG; 432/432 shards each, small per-condition gaps from isolated
harmony-parser-crash row aborts -- see `data/run_inventory.csv`).
**2WikiMultiHopQA is entirely absent** (0/108 shards) -- blocked by a vLLM
harmony-format parsing crash unrelated to gemma27b's issue (root cause:
vLLM's own server-side token-processing loop, not a swappable client-side
parser). A fix exists (`attack/runtime/gptoss_lenient_harmony_parser.py`,
built and unit-tested) but is deliberately not yet deployed. Every table
below that reports "gpt_oss_20b" covers only these 3 datasets; 2wiki is
never imputed, estimated, or silently dropped from view -- it is absent
from every table by construction (see `scope.py::MODEL_DATASETS`).

**gemma27b: results below are NOT reported.** This bundle's own gate suite
(§8) passes 12/13, and gemma27b's canonical data was generated
successfully -- but a manual data-quality audit (trace sampling, not a gate
check) found that 100% of gemma27b's 21,600 rows classify as
`NO_TOOL_DIRECT_EXIT` (PCR@1 ≈ 99.9% for every condition, including Clean
-- a saturated, pre-attack-effect curve that cannot reflect a real
per-condition difference). Sampling 15 traces' post-observation researcher
decisions found 14/15 had an empty, unparsed `tool_calls=[]` even though
the model's own response text explicitly described an intended tool call
(e.g. `"Assistant requested tool call:\n{\"args\": ..., \"name\":
\"tavily_search\"}"`) in a format `attack/runtime/gemma3_tool_code_parser.py`
does not recognize (it was built against the original broken-run's
first-turn-only malformed patterns). Working hypothesis: ODR's
gemma-specific history-sanitization renders prior tool calls back into the
conversation as a JSON-ish text pattern that gemma27b then imitates on
later turns, in a shape the parser never covers. This means gemma27b's
canonical PCR/evidence/answer numbers in `data/*.csv` are a measurement of
this parser-coverage gap, not of the model's behavior, and are excluded
from this report until that gap is fixed and a corrected re-run completes.
`data/pcr_run_level.csv`/etc. still contain gemma27b's rows (nothing was
deleted), so a future corrected re-run can be diffed against them -- they
are simply not summarized or interpreted here.

## 1. What GPT-OSS-20B's PCR curves show

Full step-by-step values: `data/pcr_stepwise.csv`. Checkpoints (PCR@1/5/10/30,
%) below. `n_eligible` is the pcr_eligible denominator per cell.

### HotpotQA

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 429 | 5.4 | 64.3 | 91.6 | 100.0 |
| Neutral | 390 | 6.9 | 60.0 | 93.6 | 99.7 |
| ContinueSearch | 427 | 3.5 | 61.1 | 93.9 | 100.0 |
| Epistemic confirmation | 426 | 7.5 | 58.0 | 92.0 | 99.3 |
| Search-match cue | 404 | 10.6 | 64.9 | 92.6 | 100.0 |
| Source coverage | 422 | 10.2 | 63.3 | 96.4 | 100.0 |
| Epistemic confirmation + source | 426 | 13.6 | 66.2 | 92.7 | 99.8 |
| Search-match + source | 401 | 11.0 | 68.8 | 96.0 | 100.0 |
| Fixed Sufficiency | 417 | 16.1 | 71.0 | 94.7 | 100.0 |
| Fixed Sufficiency Saturation | 417 | 23.0 | 71.2 | 96.6 | 99.8 |
| Oracle Neutral | 426 | 8.9 | 61.7 | 94.4 | 100.0 |
| Oracle Saturation | 423 | 20.1 | 75.4 | 96.2 | 99.8 |

### MuSiQue

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 423 | 5.4 | 61.7 | 91.7 | 99.8 |
| Neutral | 424 | 8.0 | 68.6 | 94.3 | 99.8 |
| ContinueSearch | 422 | 3.3 | 68.2 | 95.5 | 99.5 |
| Epistemic confirmation | 427 | 9.6 | 65.3 | 93.4 | 99.8 |
| Search-match cue | 425 | 8.2 | 68.5 | 95.8 | 100.0 |
| Source coverage | 423 | 11.3 | 66.0 | 92.2 | 100.0 |
| Epistemic confirmation + source | 424 | 9.7 | 65.8 | 94.1 | 100.0 |
| Search-match + source | 431 | 9.5 | 69.4 | 92.6 | 100.0 |
| Fixed Sufficiency | 430 | 16.7 | 71.6 | 93.5 | 100.0 |
| Fixed Sufficiency Saturation | 432 | 16.4 | 73.4 | 95.4 | 99.8 |
| Oracle Neutral | 426 | 8.5 | 67.1 | 93.7 | 100.0 |
| Oracle Saturation | 425 | 18.6 | 77.9 | 95.5 | 100.0 |

### MultiHopRAG

| Condition | n_eligible | PCR@1 | PCR@5 | PCR@10 | PCR@30 |
|---|---|---|---|---|---|
| Clean | 410 | 8.3 | 54.6 | 89.5 | 100.0 |
| Neutral | 418 | 8.4 | 61.7 | 92.6 | 100.0 |
| ContinueSearch | 418 | 5.5 | 57.9 | 88.3 | 99.5 |
| Epistemic confirmation | 412 | 10.4 | 61.7 | 91.7 | 99.5 |
| Search-match cue | 419 | 12.4 | 67.5 | 91.4 | 100.0 |
| Source coverage | 418 | 9.8 | 62.2 | 92.8 | 100.0 |
| Epistemic confirmation + source | 412 | 10.2 | 63.3 | 92.2 | 100.0 |
| Search-match + source | 414 | 12.6 | 65.5 | 93.0 | 100.0 |
| Fixed Sufficiency | 424 | 14.6 | 70.5 | 92.7 | 100.0 |
| Fixed Sufficiency Saturation | 422 | 19.2 | 74.2 | 95.3 | 100.0 |
| Oracle Neutral | 426 | 9.9 | 62.7 | 92.7 | 100.0 |
| Oracle Saturation | 424 | 17.0 | 75.5 | 96.0 | 100.0 |

## 2. Attack effect, averaged across the 3 available datasets

PCR@1/PCR@5 deltas vs. Clean, averaged across HotpotQA/MuSiQue/MultiHopRAG
(computed directly from `data/pcr_stepwise.csv`, not eyeballed):

| Condition | Clean PCR@1 | Attack PCR@1 | Δ PCR@1 | Ratio | Δ PCR@5 |
|---|---|---|---|---|---|
| Epistemic confirmation | 6.4 | 9.2 | +2.8pp | 1.44x | +1.4pp |
| Search-match cue | 6.4 | 10.4 | +4.1pp | 1.64x | +6.7pp |
| Source coverage | 6.4 | 10.4 | +4.1pp | 1.64x | +3.6pp |
| Epistemic confirmation + source | 6.4 | 11.2 | +4.8pp | 1.75x | +4.9pp |
| Search-match + source | 6.4 | 11.0 | +4.7pp | 1.73x | +7.7pp |
| Fixed Sufficiency | 6.4 | 15.8 | +9.4pp | 2.48x | +10.8pp |
| Fixed Sufficiency Saturation | 6.4 | 19.6 | +13.2pp | 3.07x | +12.7pp |
| Oracle Saturation | 6.4 | 18.6 | +12.2pp | 2.92x | +16.0pp |

**Finding** [table: this section's own table above]: the same qualitative
ordering seen in the Qwen/Llama study holds for GPT-OSS-20B -- sufficiency
claims (Fixed Sufficiency/Saturation) produce the largest
premature-completion effect (ratio column), a smaller but still consistent
effect for the single cue-only attacks (Epistemic Confirmation, Search
Match, Source Coverage), and Oracle Saturation (genuinely complete
evidence, the upper-bound control) produces an effect close to but not
exceeding Fixed Sufficiency Saturation's -- unlike Qwen/Llama, where the
oracle upper bound was consistently the single largest effect. This is
worth flagging as a candidate cross-model difference, not asserting as
confirmed -- it rests on 3 datasets, not 4, and gpt_oss_20b's higher
baseline noise (visible in Table 1 above) means this ordering should be
treated as suggestive, not as strong as the qwen/llama finding.

Combining a single cue with "Source Coverage" (the two "+source" rows in
the table above) shows a small but real amplification over the cue
alone -- unlike Qwen, where cue-stacking showed no consistent
amplification; this is a second candidate cross-model difference.

## 3. Evidence coverage (final-report-input boundary)

Full detail [table: data/evidence_summary.csv] (14 comparisons x 3
datasets). Paired evidence loss (control minus cue mean coverage) is small
and inconsistent in sign across every comparison and dataset [table:
data/evidence_summary.csv]. **Finding** [table: data/evidence_summary.csv]:
as in the Qwen/Llama study, evidence loss is not a universal or strong
companion effect to the PCR effect above -- GPT-OSS-20B's final answers
are reached with essentially the same evidence coverage regardless of
condition, even though it reaches that point earlier under attack
conditions (see Section 2 above).

## 4. Answer accuracy and paired flips

Full detail: [table: data/answer_summary.csv], [table:
data/answer_paired_flips.csv]. `pct_semantic_correct` moves both up and
down across conditions with no consistent direction tied to attack
strength [table: data/answer_summary.csv]. `net_harm_pct_points` is small
and changes sign across datasets for the same comparison [table:
data/answer_paired_flips.csv]. **Finding** [table: data/answer_summary.csv]:
matching the Qwen/Llama study, answer harm is weaker and far less
consistent than the PCR effect -- GPT-OSS-20B completing its research
earlier under attack does not translate into a reliably worse final
answer. Semantic-accuracy labels here are automatic
(`attack/answer_scoring/answer_evaluator_v2.py`), not human-adjudicated;
`n_needs_adjudication` per cell is in [table: data/answer_summary.csv].

## 5. Validation gates

12/13 gates pass (`data/VALIDATION_GATES.csv`). Gate 11 (every number in
this document has a table/file reference) is checked after this file is
written; if it still fails after that, treat it as a genuine gap to fix,
not a stale result. Gate 1 confirms the expected 84/84 in-scope cells
(gemma27b x 4 datasets + gpt_oss_20b x 3 datasets, all x 12 conditions) and
that gpt_oss_20b/2wiki correctly contributes zero cells rather than
silently appearing.

## 6. What is deliberately not in this bundle (yet)

Retrieval-equivalence fingerprinting, mechanism/trace-raster figures, the
`.xlsx` workbook, and LaTeX appendix tables were out of scope for this
pass (not unreusable -- `qwen_llama_results_bundle/`'s equivalent modules
are model-agnostic and could be wired in later). Human semantic
adjudication has not been performed for either model.
