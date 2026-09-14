# No-tool direct-exit content audit report (Phase 3)

Route (`NO_TOOL_DIRECT_EXIT`) and content category are two different
variables throughout this report — the route definition itself is
unchanged: *"the researcher produced a response without requesting another
tool, causing the local researcher episode to route to
`compress_research`."* Nothing here renames or reinterprets that route; this
audit only inspects what the actual response text contained.

**Scope**: all 10,003 `NO_TOOL_DIRECT_EXIT` rows in
`canonical_run_level_completion.csv` (qwen14b + llama8b, the only two
`INCLUDED_COMPLETE` models). Content extracted directly from each row's raw
trace file, read-only, via the same literal-safe `repr_parsing` helper the
existing analysis package already uses for tool-call extraction (applied
here to the `content=` kwarg instead of `tool_calls=`).

## Headline numbers (deterministic + preliminary combined — see caveat below)

| Category | n | % of 10,003 |
|---|---|---|
| `SUBSTANTIVE_RESEARCH_SUMMARY` (preliminary) | 6,443 | 64.4% |
| `SUBSTANTIVE_DIRECT_ANSWER` (preliminary) | 3,119 | 31.2% |
| `MALFORMED_TOOL_ATTEMPT` (deterministic) | 247 | 2.5% |
| `REFUSAL` (deterministic) | 123 | 1.2% |
| `EMPTY_OR_WHITESPACE` (deterministic) | 31 | 0.3% |
| `UNCLEAR` (mixed) | 22 | 0.2% |
| `PROCESS_COMMENTARY` (deterministic) | 18 | 0.2% |
| `ERROR_MESSAGE` (deterministic) | 0 | 0.0% |

**Critical caveat, stated per the task's explicit instruction not to
overclaim**: only 4.2% of rows (423/10,003) were classified by a
deterministic structural rule alone (empty text, known error strings,
refusal phrases, malformed tool syntax, or a short process-only match).
**The remaining 95.8% (9,580 rows) — including every
`SUBSTANTIVE_DIRECT_ANSWER`/`SUBSTANTIVE_RESEARCH_SUMMARY` label above — are
PRELIMINARY, length-based heuristic guesses (`needs_human_review=True` in
`direct_exit_content_audit.csv`), not confirmed classifications.** A
reproducible, stratified, seed-fixed 552-row sample of exactly these
uncertain rows is exported in `direct_exit_human_review_sample.csv` with a
rubric in `direct_exit_adjudication_guide.md` for a human to adjudicate.
The headline table above must be read as "this audit's best preliminary
estimate," not as a confirmed result, until that adjudication happens.

## By model

| Model | n | % `SUBSTANTIVE_DIRECT_ANSWER` (prelim.) | % `SUBSTANTIVE_RESEARCH_SUMMARY` (prelim.) | % `MALFORMED_TOOL_ATTEMPT` |
|---|---|---|---|---|
| `qwen14b` | 8,192 | 30.1% | 68.2% | 0.07% |
| `llama8b` | 1,811 | 36.0% | 47.2% | **13.3%** |

## By dataset

| Dataset | n | % Direct Answer (prelim.) | % Research Summary (prelim.) |
|---|---|---|---|
| HotpotQA | 2,600 | 29.3% | 66.7% |
| 2WikiMultiHopQA | 2,304 | 55.8% | 39.8% |
| MuSiQue | 2,841 | 26.6% | 68.3% |
| MultiHopRAG | 2,258 | 14.0% | **82.2%** |

## By completion step

| Step bucket | n | % Direct Answer (prelim.) | % Research Summary (prelim.) | % Malformed tool attempt |
|---|---|---|---|---|
| Step 1 (immediate) | 2,654 | 26.9% | 64.8% | 8.2% |
| Steps 2-5 | 5,630 | 35.6% | 62.3% | 0.18% |
| Steps 6+ | 1,715 | 23.2% | 70.7% | 1.17% |

## Explicit answers to the 7 required questions

**1. What proportion of `NO_TOOL_DIRECT_EXIT` rows contain a substantive
direct answer?** Preliminarily, 31.2% (3,119/10,003) — but this figure is
NOT human-confirmed; treat it as an upper-bound-ish estimate pending
adjudication of the 552-row sample.

**2. What proportion contain a research summary rather than a concise
answer?** Preliminarily, 64.4% (6,443/10,003) — MORE than twice the
"direct answer" rate, in every model, dataset, and step-bucket breakdown
computed. This is the single most robust finding in this phase: **research
summaries, not concise direct answers, are the more common preliminary
content type at every level of aggregation.**

**3. How many are empty, refusals, errors, malformed tool attempts, or
unclear?** These five categories are the ones deterministically classified
(423 rows total, 4.2%): 31 empty (0.3%), 123 refusals (1.2%), 0 error
messages (0.0%), 247 malformed tool attempts (2.5%), 22 unclear (0.2%).

**4. Does Qwen use this route differently from Llama?** **Yes, dramatically,
in two independent ways.** First, in raw route frequency: Qwen has 8,192
`NO_TOOL_DIRECT_EXIT` rows vs. Llama's 1,811 — 4.5x more in absolute terms
(from `canonical_route_decomposition.csv`, already established before this
content audit). Second, in content: **13.3% of Llama's `NO_TOOL_DIRECT_EXIT`
responses contain a `MALFORMED_TOOL_ATTEMPT` pattern vs. only 0.07% of
Qwen's** — nearly 200x higher. At step 1 specifically, this reaches **47.6%
of Llama's immediate no-tool exits** (217/456, from the per-model,
per-step-bucket breakdown). This is strong, deterministic (not preliminary)
evidence that a large fraction of Llama's `NO_TOOL_DIRECT_EXIT` classifications
at step 1 are not the model choosing to answer directly at all — they are
**failed tool-call attempts** (consistent with this project's previously-
documented "Llama stringified-list tool-arg serialization bug"), where the
route classification (`NO_TOOL_DIRECT_EXIT`, defined purely by an empty
`tool_calls` list) is technically correct per its own definition, but the
underlying cause is a parsing/formatting failure, not a genuine "the
researcher decided to answer now" event.

**5. Are immediate step-1 exits different from later no-tool exits?**
**Yes.** Step-1 exits have the highest malformed-tool-attempt rate (8.2%
overall, driven almost entirely by Llama's 47.6% step-1 rate) and the
lowest research-summary share among the three step buckets (64.8% vs. 70.7%
at step 6+). Later exits (steps 6+) skew more toward research-summary-style
content and away from malformed attempts, plausibly because by that point
the researcher has accumulated more findings to narrate.

**6. Is it defensible to write that no-tool exits are "usually direct
answers"?** **No — not under any reading of this data.** Even using the
most generous preliminary (unconfirmed) heuristic classification, direct
answers are a MINORITY (31.2% overall, never exceeding 55.8% even in the
best individual dataset breakdown, 2WikiMultiHopQA) — the plurality/majority
content type is a longer research summary, not a concise direct answer, in
every single breakdown computed (overall, by model, by dataset, by
condition, by step bucket). Additionally, for Llama specifically, a
substantial minority of step-1 no-tool exits are demonstrably failed tool
calls, not answers of any kind.

**7. What exact thesis wording is supported?** Wording of this shape is
supported: *"A no-tool exit ends the local researcher episode without a
further search call, but this behavioral definition does not by itself
indicate the response's content: in a stratified content audit of all
10,003 such exits from the two fully-completed models, only a preliminary
31% resembled a concise direct answer, while a preliminary 64% more closely
resembled an extended research summary; for Llama-3.1-8B specifically, a
substantial fraction of step-1 no-tool exits (48%) were attempted but
malformed tool calls rather than answers of any kind. These preliminary
content labels await human adjudication (Appendix/Section X.Y) and should
not be treated as fully confirmed, but they clearly do not support
characterizing no-tool exits as 'usually direct answers.'"* Do NOT write
any version of "no-tool exits are usually direct answers" or "most no-tool
exits contain the final answer" without the human-adjudication step
completed and cited.

## A note on the `downstream_final_report_agrees_prefix_match` column

Only 1.44% of rows show the direct-exit response's first 120 characters
appearing verbatim in the eventual `final_report`. This is a WEAK,
literal-prefix-matching diagnostic — the `final_report_generation` node
substantially rewrites/expands findings into report format (per
`chapter3_final_source_of_truth.md` §1, its prompt explicitly asks for a
different "Answer: ..." first-line format) — so a low verbatim-match rate
does NOT mean the final answer disagrees in substance with the direct-exit
response; it only means the text was not copied verbatim. This column is
retained as a transparency measure, not as evidence of agreement/disagreement,
and answer containment is never used as a correctness proxy anywhere in
this audit (per the task's explicit instruction).
