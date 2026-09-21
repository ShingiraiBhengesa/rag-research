# Gemma-3-27B-IT: full results analysis (SUPERSEDED — see notice below)

> **⚠️ 2026-09-14 CORRECTION — every number and conclusion below is invalid.**
> This entire document analyzes gemma27b's *first* production run
> (432/432 shards, 2026-09-13), which was later found to have been served
> with a misconfigured vLLM tool-call parser (`functiongemma`, documented
> as being for a completely different, unrelated model,
> `google/functiongemma-270m-it` — it never recognized
> `google/gemma-3-27b-it`'s actual tool calls). That run's "100%
> NO_TOOL_DIRECT_EXIT" result — and everything below built on top of it —
> measures the parser bug, not gemma27b's real behavior. The bug is fixed
> (`attack/runtime/gemma3_tool_code_parser.py`, confirmed working live
> against a real vLLM+gemma-3-27b-it endpoint) and gemma27b's production
> arrays were resubmitted 2026-09-14 (jobs 151778-151781). See
> `../../CHANGELOG.md`'s 2026-09-14 entry for the full root-cause writeup
> and `PUBLISHING.md` for current run status. The old data is quarantined
> under
> `archive/legacy_results/gemma27b_broken_functiongemma_parser_run_20260913/`,
> not deleted.
>
> This document is kept, unmodified below this notice, as an honest record
> of the investigation that led to finding the parser bug — the
> methodological lesson in its last section (a `92.3%` "needs human
> review" audit result was itself a sign of a missed deterministic
> pattern) remains valid and worth reading. Its substantive claims about
> gemma27b's behavior (route decomposition, PCR curves, evidence coverage,
> answer accuracy, "attack has no effect") do **not** hold and must be
> re-derived from the fresh run before being cited anywhere.

---

Mirrors the qwen14b/llama8b analysis already produced, now extended to
gemma27b (production matrix completed 2026-09-13, 432/432 shards). All
numbers below use the corrected parser (see `../CHANGELOG.md`) — the fix
that was required to make gemma's numbers meaningful at all.

## The headline mechanism: Gemma's tool-calling does not function in this pipeline

Gemma-3-27B-IT, served via vLLM with the `functiongemma` tool-call parser,
writes its intended tool invocation as **plain text** inside its response —
a literal ` ```tool_code\ntavily_search(query="...")``` ` block — rather
than populating the structured `tool_calls` field the graph actually reads.
Confirmed directly from raw trace inspection (see `CHANGELOG.md` for the
exact excerpt). Every one of gemma's 21,600 rows exhibits this pattern.

**Consequence for the graph:** ODR's own `force_benchmark_search` safety net
intercepts this on the very first researcher decision (its tool-call list is
genuinely empty) and executes exactly one search on the model's behalf. The
model's *second* decision — now with real search results in context — also
fails to produce a structured tool call, so the graph routes directly to
`compress_research` (the normal "no tool call → done" path, since the safety
net only fires once per episode, before any search has happened). The net
effect: **every single gemma episode terminates at T_i = 1 (the first
possible post-observation step), via `NO_TOOL_DIRECT_EXIT`, 100% of the
time, in every condition, every dataset.**

## Route decomposition (21,600 rows, all PCR-eligible)

| Route | n | % |
|---|---|---|
| `NO_TOOL_DIRECT_EXIT` | 21,600 | 100.0% |
| `RESEARCH_COMPLETE` | 0 | 0.0% |
| `MAX_BUDGET_EXIT` | 0 | 0.0% |
| `TOOL_OR_PARSER_FAILURE` | 0 | 0.0% |
| `TRACE_LOGGING_FAILURE` | 0 | 0.0% |
| `EXECUTION_FAILURE` | 0 | 0.0% |

PCR@1 = PCR@2 = ... = PCR@30 = **100.0%** for every model×dataset×condition
cell. There is no "curve" to speak of — completion is instantaneous and
universal.

## Compare to qwen14b / llama8b (corrected numbers)

| Model | N eligible | PCR@1 | PCR@5 | % RESEARCH_COMPLETE | % NO_TOOL_DIRECT_EXIT |
|---|---|---|---|---|---|
| qwen14b | 21,418 | 30.3% | 89.7% | 61.7% (of total) | 37.9% |
| llama8b | 20,753 | 2.2% | 82.7% | 87.6% (of total) | 8.4% |
| gemma27b | 21,600 | **100.0%** | 100.0% | 0.0% | **100.0%** |

Gemma is not "between" qwen and llama on some spectrum of eagerness to
stop — it is categorically different: qwen and llama both make a real
stop/continue decision through structured tool calls (qwen leans toward
`ResearchComplete`, llama toward organic no-tool exits after real search
activity); gemma never gets to make that decision at all, because its
decisions never register as tool calls in the first place.

## Attack has no measurable effect on gemma's completion timing — and cannot, structurally

`paired_checkpoint_stats.csv` (from the extended `cli.py --stage all` run)
shows **every single clean-vs-attack paired comparison for gemma27b at
100.0% vs. 100.0%, `pct_point_diff = 0.0`, McNemar p = 1.0`** — for every
dataset, every condition pair, every checkpoint. This must NOT be reported
as "gemma is robust to the premature-completion attack." It is reported
instead as: **the attack's causal channel (a cue nudging a continue-vs-stop
decision) has no opportunity to act, because gemma is not making that
decision in the first place** — its episode always ends at the earliest
possible point regardless of what text (if any) was appended to the one
search result it receives. A null result here reflects an absent mechanism,
not a resistant one.

## Evidence coverage: frozen, and lower than either other model

Coverage fraction is **identical across every measurement boundary** for
gemma27b — injection, episode-0-raw-terminal, final-report-input, and
whole-run-cumulative-telemetry all report the same 0.701 mean coverage
(49.7% full-hop coverage). This is the direct consequence of the mechanism
above: since gemma never searches again and the supervisor never
redelegates (a local episode ending via `NO_TOOL_DIRECT_EXIT` reads to the
supervisor as "this line of research is done," not "this line of research
failed"), there is no later point in the pipeline where more evidence could
possibly be added.

| Model | Mean coverage @ injection | @ episode-0 terminal | @ final-report input | Full-hop % @ final-report input |
|---|---|---|---|---|
| qwen14b | 0.758 | 0.822 | **0.930** | 85.0% |
| llama8b | 0.773 | 0.790 | 0.809 | 65.2% |
| gemma27b | 0.701 | 0.701 | 0.701 (unchanged) | 49.7% |

qwen14b recovers substantially between injection and final-report-input
(+0.17, presumably via supervisor redelegation); llama8b recovers modestly
(+0.04); gemma27b recovers **nothing**, ever, by construction.

## Answer accuracy: does NOT track evidence coverage in the expected direction

| Model | % valid answer | % semantically correct | Full-hop coverage @ final report |
|---|---|---|---|
| qwen14b | 99.8% | **35.4%** | 85.0% |
| llama8b | 99.7% | 24.7% | 65.2% |
| gemma27b | 99.6% | **27.9%** | **49.7%** |

Gemma has the *worst* evidence coverage of the three models (49.7% full-hop,
vs. llama's 65.2%) but *better* semantic accuracy than llama (27.9% vs.
24.7%). This is a genuinely interesting, non-obvious finding: coverage is
not the only thing determining answer accuracy — model-specific answer-
generation quality/robustness matters too, and llama's additional evidence
(from its own, real, if occasionally malformed, tool-use activity) does not
translate into a proportional accuracy advantage over gemma's frozen,
single-search evidence base. This should be reported as an observation, not
over-interpreted as a causal claim about "evidence doesn't matter" — no
causal mediation analysis was performed here.

## What gemma's forced-search response actually says

`gemma27b_direct_exit_content_audit.csv` — full content audit of all 21,600
responses. The first pass used the same deterministic patterns already
validated against qwen/llama, which did not know about Gemma's specific
failure syntax and left 92.3% of rows as unconfirmed preliminary guesses.
After adding one targeted, evidence-based deterministic rule (a literal
`` ```tool_code `` / `tool_code` marker — Gemma-3-27B-IT's actual, confirmed
failure syntax, documented in `CHANGELOG.md`), the picture resolves almost
completely:

| Category | n | % |
|---|---|---|
| `MALFORMED_TOOL_ATTEMPT` (deterministic) | 21,357 | **98.9%** |
| `SUBSTANTIVE_RESEARCH_SUMMARY` (preliminary) | 164 | 0.8% |
| `SUBSTANTIVE_DIRECT_ANSWER` (preliminary) | 79 | 0.4% |

Only 243 rows (1.1%) still need human review; the rest are deterministically
confirmed. **This is the central, load-bearing finding of the gemma
analysis**: essentially every `NO_TOOL_DIRECT_EXIT` response from gemma27b
is not a refusal, not an error, and not a deliberate "I'm done, here is my
answer" — it is a **planning narrative that ends in an attempted tool call
written as markdown-fenced text instead of a real structured call**. One
representative example (verbatim, from the trace that first exposed this
mechanism):

> "Okay, I understand the task. I need to find the name of the independent
> candidate in the 2010 Maine gubernatorial election who outperformed Libby
> Mitchell (the Democratic candidate). I will prioritize official election
> results and reputable news sources. Here's my plan: 1. **Initial Search:**
> ... Let's start with the initial search. ```tool_code
> tavily_search(query="2010 Maine gubernatorial election results")```"

**Methodological note, worth stating plainly**: the first-pass 92.3%
"needs human review" figure was itself a measurement artifact — the
deterministic rule set had been validated only against qwen14b/llama8b
failure syntax, not gemma27b's. This is a useful caution for anyone
reusing this content-audit methodology on a new model: a "most responses
are ambiguous" result should prompt inspecting a raw sample for a missed
deterministic pattern before accepting it, not just proceeding to human
review at scale.

## Recommended framing for the paper

Gemma-3-27B-IT should not be reported as "immune to the attack" or folded
into the same premature-completion narrative as qwen/llama. The correct,
defensible framing is: **this model's tool-calling integration does not
function in the evaluated vLLM serving configuration, which structurally
prevents the studied attack mechanism from having any opportunity to
operate.** This is itself worth reporting — it is a real finding about
model/serving-stack compatibility with structured tool use, directly
relevant to anyone deploying Gemma-3-27B-IT as a tool-using agent — but it
answers a different question than "does the cue change retrieve-vs-finish
behavior," and conflating the two would misrepresent both results.
