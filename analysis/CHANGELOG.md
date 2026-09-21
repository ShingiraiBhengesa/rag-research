# Analysis package changelog

## 2026-09-14 — CORRECTION: gemma27b's "tool-calling doesn't function" finding was a parser misconfiguration bug, not a model limitation

**This corrects the final conclusion of the 2026-09-13 entry below.** That
entry's technical fix (the forced-search branch-point gap in
`trace_schema.py`/`trace_parser.py`) is still correct and unaffected. Its
closing claim -- that Gemma-3-27B-IT's tool-calling "does not function in
this serving configuration" as a fact about the model -- was wrong. It was a
fact about a misconfigured vLLM flag.

**Root cause**: `attack/runtime/production_models.py` configured
`--tool-call-parser functiongemma` for `google/gemma-3-27b-it`. That
parser's own module docstring says it is for `google/functiongemma-270m-it`
-- a different, separately fine-tuned, unrelated model with an entirely
different call format (`<start_function_call>call:...`). It never matches
`gemma-3-27b-it`'s actual output, so vLLM never populated a structured
`tool_calls` field for any gemma27b generation in the entire 432-shard run,
regardless of what the model actually wrote. No built-in vLLM parser
supports Gemma-3's real format (a ` ```tool_code``` `-fenced block);
upstream feature requests for it were closed "not planned"
(vllm-project/vllm#15403, #16482).

**Fix**: a custom vLLM tool-parser plugin,
`attack/runtime/gemma3_tool_code_parser.py`, registered via
`--tool-parser-plugin`. Recognizes gemma27b's two observed malformed-call
sub-formats (a Python-call expression and a JSON object) and normalizes
argument names/shapes to the real tool schemas, since Gemma's calls don't
match those exactly either (e.g. it wrote `text=` or even `research_topic=`
-- ConductResearch's parameter name -- when calling `think_tool`, whose real
parameter is `reflection`). Confirmed working live against a real
vLLM+gemma-3-27b-it endpoint after fixing a second bug found only through
that live test: this pipeline runs ODR under LangGraph event streaming,
which makes vLLM call the parser's *streaming* extraction path, not the
non-streaming one unit tests alone exercised.

**Data impact**: the entire 432-shard gemma27b production run (and
everything derived from it -- `all_models_20260913/`,
`gemma27b_direct_exit_content_audit.csv`, the
`v3_final_completed_qwen_llama_gemma_20260913` evaluation output) is
invalid as a measurement of gemma27b's behavior and is quarantined under
`archive/legacy_results/gemma27b_broken_functiongemma_parser_run_20260913/`
(kept, not deleted). `gemma27b` moved back out of `INCLUDED_MODELS` into
`EXCLUDED_MODEL_FAMILIES` pending a fresh run (resubmitted 2026-09-14, jobs
151778-151781). See `analysis/canonical/gemma27b_analysis/gemma27b_results_synthesis.md`
for the full corrected writeup and `PUBLISHING.md` for resubmission status.
Fixing and unit-testing the parser corrects the pipeline going forward; it
does not retroactively validate this already-run 432-shard batch, which
remains quarantined. No gemma27b finding should be cited until a fresh
production run under the corrected parser completes and passes the same
validation-gate bar as `qwen_llama_results_bundle/`, which itself remains
Qwen14B/Llama8B only and does not include gemma27b.

**Lesson**: a "does not function" finding about a specific model, in a
pipeline with multiple layers of configuration between the model and the
observed behavior, warrants checking every one of those layers (parser
choice, not just parser correctness) before concluding it's a fact about
the model. The 98.9%/100% figures were real and reproducible -- the mistake
was in attributing them, not in measuring them.

## 2026-09-13 — Gemma-3-27B-IT tool-calling discovery + forced-search branch-point fix

**Discovery.** Extending the completed-model analysis from `qwen14b`/`llama8b`
to include `gemma27b` (its 432/432-shard production matrix finished
2026-09-13) initially produced a nonsensical result: 100% of gemma27b's
21,600 rows classified as `NO_TOOL_DIRECT_EXIT` with 0% PCR-eligible. Root
cause, confirmed by direct trace inspection: Gemma-3-27B-IT, served via vLLM
with the `functiongemma` tool-call parser, essentially never emits a
structurally valid tool call in this pipeline — it writes its intended tool
invocation as a plain-text ` ```tool_code\ntavily_search(...)``` ` block
inside its response `content`, which the parser does not convert into a real
`tool_calls` entry. ODR's own `force_benchmark_search` safety net
(`deep_researcher.py:_forced_benchmark_local_search_messages`) then executes
a search anyway, but logs the synthetic tool call on the *following*
`researcher_tools` on_chain_end event rather than on the researcher's own
decision — a distinction the offline trace parser (`trace_parser.py`,
`trace_schema.py`), built and validated only against Qwen/Llama traces where
this safety net rarely fires, had no way to recognize. Every gemma episode's
branch-point detection therefore silently failed.

**Impact check on already-reported results.** Before changing anything,
checked whether qwen14b/llama8b were silently affected by the same blind
spot: only 3 qwen14b rows and 1 llama8b row (out of 43,200) ever hit the
"first decision has empty tool_calls with no search yet" pattern that this
bug depends on. The existing qwen/llama results are not materially affected.

**Fix** (`trace_schema.py`, `trace_parser.py`): added
`ResearcherDecision.triggered_forced_search: bool`, set by a new
`trace_parser._find_forced_search_steps()` that scans `researcher_tools`
on_chain_end events for a tool-call id starting with
`forced_benchmark_local_search` (the exact, stable marker ODR emits — see
`deep_researcher.py:874-881`) and attributes each occurrence to the
immediately preceding decision (same "episodes execute strictly
sequentially" ordering invariant `search_tool_error_count` attribution
already relies on). `ResearcherEpisode._branch_decision_index` now treats a
decision with `triggered_forced_search=True` the same as one that issued a
real search tool call.

**Verification**: all 41 pre-existing `analysis/tests/`
tests still pass unmodified (including 5 hand-verified fixture regression
tests), all 117 `attack/tests/` tests still pass, all 20
`canonical_completion_v1/test_canonical_completion.py` tests still pass, and
the fix was directly verified against the real gemma27b trace that exposed
the bug (`branch_reached` now `True`, `T_i=1`, matching manual trace
inspection exactly).

**Config change** (`config.py`): `INCLUDED_MODELS` extended from
`("qwen14b", "llama8b")` to `("qwen14b", "llama8b", "gemma27b")`;
`EXCLUDED_MODEL_FAMILIES` reduced to `("gpt_oss_20b",)` only — per the plan
already recorded in `PUBLISHING.md`'s "Deferred Cleanup After All Jobs
Finish" section. `test_config_guards.py` updated to match. `gpt_oss_20b`
remains excluded (production matrix still blocked, see `PUBLISHING.md`).

**New canonical output** (does not overwrite the original qwen+llama-only
run, preserved as a frozen historical artifact per this file's own prior
instruction not to overwrite in place):
- `whitebox_paper_results/e2e_full_system_v3/production/evaluation/v3_final_completed_qwen_llama_gemma_20260913/`
- `analysis/canonical/all_models_20260913/`

**Whether this is "just a bug" or a real finding**: both. It is a genuine
gap in the offline analysis tooling (now fixed), *and* the underlying fact
it exposed — that Gemma-3-27B-IT's native tool-calling does not function in
this serving configuration, so nearly all of its search behavior in this
study comes from a scripted fallback rather than the model's own tool-use
decisions — is itself a substantive, reportable result about this model in
this pipeline, not an artifact to quietly discard.
