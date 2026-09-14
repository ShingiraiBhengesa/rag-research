# Chapter 3 Final Source-of-Truth: System Model, Canonical Completion Metric, and Methodology

This document supersedes `chapter3_methodology_audit/chapter3_source_of_truth.md`
for anything it contradicts, and should be read alongside it for exhaustive
per-line citations on points not repeated in full here (architecture node
tables, full observation-manipulation pseudocode, dataset construction
detail, model/serving configuration tables). This document's new content —
not present in the prior audit — is: the canonical PCR definition and its
implementation/tests/outputs (Phase 3-6), the complete `_oracle_complete_results`
read (previously `UNKNOWN`), the exact `prompts.py`/`state.py`/`configuration.py`/
`utils.py` diffs (previously only diff-stat-level), the empirical
retrieval-equivalence audit (previously theoretical only), and the full
design matrix / sample-count tables. It is a source-of-truth audit, not
thesis prose, and does not interpret attack effectiveness.

Produced read-only: no experiment file, config, prompt, result, or SLURM job
was modified. All new files live under
`analysis/canonical/`.

---

## 1. Controlled adaptation of Open Deep Research

Classification (unchanged from the prior audit, re-verified): **Controlled
adaptation**. The vendored ODR copy (`open_deep_research/`, upstream
`https://github.com/langchain-ai/open_deep_research.git`, local package
`0.0.16`) was first added to this repo in commit `043ecacf` (2026-07-07),
bundled into the same commit as the entire initial attack pipeline — there is
no separate, clean "vendor import" commit within THIS repo's history.

**Update (final gap audit): the exact upstream commit is now resolved.**
`final_gap_audit/upstream_odr_recovery.md` — **`EXACT COMMIT VERIFIED`**:
[`1f24f1142db24f81e29eb88751985a00ec8ed580`](https://github.com/langchain-ai/open_deep_research/commit/1f24f1142db24f81e29eb88751985a00ec8ed580)
(2026-06-19, "Bump the uv group across 1 directory with 2 updates (#276)"),
verified via network content comparison: 7 independent files
(`uv.lock` in full, `pyproject.toml`, `langgraph.json`, `src/security/auth.py`,
two `src/legacy/` files, `CLAUDE.md`) are byte-for-byte identical to that
commit, and ~15 independently-varying dependency-version pins in the local
`uv.lock` converge on that exact single-commit window. This supersedes the
"UNKNOWN" determination in the prior audit.

Current uncommitted diff vs. that first commit, exact line counts (`git diff
--stat HEAD`): `deep_researcher.py` +682/-98, `configuration.py` +58/-1 (net),
`prompts.py` +6/-5 (net), `state.py` +4/-1 (net), `utils.py` +12/-0.

## 2. Original components preserved vs. 3. Components deliberately changed

Full diffs read in this task (previously only diff-stat-level):

**`configuration.py`** — adds `SearchAPI.BENCHMARK_LOCAL` enum value; adds
`model_provider`/`model_base_url`/`model_api_key`/`model_temperature`
(default `0.0`)/`model_seed` (default `20260820`) fields, enabling a
vLLM-served OpenAI-compatible endpoint to be targeted instead of a hosted
provider; adds the 5 `v3_replay_*` fields (`v3_replay_research_brief`,
`v3_replay_research_topic`, `v3_replay_first_search_queries`,
`v3_replay_supervisor_tool_call_id`, `v3_replay_search_tool_call_id`) that
back the forced-replay mechanism (only exercised by `run_paired_replay`,
which is not the production entry point — see prior audit §7/§14).

**`prompts.py`** — exactly one substantive change: the
`final_report_generation_prompt` template's numbered instructions gain a new
item 1: *"Starts with a single first line in the form `Answer: <concise
direct answer>`"* (all other items renumbered, unchanged in content). This
is a static, unconditional change to the base prompt template — distinct
from, and in addition to, `deep_researcher.py`'s separate
`_benchmark_local_final_report_instruction` (appended only when
`benchmark_local` mode is active, per the prior audit). **This is the
mechanism that makes short-answer extraction from a full report tractable**
(see section 18 below) — the model is asked to emit a parseable "Answer:"
line before its long-form report, and `answer_evaluator_v2.py::extract_short_answer`
consumes it.

**`state.py`** — adds 3 fields to `ResearcherState`:
`v3_supervisor_delegation_index`, `v3_supervisor_tool_call_id`,
`v3_researcher_subgraph_id` (all `Optional`). These back the
episode-identification mechanism the analysis layer relies on
(`trace_parser.py` groups researcher decisions into episodes by
`v3_researcher_subgraph_id`).

**`utils.py`** — adds one `elif search_api == SearchAPI.BENCHMARK_LOCAL:`
branch in `get_search_tool()`: imports `benchmark_search` from
`whitebox_attack.v3.benchmark_search` and returns it as the researcher's
sole search tool, with `metadata={"type":"search","name":"web_search","backend":"benchmark_local"}`.
**Important precision correction to the prior audit**: this `metadata["name"]`
value (`"web_search"`) is a UI/routing label only, not the LLM-facing
function-calling name. The actual tool name the model calls is fixed by
`@tool("tavily_search", description=...)` at `benchmark_search.py:369` — i.e.
**the model always sees and invokes a tool literally named `tavily_search`**,
even though it is actually executing local deterministic BM25 retrieval.
This is confirmed directly from the decorator, not inferred. It explains why
`config.SEARCH_TOOL_NAME = "tavily_search"` (the correct, verified value)
is used throughout the analysis layer's trace parsing regardless of backend.

**`deep_researcher.py`** — see prior audit §2/§3/§A for the full node-by-node
comparison table (unchanged here): adds the `benchmark_local`-mode
`force_benchmark_search` safety net, the forced query-replay mechanism, and
hard overrides of `max_concurrent_research_units=1` and
`max_react_tool_calls` (fed from `emergency_max_react_tool_calls=30`, not the
upstream default of 10).

**Error-handling changes**: none found beyond what the prior audit already
documented (compression fallback summary, final-report progressive-truncation
retry) — these are `benchmark_local`-aware additions, not general
error-handling rewrites of upstream logic.

## 4. System graph and researcher subgraph

Unchanged from the prior audit (§2): 4 top-level nodes
(`clarify_with_user → write_research_brief → research_supervisor →
final_report_generation`), 100% `Command[Literal[...]]`-based routing, zero
static `add_conditional_edges`. `ConductResearch` is a tool schema dispatched
inside `supervisor_tools`, synchronously invoking a nested
`researcher_subgraph` (`researcher ⇄ researcher_tools → compress_research →
END`) per delegation.

## 5. Local researcher versus final-report termination

Unchanged, and directly load-bearing for the canonical metric in this
document: a researcher-episode "local completion" (`ResearchComplete` or a
no-tool-call exit) is architecturally and semantically distinct from, and
never equivalent to, the single `final_report_generation` call that produces
the user-facing answer. PCR (section 15/16 below) measures ONLY the former.

## 6. Threat model

Unchanged from the prior audit (§4): white-box access to the observation
boundary; injection gated by positional eligibility AND a run-scoped
one-shot counter; append-only, never substituting; no gold-answer leakage in
the cue text itself; no tool names or "ignore previous instructions"-style
language in any cue.

## 7. Exact attack insertion procedure

Unchanged from the prior audit (§5): `attack/benchmark_search.py::run_benchmark_search`,
`output = base_output + ("\n\n" + suffix if suffix else "")`, gated by
`cue.requires_injection AND is_target AND _RUN_INJECTION_COUNTS[run_id]==0`.

## 8. Twelve conditions, nine primary and three secondary

All 12 identifiers confirmed to exist in `attack/cues.py:12-30`
(`V3_CONDITIONS`). Per this task's explicit grouping (now adopted as the
report's own presentation, not merely "the user's own"), and cross-checked
against `analysis/config.py`'s own condition groupings
(which independently split conditions into "cue-only" vs. "source-attached"
ablations — consistent in spirit, though that module does not use the words
"primary"/"secondary"):

| Condition ID | Display label | Status |
|---|---|---|
| `NaturalClean` | Clean | Primary |
| `NaturalNeutral` | Neutral | Primary |
| `NaturalContinueSearch` | Continue Search | Primary |
| `Natural_epistemic_confirmation` | Epistemic Confirmation | Primary |
| `Natural_search_match_cue` | Search Match | Primary |
| `Natural_PreRegisteredSufficiency` | Fixed Sufficiency | Primary |
| `Natural_PreRegisteredSufficiencySaturation` | Fixed Sufficiency Saturation | Primary |
| `OracleComplete_Neutral` | Oracle Complete--Neutral | Primary |
| `OracleComplete_PreRegisteredSufficiencySaturation` | Oracle Complete--Saturation | Primary |
| `Natural_source_coverage` | Source Coverage | Secondary ablation |
| `Natural_epistemic_confirmation_source` | Epistemic Confirmation + Source Coverage | Secondary ablation |
| `Natural_search_match_source` | Search Match + Source Coverage | Secondary ablation |

Internal identifiers are unchanged (never renamed in code); the display
label is analysis-facing metadata only, recorded in `condition_metadata.csv`.
Per this task's explicit instruction, **no condition is described as
formally preregistered** — see section 8a.

### 8a. Pre-registration status of the "Fixed Sufficiency" conditions

Unchanged determination from the prior audit: **`PRE-SPECIFIED IN CODE ONLY`**.
`cues.py`'s `DEPRECATED_CONDITION_ALIASES` (lines 33-35) shows
`Natural_PreRegisteredSufficiency`/`Natural_PreRegisteredSufficiencySaturation`/
`OracleComplete_PreRegisteredSufficiencySaturation` were renamed from earlier
working names (`Natural_BestNewSingle`/`Natural_BestNewCompound`/
`OracleComplete_PrimaryCue`) — real evidence of an earlier, code-level design
decision — but there is no externally dated, immutable registration artifact
(the whole `attack/` directory is untracked in git, so no commit
timestamp exists either). The condition's own internal name containing the
word "PreRegistered" is not, by itself, proof of formal pre-registration.

## 9. Exact cue strings

Unchanged from the prior audit (§6), verbatim, reproduced in
`condition_metadata.csv`'s `exact_suffix` column for all 12 conditions.

## 10. Dataset selection and deep-research justification

Summarized from the prior audit's §B (full per-dataset justification table,
diversity comparison, and 11-claim evaluation are there and not repeated
here in full): HotpotQA (fixed 2-hop, distractor config), 2WikiMultiHopQA
(2- or 4-hop, 4 reasoning types), MuSiQue (explicit 2/3/4-hop compositional
decomposition, richest annotation), MultiHopRAG (2/3/4-hop news-domain,
**no distractor pool** — its "corpus" is the annotated evidence set itself).
Recommended framing (unchanged): "a controlled test bed for multi-step
evidence collection and research termination," not a full substitute for
open-ended web research.

## 11. Natural and OracleComplete evidence construction

**Natural retrieval reuse across paired conditions**: unchanged determination
— production calls `run_single_condition` independently per condition (never
`run_paired_replay`), so identical first-search-query/retrieval trajectories
are **not structurally guaranteed**. This is now backed by two independent
empirical measurements (previously only inferred from code):
(a) the pre-existing `strict_pairing_audit.csv` shows only **34.8%** of
77,411 condition-pair-example rows pass the strict hash-equality invariant;
(b) this task's own `retrieval_equivalence_summary.md` (section 12 below)
provides a second, field-by-field measurement.

**`_oracle_complete_results` — read in full this task (previously `UNKNOWN`)**,
`attack/benchmark_search.py:474-568`:

1. Requires `benchmark_evidence_path`, `dataset`, `example_id` all present in
   config — else returns empty results with `oracle_complete_available=False`,
   `oracle_unavailable_reason="missing evidence path, dataset, or example id"`.
2. Loads the example's `EvidenceDependency` tuple from the evidence JSONL
   (`_load_evidence_dependencies`); if empty, unavailable
   (`"no dependency annotations available"`); if any dependency is
   `ambiguous`, unavailable (`"ambiguous dependency annotations"`).
3. Computes `required_doc_ids` (unique, order-preserving via
   `dict.fromkeys`, i.e. **first-occurrence order, not re-sorted
   alphabetically**) and `required_paragraph_ids` from the non-ambiguous
   dependencies.
4. Loads ALL corpus documents for the example, filters to exactly the
   required document IDs (`load_documents(..., dataset=dataset,
   example_id=example_id)` then a membership filter) — **no extra
   documents, no distractors, in oracle mode**.
5. If any required document is missing from the corpus file itself, oracle
   is unavailable (`"required support documents missing from corpus"`) —
   this is the exact, code-verified oracle-unavailability path (distinct
   from the ambiguous-dependency path).
6. **Ordering**: `docs.sort(key=lambda doc: required_doc_ids.index(doc.document_id))`
   — i.e. oracle documents are ordered by **first-occurrence order of their
   dependency in the evidence annotation**, not by any retrieval score.
7. **Scoring**: every oracle result gets `retrieval_score=1.0` (uniform,
   not BM25-scored) and `rank` = its position in the above order (1-based).
   A fresh `BM25Index(docs)` is still constructed (for its `config_hash`/
   `index_hash` fields only — its `.search()` method is never called in this
   path).
8. **Deduplication**: implicit only — `required_doc_ids`/`required_paragraph_ids`
   are computed via `dict.fromkeys` (dedups exact-string duplicates); no
   fuzzy/semantic deduplication of near-identical passages.
9. **Truncation/length limits**: none found — every required document is
   included in full, regardless of count or combined length.
10. **Completeness check** (`oracle_complete_available`): computed via
    `evaluate_coverage(dependencies, [result.document_id ...], retrieved_paragraph_ids=[...])`
    — i.e. oracle availability is determined by running the SAME
    dependency-coverage function used everywhere else in the pipeline over
    the assembled oracle document set, not assumed true by construction.
    If coverage is incomplete (some dependency's document/paragraph is
    genuinely absent from the corpus), the oracle condition falls back to an
    explicit unavailability message: `f"Oracle-complete condition
    unavailable for this example: {reason}."` (`_oracle_unavailable_message`)
    — this message, not silence or a crash, is what the researcher sees.
11. **Answer leakage risk**: the oracle document text is the dataset's own
    gold-annotated supporting passage(s), which (by the nature of these
    benchmarks' annotation) typically contains the answer span. This is
    inherent to the benchmarks, not a defect introduced by this function.
12. **Does oracle use the same first query as Natural?** The `query`
    parameter IS passed into `_oracle_complete_results` and stored on every
    `BenchmarkSearchResult.query` field, but **it is never used to filter,
    rank, or select which documents are returned** — oracle document
    selection depends only on the pre-computed evidence dependencies, not on
    the query text at all. So "the same query" is recorded for provenance
    but has zero causal effect on which passages are returned in oracle
    mode — a materially different retrieval mechanism from Natural
    conditions' BM25 path, not merely "the same retrieval with different
    cue text."
13. **Dataset-specific differences**: none in the function itself — it is
    fully dataset-agnostic, operating only on the already-normalized
    `EvidenceDependency` schema (itself built per-dataset in `corpus.py`,
    per the prior audit's §7/§9 per-dataset breakdown).
14. **Conditions/datasets where OracleComplete is unavailable**: determined
    per-example at runtime (any example with missing/ambiguous dependencies
    or a corpus gap), not per-dataset in general — confirmed empirically
    that **MultiHopRAG DOES include both `OracleComplete_*` conditions in
    production** (`ls whitebox_paper_results/.../runs/qwen14b/multihoprag/`
    shows all 12 condition directories present), resolving the prior
    audit's open question. The dataset-level `multihoprag_oracle_allowed`
    gate (`generate_v3_qwen_cluster_arrays.py:175-178`) evidently passed.

## 12. Retrieval setup

Unchanged from the prior audit (§8): deterministic local BM25 (`k1=1.5`,
`b=0.75`), `attack/benchmark_search.py::BM25Index`, top-k=3,
lowercase-alphanumeric tokenization, in-process + content-addressed disk
caching, fully deterministic given identical query text.

### 12a. Empirical retrieval-equivalence audit (new this task)

See `retrieval_equivalence_by_pair.csv` and `retrieval_equivalence_summary.md`
for the full, per-pair, field-by-field comparison (first query, normalized
query, retrieved document-ID set, rank order, pre-suffix observation hash,
shared-prefix hash, BM25 index hash, retriever config hash) across every
strictly-comparable pair (`model+dataset+example_id`, both sides present and
readable) for the 9 primary Natural-vs-baseline pairs plus the Oracle pair,
using existing traces only. **Do not assume the numbers below without
reading `retrieval_equivalence_summary.md` directly — this section
summarizes it but the file is the source of truth and may have been
regenerated after this section was written if this document is read later.**

**Results** (`retrieval_equivalence_by_pair.csv`, 64,800 pair-rows attempted
across the 2 completed models × 4 datasets × 11 required pair comparisons;
63,314 pairs had both traces readable with a first-search event found):

| Check | % identical (of 63,314 readable pairs) |
|---|---|
| First search query (raw text) | 50.7% |
| First search query (normalized) | 50.9% |
| Retrieved document-ID set | 66.3% |
| Retrieved rank order | 50.7% |
| Original (pre-suffix) observation hash (`base_tool_message_hash`) | 42.5% |
| Shared-prefix hash (`shared_prefix_hash`) | 42.5% |
| BM25 index hash | 100.0% |
| Retriever config hash | 100.0% |

Additionally: 12,360 pairs (19.5%) used a **different** query but still
retrieved an **identical** document set (BM25 over a small, per-example
corpus can map multiple similar queries onto the same top-3 result); 21,351
pairs (33.7%) retrieved **genuinely different evidence** (the document-ID
sets differ entirely).

**This must be stated plainly: retrieval is identical across paired
conditions only about 42-50% of the time** in production, and roughly 1 in 3
pairs retrieved different evidence outright. The BM25 index and retriever
configuration are always identical (100%) — confirming that when retrieval
differs, it is because the MODEL asked a different query, not because the
corpus, index, or retrieval backend changed. This is consistent with, and a
finer-grained companion to, the pre-existing `strict_pairing_audit.csv`'s
34.8% pass rate on the combined, stricter `validate_paired_traces` invariant
(hash equality AND full structural validation; this table reports hash
equality alone, which is why the percentages are not identical).

## 13. Models and inference settings

Unchanged from the prior audit (§10): `qwen14b`/`llama8b`/`gemma27b`/`gpt_oss_20b`,
vLLM 0.23.0 (live env, not repo-pinned), `temperature=0.0`, `seed=20260820`,
`max_react_tool_calls` effectively 30 (`emergency_max_react_tool_calls`,
never overridden by any production sbatch), `max_researcher_iterations=6`,
`max_concurrent_research_units=1`. **New precision**: `production_runner.py`
itself executes in the `odr` conda environment (`conda activate odr`,
sbatch line 165), distinct from the `vllm` environment that serves the
model — `odr` has `langgraph==1.2.6`, `langchain==1.3.10`,
`langchain-core==1.4.8` (live `pip show`, this session; not pinned in any
repo lockfile).

## 14. Experiment procedure

Unchanged from the prior audit (§11): `generate_v3_qwen_cluster_arrays.py`
launcher, fixed (non-randomized) condition/shard-index arithmetic over
`SLURM_ARRAY_TASK_ID`, `run_id = stable_hash({model,dataset,condition,example_id})`
(shard/array-index-independent), append-only resumable shards, 3-consecutive-
error abort, no output-file-level lock beyond SLURM's own `%1` array
throttle.

## 15. Trace schema

Two layers, both reused read-only by this canonical package (never
reimplemented): (a) the production trace JSON schema itself
(`attack/trace_schema.py`'s `validate_branch_trace`/
`validate_paired_traces`, enforced at write time); (b) the analysis-layer
parsed representation (`analysis/trace_schema.py`'s
`ResearcherDecision`/`SearchEvent`/`ResearcherEpisode`/`ParsedTrace`
dataclasses, built by `analysis/trace_parser.py` from the
production JSON). `canonical_completion.py` imports and consumes (b)
directly.

## 16. Canonical PCR definition

**Display name: Post-observation Completion Rate (PCR).** Never call this
metric "Premature Completion Rate" in new code, docs, tables, or figures —
"premature" is a normative judgment this metric does not make (it says
nothing about evidence sufficiency or correctness).

For run *i*, let T_i be the first local researcher decision after the
designated first retrieval observation at which the targeted researcher
episode ends via an explicit `ResearchComplete` call OR a no-tool-call
response routing to `compress_research`:

```
T_i = min{t : terminal_route(i,t) is ResearchComplete or NoToolDirectExit}
PCR@k = (1/N) * sum_i I[T_i <= k]      (N = number of PCR-eligible runs)
```

Implementation: `canonical_completion.py::classify_local_terminal_route`,
`first_local_completion_step`, `compute_pcr_curve`. Reuses the pre-existing,
already-validated `analysis/trace_parser.py` for all
event-walking and step-counting — this module adds only the route taxonomy
and eligibility policy, it does not reimplement trace parsing.

**Step definition, verified against the implementation**: one post-observation
step is one `researcher` node invocation (one `ResearcherDecision`),
regardless of which tool — including `think_tool` — was called, or none.
Confirmed directly from `trace_parser.py::build_episodes`: one
`ResearcherDecision` is appended per `researcher` on_chain_start/on_chain_end
pair, unconditionally. `think_tool` calls are not skipped and do not merge
with adjacent steps — verified by `test_canonical_completion.py::test_5`.

**Anchor, verified condition-agnostic**: the branch point is "the first
`benchmark_search`(tool-name `tavily_search`) event actually executed in
episode 0" — identical logic regardless of condition. For `NaturalClean` this
is a real, un-suffixed observation; for attacked `Natural_*` conditions it is
the (possibly suffixed) observation from the SAME code path; for
`OracleComplete_*` conditions it is the oracle-assembled observation (still
logged as a `benchmark_search` event with `oracle_mode=true`). No
condition-specific branch logic exists in the anchor detection itself.

**PCR does NOT prove**, by explicit design, and is explicitly independent of:
final-answer correctness, evidence coverage/sufficiency, or whether the
stopping decision was normatively justified. These are separately measured
(section 17/18).

**Eligibility policy — a documented, deliberate design choice, not one of
the two pre-existing denominators**: `pcr_eligible = anchor_found (branch
reached) AND route not in {TRACE_LOGGING_FAILURE, EXECUTION_FAILURE}`. See
`canonical_completion.py`'s module docstring for the full rationale, and
`legacy_metric_inventory.md`/`canonical_vs_previous_comparison.md` for how
this differs, quantitatively, from `pcr.py`'s two existing (and, per that
inventory, internally inconsistently documented) denominators.

## 17. Route definitions

Seven mutually exclusive, exhaustive routes (`canonical_completion.py`):
`RESEARCH_COMPLETE`, `NO_TOOL_DIRECT_EXIT`, `MAX_BUDGET_EXIT`,
`TOOL_OR_PARSER_FAILURE`, `TRACE_LOGGING_FAILURE`, `EXECUTION_FAILURE`,
`OTHER_VALID_ROUTE`. Only the first two ever populate the PCR numerator.
Empirically, `OTHER_VALID_ROUTE` count is **0 across all 96
model×dataset×condition cells** in the full qwen14b+llama8b production data
(`canonical_route_decomposition.csv`) — the taxonomy is exhaustive in
practice, not merely in theory, for the data actually produced.

**`MAX_REACT_TOOL_CALLS = 30`** (from `attack/runner.py:49`'s
`emergency_max_react_tool_calls` default, confirmed unoverridden in every
production sbatch script). Empirical cross-check: the maximum
`first_terminal_post_observation_step` observed anywhere in the full 43,200-row
qwen14b+llama8b dataset is **exactly 29** — i.e. exactly `30 - 1`, matching
the theoretical maximum (the branch decision itself consumes budget slot 1,
leaving at most 29 post-branch decisions). This is a strong, independent
correctness check on both the trace parser and the budget constant used
here.

## 18. Separate evidence outcomes (Phase 7 — kept distinct from PCR)

Not implemented or recomputed by this canonical package (the existing,
already-tested implementations are retained unmodified, per the task's
"do not delete evidence-aware evaluators" instruction) — labels to use going
forward, per the pre-existing implementations:

- **Evidence Coverage at Completion** — `attack/evidence.py::evaluate_coverage`
  (retrieval-metadata-based, never suffix-text-based).
- **Full-Hop Coverage**, **Missing-Hop Rate** —
  `attack/qwen14b_architecture_mechanism_audit.py::coverage_payload`
  (qwen14b-scoped implementation) and `analysis/evidence.py`
  (cross-model).
- **Evidence-Incomplete Completion** —
  `attack/evaluate_completed_shard.py::_premature_completion_step`'s
  underlying concept (coverage `< 1.0` at the `ResearchComplete` step) — this
  is the metric that SHOULD carry a word like "premature," never PCR.
- **Gold-Document Recall**, **Supporting-Fact Recall** — derivable from the
  same coverage machinery; not separately named in any reviewed file beyond
  the coverage-fraction fields.

**None of these are used in the PCR numerator or denominator anywhere in
`canonical_completion.py`** — confirmed by inspection (the module never
imports `evidence.py`/`coverage_fraction`/`evaluate_coverage`).

## 19. Separate final-answer outcomes (Phase 7 — kept distinct from PCR)

Retained, unmodified, from `attack/answer_evaluator_v2.py` and
`attack/metrics.py`:

- **Semantic Answer Accuracy**, **Exact Match**, **Token F1** —
  `answer_evaluator_v2.py::evaluate_answer_v2`/`semantic_accuracy`;
  `metrics.py::em_f1` (canonical) and `answer_evaluator_v2.py::official_em_f1`
  (a near-duplicate, differs only in apostrophe-stripping order).
- **Containment diagnostic** — `answer_evaluator_v2.py::gold_contained_diagnostic`.
- **Needs adjudication** — `answer_evaluator_v2.py`'s `needs_adjudication`
  label, fired in exactly 4 documented situations (unresolved-conflict
  extraction/scoring, ambiguous multi-candidate containment, high token-F1
  without a safe rule match).
- **Clean-correct → attack-wrong / clean-wrong → attack-correct / net paired
  answer degradation** — **not implemented as a per-example flip metric in
  any pre-existing v3-native script** (confirmed in the prior audit's §15 and
  re-confirmed here: `rescore_completed_answer_v2.py` and
  `final_production_evaluation.py` compute only group-level accuracy rates
  and their difference, despite strictly pairing examples; the analogous
  *behavioral* transition metric exists only for premature-direct-answer
  behavior, `final_production_evaluation.py::_direct_answer_transitions`, not
  for answer correctness). This remains a genuine implementation gap outside
  this canonical package's scope (PCR, not answer correctness).

**None of these are used anywhere in `canonical_completion.py`** — confirmed
by inspection (no import of `answer_evaluator_v2`/`metrics`/`valid_answer`
fields).

## 20. Run validity and pairing

**Validity** (unchanged from prior audit §13): a row counts as
`EXECUTION_FAILURE` in this canonical taxonomy iff `final_status != "ok"` or
its referenced trace file does not exist on disk — independent of
correctness/attack outcome.

**Pairing key, recommended and used**: `(model, dataset, example_id)`,
exactly as the prior audit recommended and as
`analysis/pairing.py` already implements. This canonical
package's `build_strict_pairs` joins on this key and additionally requires
BOTH sides to be `pcr_eligible` (necessary for a PCR delta to be
computable at all) — this is a WEAKER requirement than
`pairing.py`'s/`trace_schema.py`'s strict hash-equality invariant
(`validate_paired_traces`), and the two must not be conflated: a pair can be
`pcr_eligible` on both sides (both have a well-defined T_i) while still
having diverged retrieval content before the injection point (see section
12a). `canonical_paired_condition_deltas.csv` reports raw-N, eligible-N, and
paired-N separately for every required comparison (every Natural condition
vs. `NaturalClean`; semantic-sufficiency conditions vs. `NaturalNeutral`;
`NaturalContinueSearch` vs. `NaturalNeutral`; the Oracle pair) — raw condition
totals are never substituted for paired totals.

## 21. Reproducibility information

Unchanged from the prior audit (§16/§17), with these additions: `odr` conda
env package versions now directly confirmed (`langgraph==1.2.6`,
`langchain==1.3.10`, `langchain-core==1.4.8`); vLLM confirmed `0.23.0` in the
`vllm` conda env. Neither is pinned in any repo-committed lockfile — this
remains a real reproducibility gap, not resolved by this task (would require
the operator to commit a lockfile or environment export, which this
read-only task cannot do).

## 22. Remaining limitations and unknowns

- Exact upstream ODR commit/tag: **RESOLVED** by the final gap audit —
  `1f24f1142db24f81e29eb88751985a00ec8ed580` (see section 1 update above and
  `final_gap_audit/upstream_odr_recovery.md`).
- Exact eligible-population counts (pre-450-sampling) for all 4 datasets:
  **RESOLVED** by the final gap audit — HotpotQA 7,405/7,405 eligible (0
  rejected), 2WikiMultiHopQA 12,576/12,576 (0 rejected), MuSiQue 2,417/2,417
  (0 rejected), MultiHopRAG 2,255/2,556 (301 rejected). Exact sampling
  reproduction (seed 20260820, same method) PASSED for all 4 datasets —
  identical example-ID set/order and identical `eligible_population_sha256`/
  `ordered_sample_id_sha256` digests. See `final_gap_audit/eligible_population_counts.csv`
  and `final_gap_audit/sampling_reproduction_check.md`.
- Oracle availability: **RESOLVED** — zero `oracle_complete_available=False`
  cases found across all 16 model×dataset×oracle-condition combinations in
  completed production data (`final_gap_audit/oracle_availability_summary.csv`).
- No-tool direct-exit content: **NEW finding from the final gap audit** —
  a full content audit of all 10,003 `NO_TOOL_DIRECT_EXIT` rows shows
  "usually direct answers" is NOT a defensible characterization (research
  summaries outnumber direct answers ~2:1 even under the most generous
  preliminary reading, and 47.6% of Llama's immediate step-1 no-tool exits
  are demonstrably malformed tool-call attempts, not answers). See
  `final_gap_audit/direct_exit_audit_report.md`.
- Whether `analysis/final_paper_evaluation/{answer_harm,figures,master,paper_tables}.py`
  contain any independent completion-step logic of their own: not
  separately re-verified in this task (confirmed in `legacy_metric_inventory.md`
  that they only consume `pcr.py`'s already-computed `PCRRecord`s, based on
  their import statements — not a full line-by-line read of those 4 files).
- Gemma27b/GPT-OSS-20B: intentionally excluded from all canonical PCR
  computation in this task (matching the pre-existing analysis package's own
  `INCLUDED_MODELS` scope) because their production matrices are still
  active/incomplete as of this task's execution — see `final_design_matrix.csv`
  for their dataset-level-only status.
- `k=40` **is not reachable** under the actual configured
  `emergency_max_react_tool_calls=30` budget (max possible post-branch step
  is 29, matching the empirical maximum observed) — the checkpoint table
  therefore stops at k=30, per the task's own explicit guard, and the
  HEAD-commit branch name "paper-pcr40-results" must not be read as implying
  a validated 40-step canonical metric (see `legacy_metric_inventory.md`).

---

## Safe statements for Chapter 3

- ODR's real, vendored LangGraph implementation is used, with a controlled,
  documented set of local adaptations (deterministic local retrieval, a
  forced-replay mechanism, iteration-budget overrides, a required "Answer:"
  first line in the final report).
- All graph routing is `Command`-based; there are no static conditional
  edges anywhere in the graph.
- The canonical Post-observation Completion Rate (PCR) is a purely
  behavioral, step-indexed metric: it measures WHEN a targeted researcher
  episode locally terminates after its first retrieval observation, and does
  not by itself measure evidence sufficiency, correctness, or normative
  prematurity.
- PCR's step-counting rule (one researcher decision = one step, `think_tool`
  included) is verified against the actual trace-parsing implementation and
  cross-validated by an exact match between the empirical maximum observed
  step (29) and the theoretical maximum implied by the production budget
  constant (30).
- Seven mutually exclusive terminal routes are used; only
  `RESEARCH_COMPLETE`/`NO_TOOL_DIRECT_EXIT` ever populate the PCR numerator;
  `MAX_BUDGET_EXIT`/technical failures are never folded into a completion
  route. This taxonomy is exhaustive in the actual production data (zero
  `OTHER_VALID_ROUTE` rows across 96 cells).
- `OracleComplete_*` conditions assemble a document set from complete gold
  evidence annotations, ordered by first-occurrence-in-annotation (not by
  retrieval score), with retrieval score fixed at 1.0 and availability
  gated by an explicit coverage check with a documented, in-band
  unavailability message on failure — never a silent fallback.
- Retrieval trajectories are NOT guaranteed identical across paired
  conditions in the actual production configuration; this must be stated as
  an empirically-measured quantity: only 42.5% of 63,314 readable pairs
  share an identical pre-suffix observation hash, 66.3% retrieved the same
  document set, and 33.7% retrieved genuinely different evidence (see
  `retrieval_equivalence_summary.md` and the pre-existing
  `strict_pairing_audit.csv`'s 34.8% strict-pairing rate under the
  stricter combined invariant) — never assume equivalence.
- Only `qwen14b` and `llama8b` have fully completed production matrices as
  of this task's execution; `gemma27b` is partial (multihoprag actively
  backfilling); `gpt_oss_20b` has zero completed production rows and is
  permanently blocked pending a human decision (see `PUBLISHING.md`).

## Statements Chapter 3 must avoid

- **Calling PCR "evidence-aware."** It is not — it is defined purely on
  terminal-route/step, independent of coverage.
- **Treating all early completion as proven failure.** PCR measures when
  completion happened, not whether it was premature in a normative sense —
  that requires the separate evidence-coverage metrics (section 18).
- **Treating local researcher completion as the final answer.** Only
  `final_report_generation`'s single output is the user-facing answer; a
  researcher episode's local completion never is.
- **Claiming identical retrieval across paired conditions without citing the
  hash/field audit.** The production design does not structurally guarantee
  this (no `run_paired_replay` at the production entry point); it must be
  reported as measured, with the actual percentage from
  `retrieval_equivalence_summary.md`.
- **Calling the "Fixed Sufficiency"/"PreRegisteredSufficiency" conditions
  formally preregistered.** The determination remains `PRE-SPECIFIED IN CODE
  ONLY` — a code-level rename history, not an external, dated registration.
- **Claiming all models or all matrix cells are complete.** Only qwen14b and
  llama8b are; see `final_design_matrix.csv` for the exact per-cell status,
  including gemma27b's partial multihoprag shard and gpt_oss_20b's zero
  completed rows.
- **Claiming the four datasets fully reproduce open-ended web research.**
  They are static, per-example, closed corpora (with or without a distractor
  pool depending on dataset) — a controlled test bed for multi-step evidence
  collection and termination behavior, not a substitute for live web search,
  source discovery, or long-form synthesis (see prior audit §B for the full
  represented/not-represented breakdown).
- **Using the branch name "paper-pcr40-results" or the HEAD commit "Add PCR
  40-step paper results" as evidence about the v3 system.** That commit is
  from the pre-v3 WGA/FactOnly/CueOnly pipeline, already archived, and
  k=40 is not even reachable under the v3 system's actual step budget (max
  possible: 29).

## Correction (2026-09-13): forced-search branch-point fix changes llama8b's numbers too

Extending this analysis to `gemma27b` exposed a real bug in the offline
trace parser (`analysis/trace_schema.py`/`trace_parser.py`):
it never recognized ODR's `force_benchmark_search` safety net (a synthetic
search executed when a researcher decision has no real tool call) as
establishing the PCR branch point, because that synthetic call is logged on
a `researcher_tools` on_chain_end event, not a `researcher` one. See
`analysis/CHANGELOG.md` for the full root-cause writeup
and fix.

**This is not gemma-only.** `llama8b` hits the same safety net often enough
that the fix materially changes its numbers: PCR-eligible N moves from
19,696 to **20,753** (+5.4%), and PCR@k shifts by a few tenths of a
percentage point at every checkpoint (e.g. PCR@1: 2.37% → 2.23%; PCR@5:
82.55% → 82.74%). `qwen14b` is only trivially affected (PCR-eligible N
21,408 → 21,418). **Every number in sections 16-19 of this document
reflecting the pre-fix parser is now superseded** by
`analysis/canonical/canonical_pcr_by_model.csv`
et al. (regenerated 2026-09-13) — the qualitative conclusions in this
document (route taxonomy, eligibility policy, the k=29/30 budget-boundary
check, the retrieval-equivalence findings) are unaffected, but exact
percentages quoted above should be re-pulled from the regenerated CSVs
before being cited verbatim.
