# Local patches to Open Deep Research

**Upstream:** [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research),
package version 0.0.16, MIT licence (`LICENSE`, unchanged).

It was vendored unmodified in commit `043ecacf` (2026-07-07). All changes
since then are listed below. To see the exact diff against the pristine
snapshot:

```bash
git diff 043ecacf HEAD -- third_party/open_deep_research/src   # paths before 2026-10-16: open_deep_research/src
```

**Design rule:** no patch changes ODR's research logic. The graph, the
supervisor/researcher prompts, `think_tool`, `ResearchComplete`, compression
and report generation all run as upstream wrote them. The patches fall into
two groups:
* plumbing that lets ODR run against local vLLM models and a local search
  index;
* hooks that are inert unless a specific configuration key or environment
  variable is set.

## Files touched

`configuration.py`, `deep_researcher.py`, `prompts.py`, `state.py`, `utils.py`.
Nothing else under `src/` differs from upstream.

## 1. Local search backend (`benchmark_local`)

* **`configuration.py`:** adds `SearchAPI.BENCHMARK_LOCAL`.
* **`utils.py` (`get_all_tools`):** when `search_api == benchmark_local`, the
  search tool is `prematurestop.system.benchmark_search.benchmark_search`, a
  deterministic BM25 index over the benchmark corpus. It is exposed under the
  same tool name, `web_search`, so the model sees the same tool interface as
  upstream.
* **`deep_researcher.py`:** two forced steps guarantee that every question
  has a first observation (d1), the point where cues are injected:
  * if the supervisor's first turn delegates no research, it issues a
    `ConductResearch` call with the research brief as the topic
    (`_forced_initial_benchmark_research`);
  * if a researcher's first turn contains no search call, it issues one
    search (`_forced_benchmark_local_search_messages`).

  The forced calls have IDs prefixed `forced_`, so traces show when either
  safety net fires.

## 2. Local OpenAI-compatible model serving

* **`configuration.py`:** adds these fields:
  * `model_provider`, `model_base_url` and `model_api_key`, to reach a vLLM
    server;
  * `model_temperature` (default 0.0) and `model_seed` (default 20260820),
    for deterministic decoding;
  * `model_reasoning_effort`, used by gpt-oss.
* **`deep_researcher.py` (`_model_config`):** passes these fields to every
  chat model ODR builds.

## 3. Message-history compatibility for open-weight chat templates

* **`deep_researcher.py`** (`_messages_for_model` and the helpers next to it):
  These adaptations apply only when `model_base_url` is set, i.e. a locally
  served model.
  * Gemma-3's chat template accepts only strictly alternating user/assistant
    turns, so for Gemma the history is flattened into that form.
  * Llama-3.1's template accepts only one tool call per assistant turn, so for
    Llama the history keeps one call per turn.
  * Assistant messages are cleaned up before they are sent back to vLLM.
* The message content is unchanged. Only the turn structure is adapted.

## 4. Final-report robustness

* **`prompts.py`:** the final report must start with a line
  `Answer: <concise answer>`. This makes answer extraction deterministic. The
  rest of the report instructions are unchanged, apart from renumbering.
* **`deep_researcher.py`:** final-report generation is called with
  `tool_choice="none"`, because gpt-oss-20b sometimes emitted a hallucinated
  tool call there instead of the report.
* **`deep_researcher.py`:** an empty report is retried, with progressively
  shorter findings. If retries are exhausted, the run falls back to a summary
  of the raw notes. Traces record each fallback.

## 5. Trace provenance

* **`state.py`:** adds `v3_supervisor_delegation_index`,
  `v3_supervisor_tool_call_id` and `v3_researcher_subgraph_id`.
* **`deep_researcher.py`:** passes these fields into each researcher
  subgraph, so a trace can attribute every search and decision to the
  supervisor delegation that caused it.

## 6. Replay hooks (inert unless configured)

* **`configuration.py`:** adds five `v3_replay_*` fields:
  * `v3_replay_research_brief` and `v3_replay_research_topic`;
  * `v3_replay_first_search_queries`;
  * `v3_replay_supervisor_tool_call_id` and `v3_replay_search_tool_call_id`.
* **`deep_researcher.py`:** when these are set, the run reuses a recorded
  research brief and recorded first search queries, instead of generating
  them. This was used by E2 v1.
  * Known limitation: replay skips the `think_tool` call and the supervisor's
    planning message, so a replayed run is not identical to a fresh run.
  * E2 v2 therefore does **not** use replay.

## 7. Pinned prompt date (inert unless configured)

* **`utils.py` (`get_today_str`):** if the environment variable
  `V3_PINNED_TODAY` is set, ODR uses that date in its prompts instead of
  today's date.
* E2 v2 sets it to the date of the anchor (neutral) run, so a cue arm run
  days later sees byte-identical prompts.
* E1 never set it.

## 8. Import path (2026-10-16 restructure)

* **`utils.py`:** the import of the local search tool changed from
  `attack.runtime.benchmark_search` to
  `prematurestop.system.benchmark_search`. Only the module path changed.

## Updating upstream

To rebase onto a newer ODR:
1. Re-vendor the upstream release into a clean directory.
2. Re-apply sections 1–8 by hand.
3. Run `tests/system/test_v3_infra.py` and an E1 runtime gate on one model
   before trusting any results. Upstream prompt changes alone are enough to
   change agent behaviour.
