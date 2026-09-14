# Analysis package

Canonical, architecture-aware analysis of completed **Qwen (qwen14b)**,
**Llama (llama8b)**, and **Gemma (gemma27b)** v3 production traces (gemma27b
added 2026-09-13 when its production matrix reached 432/432 shards).
Supersedes all prior ad hoc evaluation scripts (now archived under
`archive/legacy_evaluators/`, kept on disk for provenance only, not
git-tracked -- see `attack/README.md`).

**GPT-OSS-20B is out of scope for this package by construction** (its
production matrix is still blocked, see repo-root `PUBLISHING.md`). Every
entry point takes a model list defaulting to
`config.INCLUDED_MODELS = ("qwen14b", "llama8b", "gemma27b")`, and every
file-opening function in this package calls
`config.assert_path_not_excluded(...)` first, which raises
`ExcludedModelPathError` on any path naming `gpt_oss_20b`. This package
never imports, runs, or modifies any file under `attack/runtime/` or
`attack/launch/` -- those are live production runtime code (see the
"Active-Job Lock" note in `attack/README.md`) and are only ever imported
read-only here (`attack.runtime.cues` for the frozen condition registry,
`attack.runtime.production_models` for the frozen model registry,
`attack.runtime.trace_schema` for validated pairing invariants,
`attack.answer_scoring.answer_evaluator_v2` for the validated semantic-
accuracy procedure).

**The canonical PCR/completion-route implementation lives in
`canonical/canonical_completion.py`**, not in this directory's own `pcr.py`.
`pcr.py` remains the source `cli.py`'s pipeline uses for evidence/answer-
accuracy/pairing (all still valid and unaffected), but for completion-step/
terminal-route claims specifically, `canonical/`'s output is the one to
cite -- it fixes a real bug (see `CHANGELOG.md`) that `pcr.py` does not have
fixed, and defines one unambiguous eligibility policy where `pcr.py`
computes two different, internally-documented-as-inconsistent denominators
(see `canonical/legacy_metric_inventory.md`).

## Layout

```text
analysis/
├── config.py                 # repo-relative paths, model/condition registries, exclusion guards
├── repr_parsing.py           # literal-safe extraction from Command(...)/AIMessage(...) repr strings
├── trace_schema.py           # typed records (ResearcherDecision, ResearcherEpisode, ParsedTrace)
├── trace_parser.py           # parses a raw trace.json into the typed records above
├── completion_inventory.py   # completed-shard inventory; single choke point normalizing
│                             # stale pre-2026-09-14 trace_path values (see config.py)
├── pcr.py                    # completion-time CDF, unconditional incidence, route/denominator audits
├── pairing.py                # strict shared-prefix pairing audit
├── evidence.py                # four-boundary evidence-coverage analysis
├── answer_evaluation.py       # final-answer scoring
├── statistics.py               # paired McNemar/bootstrap tests, BH correction
├── cli.py                       # single reproducible entry point (all 6 stages)
├── final_paper_evaluation/      # mechanism/recovery/termination-route/paper-table generation
├── canonical/                   # canonical PCR implementation + full audit trail (see below)
└── tests/                        # 41 tests
```

`canonical/` (formerly `canonical_completion_v1/`) contains:
`canonical_completion.py` (the metric itself), `test_canonical_completion.py`
(20 tests), `legacy_metric_inventory.md`, `canonical_vs_previous_comparison.md`,
`gemma27b_analysis/` (the full Gemma-3-27B-IT tool-calling writeup),
`final_gap_audit/` (upstream-commit recovery, dataset-sampling reproduction,
direct-exit content audit, thesis/citation audit), and `reorg_prep/` (the
scripts that performed the 2026-09-13/14 reorganizations, kept as a record
of exactly what moved where).

## Key empirical facts this package's design depends on

(Established by direct inspection of real completed traces before writing
any aggregation logic -- see docstrings in `trace_parser.py` and
`evidence.py` for the full reasoning and verification method.)

1. `trace["events"]` interleaves two record kinds: `odr_event` (raw
   LangGraph callbacks) and `benchmark_search` (search-coverage snapshots).
   The `benchmark_search` records' own `step` field is *not* chronological
   -- it is a counter appended after all `odr_event` steps. True search
   order is recovered by zipping against `tavily_search` `on_tool_end`
   events.
2. `trace["model"]` is the model client config dict, not the short alias;
   the alias is recovered from the trace path.
3. A `researcher` node's terminal decision (`ResearchComplete` tool call vs.
   empty `tool_calls`) is stored as a Python `repr()` string, not JSON.
4. Llama sometimes serializes the search tool's `queries` argument as a
   stringified Python list, which fails the tool's schema validation
   (`on_tool_error`) -- a genuine, systematic technical failure distinct
   from the researcher choosing not to search.
5. Not every Clean/Neutral/Attack triple for the same `example_id` shares
   an identical first-search branch point (no forced replay anchor is
   engaged for most example/condition combinations on this dataset) --
   strict causal pairing succeeds for a minority of comparisons and must be
   audited per-pair (`pairing.py`), never assumed.
6. **(discovered 2026-09-13; fixed directly in this package's own
   `trace_schema.py`/`trace_parser.py`)**: ODR's `force_benchmark_search`
   safety net executes a search whose synthetic tool-call is logged on the
   *following* `researcher_tools` on_chain_end event, never on the
   researcher decision's own on_chain_end event -- naive branch-point
   detection misses this entirely. This is not a qwen/llama edge case:
   `gemma27b` hits it on 100% of its rows (its tool-calling does not
   function at all in this serving configuration -- see
   `canonical/gemma27b_analysis/gemma27b_results_synthesis.md`), and it
   materially affects `llama8b` too (~5% of rows). Any output generated
   before 2026-09-13 (including this package's own `pcr.py` outputs under
   `v3_final_completed_qwen_llama_20260910/`) predates the fix and should
   be treated as superseded for completion/route-related numbers
   specifically.
7. **(2026-09-14, top-level reorg)**: every `trace_path` value stored
   inside a production shard row written before this date still contains
   the OLD, now-nonexistent path prefix
   (`whitebox_paper_results/e2e_full_system_v3/production/`) as literal
   string data -- moving `results/` does not retroactively rewrite bytes
   already persisted in existing JSONL rows. `completion_inventory.py`'s
   `_load_shard_rows` normalizes every stored `trace_path` through
   `config.normalize_stored_trace_path()` before it ever reaches a
   `RowRecord` -- every downstream consumer in this package reads through
   `RowRecord.trace_path`, so this one fix covers the whole package.
   Verified against the full 64,800-row inventory: 64,147/64,147 `ok` rows
   resolve to an existing trace file. The two legacy evaluators now archived
   under `archive/legacy_evaluators/` (`qwen14b_architecture_mechanism_audit.py`,
   `generate_pcr_cdf_tables.py`) read raw rows independently and have their
   own local copy of the same normalization.

## Reproduction

```bash
python -m analysis.cli \
    --production-root results \
    --output-root results/evaluation/v3_final_completed_qwen_llama_gemma_20260913 \
    --stage all
```

`v3_final_completed_qwen_llama_gemma_20260913/` (post-fix, 3 models) is the
current output directory. `v3_final_completed_qwen_llama_20260910/` (pre-fix,
2 models) is preserved as a historical, now-superseded artifact -- do not
cite its completion/route numbers going forward (see fact 6 above); its
evidence/answer-accuracy numbers were not affected by that fix and remain
valid as a 2-model snapshot, but the 3-model directory supersedes it
entirely for new work.

Run from the `whitebox` environment (see repo-root README.md's "Environment
setup" and `requirements-whitebox.txt` -- needs `numpy` for `attack.metrics`,
`datasets`/`transformers` for `attack.runtime.production_manifest`). Each
stage can also be run independently via
`--stage {inventory,pcr,pairing,evidence,answers,statistics}`.

## Tests

```bash
python -m unittest discover -s analysis/tests -p "test_*.py"
python -m unittest analysis.canonical.test_canonical_completion
```

`pytest` is not installed in the `whitebox` conda environment used for this
codebase (it is present in the separate `odr` environment); tests are
written against the standard library `unittest` so they run in either.
Trace-fixture tests (`test_trace_parser.py`) skip automatically when the
raw production traces they reference aren't present on disk (they are not
published) -- `test_repr_parsing.py`, `test_config_guards.py`, and
`test_pcr.py` have no such dependency and always run.
