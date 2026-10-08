# Handover — 2026-09-28

Branch: `paper-pcr40-results`, HEAD `959b6315` (unchanged — **nothing from today's
session is committed yet**, see "Git state" below). Read this whole file before
touching gemma27b, gpt_oss_20b, or `analysis/canonical/canonical_completion.py`.

## TL;DR — what's running right now

5 SLURM array jobs submitted today, all `PD` (pending, priority) as of this
writing:

| Job ID | Array | Model/dataset | Tasks |
|---|---|---|---|
| 166533 | `0-107%1` | gemma27b / hotpot | 108 |
| 166534 | `0-107%1` | gemma27b / 2wiki | 108 |
| 166535 | `0-107%1` | gemma27b / musique | 108 |
| 166536 | `0-107%1` | gemma27b / multihoprag | 108 |
| 166537 | `0-107%1` | gpt_oss_20b / 2wiki | 108 |

First command to run in the next session: `squeue -u $USER`. Then check shard
progress:
```bash
for d in hotpot 2wiki musique multihoprag; do
  echo "gemma27b/$d: $(find results/runs/gemma27b/$d -name '*.jsonl' 2>/dev/null | wc -l)/108"
done
echo "gpt_oss_20b/2wiki: $(find results/runs/gpt_oss_20b/2wiki -name '*.jsonl' 2>/dev/null | wc -l)/108"
```
Each array is throttled `%1` (one task at a time) with a 30h wall-time ceiling
per task — this is a genuinely multi-day rerun, not something to expect
finished soon. `sacct -j <jobid> --format="Start,End,State,Elapsed"` gives
per-task timing once tasks start completing.

## Why these jobs exist (root causes, both newly fixed today)

### 1. gemma27b: imitated-tool-call bug (the big one)

Discovered by trace-sampling: gemma27b's `NO_TOOL_DIRECT_EXIT` rows (its
"stopped without calling a tool" route) were 93% (56/60 sampled) a specific
artifact, not genuine model behavior. Root cause, confirmed by reading
`open_deep_research/src/open_deep_research/deep_researcher.py`:

- `_requires_plain_alternating_history()` gates a gemma-only code path
  (`"gemma" in model_text`) that flattens all messages to plain
  user/assistant turns (gemma's server template rejects OpenAI tool-role
  messages).
- The old `_plain_turn_from_message()` rendered a **past** (already-executed)
  tool call back into history as literal text:
  `"Assistant requested tool call:\n" + json.dumps({"name":...,"args":...,"id":...}, sort_keys=True)`.
- gemma27b would then **imitate that exact template** on a later turn instead
  of writing a real `\`\`\`tool_code\`\`\`` fence or a final answer. Since the
  imitation has empty `tool_calls=[]`, it's genuinely a no-tool decision at
  the LangGraph level (confirmed: the episode goes straight to
  `researcher_tools` with nothing to run, then `compress_research` — the
  imitated call is never executed). But it is **not evidence the researcher
  intended to stop** — it's a copy-paste artifact.

**Fix applied** (root cause, not a downstream patch): rewrote
`_plain_turn_from_message`'s `AIMessage` branch
(`open_deep_research/src/open_deep_research/deep_researcher.py`) to render
past tool calls as prose via a new `_plain_tool_call_summary()` helper —
never as the copyable `{"name":...,"args":...,"id":...}` JSON dict. The new
text explicitly tells the model "this already happened, do not repeat it as
text." This only touches the gemma-only rendering path; qwen/llama/gpt-oss
are unaffected (confirmed by re-running `attack/tests/test_v3_production.py`
under the `odr` conda env — 35/35 pass, including the updated
`test_gemma_history_is_rendered_without_tool_roles_for_vllm`).

**This means gemma27b's entire existing production corpus (432/432 shards,
all 4 datasets) was invalidated** — the fix changes what the model sees on
every post-first-turn decision, corpus-wide, not just a few rows. That's why
all 4 datasets are being rerun from scratch, not just the affected rows.

Sample-size math that justified a full rerun rather than trying to salvage
the old data: after fixing a *separate* downstream bug (see #3 below) so
these rows are correctly excluded from PCR, only 928/21,600 (4.3%) of
gemma27b's OLD rows were PCR-eligible, and per (dataset, condition) cell that
ranged from 2 to 123 — nowhere near usable for a real curve. Confirmed via:
```python
from analysis.completion_inventory import build_inventory
from analysis.canonical import canonical_completion as cc
from analysis import config
result = build_inventory(config.DEFAULT_PRODUCTION_ROOT, models=('gemma27b',))
records = cc.build_canonical_records(result.rows, max_workers=8)
```
(this was run against the OLD, now-quarantined trace corpus — don't expect
to reproduce it against current `results/traces/gemma27b/`, which is empty
until the new jobs land).

### 2. gpt_oss_20b/2wiki: harmony-parser crash (pre-existing, now deployed)

Older, separately-diagnosed bug: vLLM's own server-side
`openai_harmony.StreamableParser` crashes (`HarmonyError`) on certain
malformed gpt-oss-20b tool-call headers, taking down the whole request. Fix
was already built and unit-tested (`attack/runtime/gptoss_lenient_harmony_parser.py`,
a monkey-patch that catches `HarmonyError` and freezes forward progress
instead of raising) but had **never been deployed** — this session wired it
in:
- `attack/launch/generate_production_arrays.py`: `gpt_oss_20b`'s
  `ModelResources.tool_parser_plugin` changed from `""` to
  `"attack/runtime/gptoss_lenient_harmony_parser.py"` (`--tool-call-parser`
  stays `"openai"`, unchanged, per the fix's own docstring).
- Regenerated **only** `results/sbatch/run_v3_2wiki_gpt_oss_20b.sbatch` in
  place (verified the diff against the old frozen version was exactly one
  new line — the `--tool-parser-plugin` flag). Did **not** touch the other 3
  gpt_oss_20b sbatch scripts or their already-complete data (hotpot/musique/
  multihoprag, 432/432 shards, fine as-is).
- Confirmed the 10 unit tests for the harmony-parser fix actually pass under
  the real `vllm` conda env (they were previously only ever run under
  `whitebox`, where they silently skip): `/home/sbhengesa/.conda/envs/vllm/bin/python
  -m unittest attack.tests.test_gptoss_lenient_harmony_parser` → 10/10 OK.

gpt_oss_20b's hotpot/musique/multihoprag data was **not** touched, quarantined,
or invalidated — only 2wiki was ever broken for this model.

### 3. `pcr_eligible` computation bug (analysis-layer, now fixed)

Separate from both of the above — a bug in how *any* future
`TOOL_OR_PARSER_FAILURE` route gets counted. In
`analysis/canonical/canonical_completion.py::build_canonical_record`, the
formula was:
```python
pcr_eligible = route not in (TRACE_LOGGING_FAILURE, EXECUTION_FAILURE) and ep0.branch_reached
```
This implicitly relied on `TOOL_OR_PARSER_FAILURE` always co-occurring with
`branch_reached=False` (true for the pre-existing Llama search-tool-error
case, which is always caught by the `and ep0.branch_reached` clause). The
gemma imitation case (once classified — see next paragraph) produces
`TOOL_OR_PARSER_FAILURE` from the **anchored** (`branch_reached=True`) path,
which the old formula didn't catch, so it was wrongly marking those rows
`pcr_eligible=True`. **Fixed** by adding `TOOL_OR_PARSER_FAILURE` explicitly
to the exclusion tuple. Covered by the existing
`test_13_tool_parser_failure_not_counted_as_no_tool_exit` in
`analysis/canonical/test_canonical_completion.py` (21/21 pass).

This fix depends on infrastructure built **earlier in this same session**
(before a context-compaction, so not re-described in full here, just
pointers):
- `analysis/trace_parser.py`: `_reconstruct_imitated_tool_call()` +
  `_IMITATED_TOOL_CALL_RE` — detects the imitation pattern in a decision's
  `content` text. Deliberately does **not** change `has_tool_calls`/
  `tool_call_names`/`is_no_tool_exit` (an earlier version of this fix did,
  which was wrong — see "Errors and fixes" logic below — it was corrected to
  be flag-only).
- `analysis/trace_schema.py`: `ResearcherDecision.reconstructed_from_imitated_history_text`
  field + `ResearcherEpisode.terminal_decision_is_imitated_no_tool_call`
  property.
- `analysis/canonical/canonical_completion.py`: `REASON_IMITATED_NO_TOOL_CALL`
  constant + `classify_local_terminal_route()` routes a terminal
  no-tool-exit decision carrying that flag to `TOOL_OR_PARSER_FAILURE`
  instead of `NO_TOOL_DIRECT_EXIT`.

All of this is validated end-to-end: `analysis/tests/test_trace_parser.py`'s
`ImitatedToolCallReconstructionTests` (9 tests, including one against a real
hand-verified production trace fixture) + `test_canonical_completion.py` (21
tests) + full `analysis/tests/` discover (56 tests) all pass.

## What was quarantined (not deleted)

```
archive/legacy_results/gemma27b_imitated_tool_call_history_bug_run_20260928/
├── traces_gemma27b/   (was results/traces/gemma27b, 7.4GB)
├── runs_gemma27b/     (was results/runs/gemma27b)
└── logs_gemma27b/     (was results/logs/gemma27b, SLURM .out/.err)
```
`archive/` is entirely git-ignored (confirmed via `git check-ignore`) —
matches the existing convention already used for two earlier gemma27b
parser-bug invalidations (`gemma27b_broken_functiongemma_parser_run_20260913`,
`gemma27b_stale_functiongemma_gate_20260914` — same directory, for context).
Nothing was permanently deleted; this is a plain `mv`, fully reversible.
`results/traces/gemma27b`, `results/runs/gemma27b`, and the 4 dataset
subdirectories under `results/logs/gemma27b/` were recreated empty
(SLURM needs the log output directories to pre-exist) so the new jobs have
somewhere to write.

gpt_oss_20b's data was **not** quarantined — only 2wiki was ever empty
(0/108), the other 3 datasets are untouched and still valid.

## Git state — IMPORTANT, read before committing anything

**Nothing from today was committed.** Per the standing rule (never commit
without being asked), all of today's fixes sit uncommitted in the working
tree, mixed in with **older, unrelated uncommitted work from earlier in this
same long session** (the gemma/gpt-oss detailed-analysis bundle, PCR figure
redesign, `INCLUDED_MODELS` config expansion). `git status --short` currently
shows 74 lines. Do not `git add -A` blindly.

Files that belong to **today's fix** (if/when the user asks for a commit,
these are the right scope — do NOT bundle in the unrelated bundle-building
work below):
- `analysis/canonical/canonical_completion.py` (pcr_eligible fix)
- `analysis/trace_parser.py`, `analysis/trace_schema.py` (imitation
  reconstruction — built earlier this session, part of the same fix chain)
- `analysis/tests/test_trace_parser.py` (includes 2 corrected stale
  assertions from an earlier, wrong version of the reconstruction fix)
- `attack/launch/generate_production_arrays.py` (gpt_oss_20b tool_parser_plugin
  wiring)
- `attack/tests/test_v3_production.py` (updated gemma-history-rendering
  assertion)
- `open_deep_research/src/open_deep_research/deep_researcher.py` (the actual
  root-cause fix)
- New untracked files: `attack/runtime/gptoss_lenient_harmony_parser.py`,
  `attack/tests/test_gptoss_lenient_harmony_parser.py` (built earlier this
  session, deployed today)
- Regenerated (not git-tracked, lives under `results/`, git-ignored):
  `results/sbatch/run_v3_2wiki_gpt_oss_20b.sbatch`

Files modified/untracked that are **NOT part of today's fix** (leftover from
earlier in this session — leave alone unless the user asks about them
specifically):
- `analysis/__init__.py`, `analysis/cli.py`, `analysis/config.py`,
  `analysis/tests/test_config_guards.py` — the `INCLUDED_MODELS` expansion
  to add gemma27b/gpt_oss_20b to the analysis scope.
- `analysis/canonical/results_bundle/figures_bundle.py` — the `datasets`
  parameter added to `plot_pcr_attack_panels` for asymmetric dataset
  coverage (used by the new bundle below).
- `analysis/canonical/gemma_gptoss_bundle/` (new package),
  `gemma_gptoss_results_bundle/` (new output dir) — the gemma27b/gpt_oss_20b
  detailed-analysis bundle. **`gemma_gptoss_results_bundle/RESULTS.md`
  currently documents gemma27b as excluded pending a fix + rerun — this is
  now stale/outdated language** (it predates today's ODR fix and the
  rerun being in flight) but the underlying data files
  (`data/pcr_run_level.csv` etc.) still hold the OLD, now-superseded
  gemma27b rows. Once the rerun completes, this whole bundle needs
  regenerating for gemma27b (see "Next steps" below) — don't trust its
  current gemma27b numbers, gpt_oss_20b's 3-dataset numbers in it are still
  valid.
- `qwen_llama_results_bundle/*` (modified/deleted PCR figures),
  `requirements-whitebox.txt` — earlier PCR-curve figure redesign work,
  fully unrelated to any of this.

## Next steps, in order

1. **Wait for the jobs.** Don't poll aggressively — `sacct`/`squeue` a
   couple times a day is plenty for a 30h-per-task, `%1`-throttled array.
   Watch for early failures in the first few tasks especially for
   gemma27b (confirms the ODR fix didn't break something else) and
   gpt_oss_20b/2wiki (confirms the harmony patch actually prevents the
   crash in a live server, not just in unit tests).
2. Once each dataset reaches 108/108 shards, spot-check a handful of fresh
   gemma27b traces by hand for the imitation pattern (grep for
   `"Assistant requested tool call:"` in `results/traces/gemma27b/` — should
   be zero hits now; also manually confirm the model is now producing real,
   executed tool calls on turn 2+ instead of silently stopping) before
   trusting the corpus at scale.
3. Rebuild gemma27b's canonical records
   (`cc.build_canonical_records(build_inventory(..., models=('gemma27b',)).rows)`)
   and sanity-check the `pcr_eligible` distribution looks like a real
   corpus now (expect the vast majority of rows either `RESEARCH_COMPLETE`
   or genuine `NO_TOOL_DIRECT_EXIT`, not 96% `TOOL_OR_PARSER_FAILURE` like
   before).
4. Regenerate the full `gemma_gptoss_results_bundle` (both stages: `python -m
   analysis.canonical.gemma_gptoss_bundle.build_bundle --stage all`) once
   both gemma27b and gpt_oss_20b (now 4/4 datasets) are complete — this
   naturally picks up gpt_oss_20b/2wiki too, so `MODEL_DATASETS` in
   `analysis/canonical/gemma_gptoss_bundle/scope.py` should be updated to
   drop the gpt_oss_20b/2wiki exclusion (currently
   `("hotpot","musique","multihoprag")` for gpt_oss_20b — change to
   `config.DATASETS` once 2wiki is real).
5. Rewrite `gemma_gptoss_results_bundle/RESULTS.md` §0 and the gemma27b
   findings sections with real numbers instead of the current
   "excluded pending fix" language.
6. Only then consider a commit — and even then, ask the user first, scoped
   to just today's fix files (see the list above), not the unrelated
   leftover work.
7. Separately, still pending from before this whole detour: the **543-row
   human semantic-adjudication review** — this was flagged as "the next
   task" after an earlier repository-cleanup effort, and has not been
   started.

## Useful commands

```bash
# Job status
squeue -u $USER
sacct -j 166533,166534,166535,166536,166537 --format="JobID,State,Elapsed,ExitCode" | head -40

# Shard progress
for d in hotpot 2wiki musique multihoprag; do echo "gemma27b/$d: $(find results/runs/gemma27b/$d -name '*.jsonl' 2>/dev/null | wc -l)/108"; done
echo "gpt_oss_20b/2wiki: $(find results/runs/gpt_oss_20b/2wiki -name '*.jsonl' 2>/dev/null | wc -l)/108"

# Spot-check for the imitation pattern in fresh gemma27b traces (should be empty)
grep -rl "Assistant requested tool call:" results/traces/gemma27b/ 2>/dev/null | head

# Test suites touched by today's fix (all passed as of this handover)
python -m unittest discover -s analysis/tests                                    # whitebox env, 56 tests
python -m unittest analysis.canonical.test_canonical_completion                  # whitebox env, 21 tests
/home/sbhengesa/.conda/envs/vllm/bin/python -m unittest attack.tests.test_gemma3_tool_code_parser attack.tests.test_gptoss_lenient_harmony_parser   # vllm env, 21 tests
/home/sbhengesa/.conda/envs/odr/bin/python -m unittest attack.tests.test_v3_production   # odr env, 35 tests
```

## Conda envs (in case the next session doesn't already know)

- `whitebox` — analysis/, no vllm/langchain/torch. Default for anything under
  `analysis/`.
- `vllm` — has `vllm`, `openai_harmony`. Needed for
  `attack/runtime/gemma3_tool_code_parser.py` and
  `attack/runtime/gptoss_lenient_harmony_parser.py`'s own tests.
- `odr` — has `langchain_core`, `langgraph`, `open_deep_research` importable.
  Needed for `attack/tests/test_v3_production.py` and anything touching
  `deep_researcher.py` directly.
