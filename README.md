# prematurestop

**Premature-completion attacks on agentic retrieval-augmented generation.**

An agentic RAG system decides on its own when it has gathered enough evidence
to stop searching and answer. This repository studies how a short piece of
text planted in a retrieved document, which we call a *cue*, can push that
decision forward. The agent stops before it has retrieved the evidence a
multi-hop question needs, so its answer is built on incomplete evidence.

The victim system is [Open Deep Research](https://github.com/langchain-ai/open_deep_research)
(ODR), a LangGraph supervisor/researcher agent. It is run unmodified except
for a local, deterministic search backend and a set of logging and
reproducibility hooks (see [`third_party/open_deep_research/PATCHES.md`](third_party/open_deep_research/PATCHES.md)).

All current experiments are **white-box**: we serve the victim models
ourselves (vLLM) and record full agent traces. A black-box track (hosted APIs,
no trace access) will be added later under the same `experiments/` and
`results/` hierarchy as `blackbox/`.

---

## The two experiments

| | E1 — fixed-cue matrix | E2 — bridge mechanism |
|---|---|---|
| Question | Do fixed cues cause premature completion, and how much? | *Why* do they work: structure, truth, or selection of the bridge fact? |
| Conditions | 12 (clean, neutral, continue-search, 7 fixed cues, 2 oracle) | 8 arms (neutral, fixed and generated sufficiency/saturation cues, entity-only, unsupported) |
| Victims | Qwen2.5-14B, Llama-3.1-8B, Gemma-3-27B, gpt-oss-20b | Qwen2.5-14B, Llama-3.1-8B |
| Datasets | HotpotQA, 2WikiMultihopQA, MuSiQue, MultiHop-RAG | HotpotQA, 2WikiMultihopQA |
| n | 450 questions per dataset × condition | Same canonical 450 questions per dataset |
| Status | **Complete.** Results in `results/whitebox/e1_fixed_cue_matrix/` | v1 complete but superseded (see caveat); v2 pending |
| Experiment folder | [`experiments/whitebox/e1_fixed_cue_matrix/`](experiments/whitebox/e1_fixed_cue_matrix/README.md) | [`experiments/whitebox/e2_bridge_mechanism/`](experiments/whitebox/e2_bridge_mechanism/README.md) |

### Outcomes (shared by both experiments)

* **PCR@k: post-observation completion rate.** This is the fraction of runs in
  which the agent's researcher declares `ResearchComplete` within *k* decisions
  after the cued observation. It is the primary measure of premature stopping.
  The single authoritative implementation is
  `src/prematurestop/evaluation/completion.py`.
* **Evidence coverage.** This is the fraction of the dataset's annotated gold
  supporting documents that the agent actually retrieved, and that reached
  the final report.
* **Final-answer accuracy.** This is semantic answer correctness against the
  gold answer (`src/prematurestop/evaluation/answer_scoring.py`).

Runs are always carried to the final answer. The study never truncates a run
at a fixed step, because evidence coverage and answer accuracy need the full
trajectory.

### Threat model (white-box)

The attacker controls the content of **one** document that the agent
retrieves on its first search (d1). It cannot change the question, the
prompts, the model or any other document. The cue is appended to d1's text,
and everything after that is the agent's own behaviour.

---

## Repository layout

```text
prematurestop/
├── src/prematurestop/            # the Python package (all project code)
│   ├── system/                   # the victim harness: runs ODR against local search
│   │   ├── runner.py             #   run one question under one condition, write the trace
│   │   ├── benchmark_search.py   #   deterministic BM25 "web_search" over the benchmark corpus
│   │   ├── models.py             #   victim model registry (served model IDs, tool parsers)
│   │   └── tool_parsers/         #   vLLM tool-call parser plugins (Gemma-3, gpt-oss)
│   ├── data/                     # datasets: question manifests, corpora, gold evidence, splits
│   ├── cues/                     # attack conditions
│   │   ├── fixed/                #   E1 fixed cue texts and the condition registry
│   │   └── dynamic/              #   E2 bridge-anchored cue generator (decompose → anchor →
│   │                             #   generate K → validity gates V1–V5 → score → select)
│   ├── evaluation/               # outcome definitions: trace parsing, PCR, evidence, answers
│   ├── stats/                    # paired bootstrap, McNemar, multiple-comparison control
│   ├── infra/                    # hashing, append-only checkpoint store
│   ├── pipelines/                # experiment orchestration
│   │   ├── fixed_cue_matrix/     #   E1: matrix generation, SLURM arrays, runtime gate
│   │   └── bridge_mechanism/     #   E2: phases, freeze manifest, analysis
│   ├── analysis/fixed_cue_matrix/  # E1 result bundles, audits, paper figures
│   └── paths.py                  # every filesystem location, defined once
├── third_party/open_deep_research/  # vendored victim agent + PATCHES.md
├── experiments/whitebox/         # per-experiment design docs, freeze manifests, launch scripts
│   ├── e1_fixed_cue_matrix/
│   └── e2_bridge_mechanism/
├── results/whitebox/             # published, citable results (CSV / Markdown / figures)
│   ├── e1_fixed_cue_matrix/
│   └── e2_bridge_mechanism/
├── tests/                        # unit tests, mirroring src/ layout
├── docs/                         # changelog, audits, legacy notes from earlier layouts
├── environments/                 # pinned dependency lists per environment role
└── pyproject.toml
```

Gitignored, local only:

* `local_data/` holds raw victim traces, per-run records, corpora, manifests
  and logs. Its location can be overridden with `PREMATURESTOP_LOCAL_DATA`.
* `archive/` holds superseded pilots and notes, kept for provenance.

---

## Results: where to look first

| You want | Open |
|---|---|
| E1 headline numbers for Qwen-14B / Llama-8B | `results/whitebox/e1_fixed_cue_matrix/qwen_llama/MASTER_RESULTS.md` |
| E1 Gemma-27B / gpt-oss-20b | `results/whitebox/e1_fixed_cue_matrix/gemma_gptoss/RESULTS.md` |
| Canonical metric definitions and all-model PCR tables | `results/whitebox/e1_fixed_cue_matrix/canonical/chapter3_final_source_of_truth.md` |
| Exact text of every E1 cue | `results/whitebox/e1_fixed_cue_matrix/canonical/condition_metadata.csv` |
| Evidence coverage and per-hop analysis | `results/whitebox/e1_fixed_cue_matrix/evidence_hop_coverage/`, `.../evidence_distributions/` |
| E2 results so far (with caveat) | `results/whitebox/e2_bridge_mechanism/README.md` |

Every table is a plain CSV next to a Markdown summary that cites it.

**One file is stored compressed.** Uncompressed, the per-run answer table
`gemma_gptoss/data/answer_run_level.csv` is 160 MB, which is over GitHub's
limit. It is committed as `answer_run_level.csv.gz`, which is 48 MB. pandas
reads it directly with `pd.read_csv("....csv.gz")`, or you can decompress it
with `gunzip -k`.

---

## Setup

Python 3.11. The project uses three environments because a run has three
roles: serve the model, run the agent, analyse. They have incompatible
dependencies.

| Role | What runs there | Install |
|---|---|---|
| **analysis** | tests, analysis, launch generation | `pip install -r environments/analysis.txt && pip install -e .` |
| **agent** | the ODR graph during a run | `pip install -e third_party/open_deep_research && pip install -e .` |
| **serving** | vLLM OpenAI-compatible server for the victim; also the E2 cue generator and NLI checker | `pip install vllm==0.23.0` (bf16, no quantisation) |

Point the launchers at your interpreters with these variables:

```bash
export PREMATURESTOP_ANALYSIS_PYTHON=/path/to/analysis/bin/python
export PREMATURESTOP_AGENT_PYTHON=/path/to/agent/bin/python
export PREMATURESTOP_VLLM_ENV=/path/to/serving/env
export PREMATURESTOP_LOCAL_DATA=/path/to/local_data   # optional; default <repo>/local_data
```

If the package isn't installed, put both source roots on the path:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/open_deep_research/src"
```

No GPU is needed for the tests, the analysis, or generating launch scripts.
A GPU is needed only to actually run the victim.

### Tests

```bash
python -m unittest discover -s tests -t .
```

The tests need numpy and matplotlib. Tests for the vLLM tool parsers skip
themselves unless `vllm` / `openai_harmony` are importable.

---

## Reproducibility rules

These rules held for every run reported here.

1. **Frozen before running.** Each experiment writes a freeze manifest with
   SHA-256 hashes of its code, prompts, splits and victim settings, and
   verifies it at job start. A run with a hash mismatch refuses to start.
2. **Deterministic victim.** Temperature 0 with a fixed seed. vLLM serves
   with `--generation-config vllm`, bf16 and eager mode on L40/H100 GPUs. No
   quantisation is used.
3. **No silent dropping.** Every question in the manifest gets a row,
   including technical failures. Those rows carry a status and a reason, so
   the denominators are always visible.
4. **Paired design.** Every condition runs on the same questions as its
   control. Contrasts are paired: bootstrap CIs and McNemar tests.
5. **Append-only records.** Run outputs are fsync'd JSONL checkpoints, so
   resuming a run never overwrites a finished row.

---

## History

The code was reorganised into this layout on 2026-10-16. Every file was moved
with `git mv`, so `git log --follow <file>` shows its full history. The
old→new path table is `docs/MIGRATION_2026-10-16.md`. The
pre-restructure branches are kept as `archive/*` tags. The previous top-level
README is preserved at `docs/notes/README_pre_restructure_legacy.md`.

## Contact

Shingirai Bhengesa, Saint Louis University. This is PhD research in progress;
please ask before citing numbers from `results/`.
