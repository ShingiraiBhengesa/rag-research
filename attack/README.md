# Attack Paper Infrastructure

Black-box adversarial pipeline against deployed Adaptive-RAG on MuSiQue.

## Files

- `deployed_adaptive_rag.py` — wrapper around the deployed system. Exposes `query(text) → (answer, latency, num_sources)` via classifier + predict.py subprocess.
- `run_pilot_baseline.py` — runs batches of MuSiQue queries through the wrapper, saves CSV + JSONL with full observables.
- `classifier_wrapper.py`, `run_textattack*.{py,sh}` — TextAttack harness for the classifier (white-box baseline, for future).

## Results (`results/final/`)

Paper-matching FLAN-T5-XL baselines on MuSiQue (full 500 queries routed by the classifier):

| Route | EM (official MuSiQue) | n    | Notes                                      |
|-------|----------------------|------|--------------------------------------------|
| A (nor_qa)      | 11.1% | 9   | No retrieval, parametric answer            |
| B (oner_qa)     | 26.5% | 83  | Single-step BM25 retrieval (15 docs)       |
| C (ircot_qa)    | 20.6% | 408 | Iterative retrieval + chain-of-thought     |
| **Adaptive (weighted)** | **21.3%** | **500** | Reproduces audit-paper figure within 0.1pp |

Scoring uses `normalize_answer()` verbatim from `adaptive-rag/official_evaluation/musique/metrics/answer.py`.

Per-query records in `baseline_musique_route_{A,B,C}.jsonl` include: qid, question, gold, predicted_route, actual_route, system_answer, latency_ms, num_sources, exact_match, loose_match.

## Configs (`configs/`)

Modified base configs matching the cached 2024 paper baselines:
- `oner_qa_flan_t5_xl_musique_paper.jsonnet`: `bm25_retrieval_count=15`, `distractor_count="1"`
- `ircot_qa_flan_t5_xl_musique_paper.jsonnet`: `bm25_retrieval_count=6`, `distractor_count="1"`

These were created because the `--variable-replacements` flag depends on a missing `hp_manager` module. Drop-in replacements at `adaptive-rag/base_configs/` would replicate the configs used in 2024 paper predictions directory names.

## Chains Archive

`results/final/chains_archive/attack_chains_20260423.tar.gz` contains 701 chains files (per-query retrieval traces) for post-hoc analysis of which documents the pipeline saw on each query.

## Deployed System

The wrapper expects the retriever and LLM servers running at:
- Retriever: `http://localhost:8000/retrieve/` (POST)
- LLM: `http://localhost:8010/generate/` (GET, trailing slash required)

Launch script: see `adaptive-rag/` tree (pending port into this repo).
