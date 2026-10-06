"""Driver for the adaptive-cue pilot: smoke, development, and held-out runs.

Idempotent: each cell writes one JSONL line keyed by a stable cell id; a rerun
skips cells already present with all expected arms. Raw outputs are append-only
and never overwritten.

Execution requires two OpenAI-compatible vLLM endpoints (victim and attacker);
they may be the same served process if request state is isolated (it is: every
victim run has its own run_id/thread and MemorySaver, and every attacker call is
stateless). All inference must run on NRP; see nrp/ for launch assets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from attack.adaptive_pilot import config as pilot_config
from attack.adaptive_pilot.attacker import AttackerConfig
from attack.adaptive_pilot.feedback import ACCESS_SETTINGS, BOUNDED_FEEDBACK, CONTEXT_ONLY
from attack.adaptive_pilot.runner import PilotCellConfig, run_pilot_cell
from attack.adaptive_pilot.validation import LocalTokenizerCounter, TokenCounter, VllmTokenCounter
from attack.runtime.hashing import stable_hash
from attack.runtime.runner import V3ModelConfig

PILOT_RESULTS_ROOT = pilot_config_default = Path("results/adaptive_pilot")


def cell_id(victim_model: str, dataset: str, example_id: str, condition: str, access: str) -> str:
    return stable_hash(
        {"victim_model": victim_model, "dataset": dataset, "example_id": example_id, "condition": condition, "access": access},
        length=16,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_paths(split_dir: Path, dataset: str, split: str) -> dict:
    base = split_dir / dataset / split
    return {
        "manifest": base / f"{dataset}_{split}_manifest.jsonl",
        "corpus": base / f"{dataset}_{split}_corpus.jsonl",
        "evidence": base / f"{dataset}_{split}_evidence.jsonl",
    }


def _victim_config(model_key: str, base_url: str, api_key: str) -> V3ModelConfig:
    spec = pilot_config.VICTIM_MODELS[model_key]
    return V3ModelConfig(
        model_name=spec["model_id"], base_url=base_url, provider="openai", api_key=api_key,
        research_max_tokens=pilot_config.RESEARCH_MAX_TOKENS,
        compression_max_tokens=pilot_config.COMPRESSION_MAX_TOKENS,
        final_report_max_tokens=pilot_config.FINAL_REPORT_MAX_TOKENS,
        temperature=pilot_config.VICTIM_TEMPERATURE, seed=pilot_config.VICTIM_SEED,
    )


def _attacker_config(base_url: str, api_key: str, model_key: str) -> AttackerConfig:
    spec = pilot_config.VICTIM_MODELS[model_key]
    return AttackerConfig(
        base_url=base_url, model_id=spec["model_id"], revision=spec["revision"], api_key=api_key,
        temperature=pilot_config.ATTACKER_TEMPERATURE, top_p=pilot_config.ATTACKER_TOP_P,
    )


def _token_counter(args, attacker_cfg: AttackerConfig) -> TokenCounter:
    if args.tokenizer_json:
        return LocalTokenizerCounter(Path(args.tokenizer_json))
    return VllmTokenCounter(args.attacker_base_url, attacker_cfg.model_id)


def _iter_cells(args, victims: Iterable[str], conditions: Iterable[str], datasets, split, split_dir):
    for dataset in datasets:
        paths = _split_paths(split_dir, dataset, split)
        questions = _load_jsonl(paths["manifest"])
        if args.max_questions_per_dataset:
            questions = questions[: args.max_questions_per_dataset]
        for q in questions:
            for model_key in victims:
                for condition in conditions:
                    yield dataset, paths, q, model_key, condition


def run(args) -> int:
    split_dir = Path(args.split_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / f"{args.split}_{args.access}_cells.jsonl"
    done = {c.get("cell_id") for c in _load_jsonl(results_path)}

    victims = args.victims or list(pilot_config.PILOT_VICTIMS)
    conditions = args.conditions or list(pilot_config.LAUNCH_CONDITION_IDS)
    datasets = args.datasets or ["hotpot", "2wiki"]

    attacker_cfg = _attacker_config(args.attacker_base_url, args.api_key, args.attacker_model_key)
    token_counter = _token_counter(args, attacker_cfg)

    processed = skipped = failed = 0
    for dataset, paths, q, model_key, condition in _iter_cells(args, victims, conditions, datasets, args.split, split_dir):
        cid = cell_id(model_key, dataset, q["example_id"], condition, args.access)
        if cid in done:
            skipped += 1
            continue
        cell_cfg = PilotCellConfig(
            victim_model_key=model_key,
            victim=_victim_config(model_key, args.victim_base_url, args.api_key),
            attacker=attacker_cfg,
            dataset=dataset, example_id=q["example_id"], question=q["question"], condition=condition,
            corpus_path=str(paths["corpus"]), evidence_path=str(paths["evidence"]),
            output_dir=str(out_root / "traces" / model_key),
            access_setting=args.access, top_k=pilot_config.TOP_K,
            emergency_max_react_tool_calls=pilot_config.EMERGENCY_MAX_REACT_TOOL_CALLS,
            max_researcher_iterations=pilot_config.MAX_RESEARCHER_ITERATIONS,
        )
        last_exc = None
        for attempt in range(pilot_config.MAX_TECHNICAL_RETRIES + 1):
            try:
                record = run_pilot_cell(cell_cfg, token_counter)
                record["cell_id"] = cid
                record["split"] = args.split
                record["technical_attempt"] = attempt
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                done.add(cid)
                processed += 1
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 20))
        if last_exc is not None:
            failed += 1
            err_path = out_root / "infra_failures.jsonl"
            with err_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"cell_id": cid, "error": repr(last_exc), "ts": time.time()}) + "\n")
        print(json.dumps({"processed": processed, "skipped": skipped, "failed": failed, "last": cid}))
    print(json.dumps({"done": True, "processed": processed, "skipped": skipped, "failed": failed, "results_path": str(results_path)}))
    return 1 if failed and args.fail_on_error else 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the adaptive-cue pilot.")
    ap.add_argument("--split", choices=("smoke", "dev", "heldout"), required=True)
    ap.add_argument("--access", choices=ACCESS_SETTINGS, default=BOUNDED_FEEDBACK)
    ap.add_argument("--split-dir", default=str(Path("results/adaptive_pilot/splits")))
    ap.add_argument("--out-root", default=str(PILOT_RESULTS_ROOT))
    ap.add_argument("--victim-base-url", required=True, help="OpenAI-compatible vLLM endpoint for victim inference")
    ap.add_argument("--attacker-base-url", required=True, help="OpenAI-compatible vLLM endpoint for attacker inference")
    ap.add_argument("--attacker-model-key", default=pilot_config.ATTACKER_MODEL_KEY,
                    choices=sorted(pilot_config.VICTIM_MODELS),
                    help="which model serves as the attacker (default qwen14b; use qwen7b to fit a 24GB GPU)")
    ap.add_argument("--api-key", default=os.environ.get("PILOT_API_KEY", "EMPTY"))
    ap.add_argument("--tokenizer-json", default=None, help="local tokenizer.json for length counting (else vLLM /tokenize)")
    ap.add_argument("--victims", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-questions-per-dataset", type=int, default=None)
    ap.add_argument("--fail-on-error", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
