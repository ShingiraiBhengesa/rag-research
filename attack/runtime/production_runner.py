"""Production shard runner for benchmark-local ODR arrays."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from attack.metrics import em_f1
from attack.runtime.hashing import stable_hash
from attack.runtime.production_models import FINAL_MODEL_IDS
from attack.runtime.runner import V3ModelConfig, V3RunRequest, run_single_condition
from attack.runtime.trace_schema import validate_branch_trace


MODEL_IDS = dict(FINAL_MODEL_IDS)

STOP_REQUESTED = False


def _handle_stop(signum, frame) -> None:  # pragma: no cover - exercised by SLURM.
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stable_run_id(*, model: str, dataset: str, condition: str, example_id: str) -> str:
    return stable_hash(
        {
            "schema": "v3_production_run_id_v1",
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "example_id": example_id,
        },
        length=None,
    )


def shard_for_index(rows: list[dict], *, shard_index: int, num_shards: int) -> list[dict]:
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


async def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    signal.signal(signal.SIGTERM, _handle_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_stop)

    model_id = args.model_id or MODEL_IDS[args.model]
    rows = _load_manifest(args.manifest)
    rows = [row for row in rows if row["dataset"] == args.dataset]
    rows = sorted(rows, key=lambda row: int(row["sample_ordinal"]))
    shard_rows = shard_for_index(rows, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.limit is not None:
        shard_rows = shard_rows[: args.limit]

    out_dir = args.out_root / "runs" / args.model / args.dataset / args.condition
    result_path = out_dir / f"v3_{args.model}_{args.dataset}_{args.condition}_shard{args.shard_index:02d}of{args.num_shards:02d}.jsonl"
    progress_path = out_dir / f"progress_shard{args.shard_index:02d}of{args.num_shards:02d}.json"
    done = _load_done_run_ids(result_path)

    model = V3ModelConfig(
        model_name=model_id,
        base_url=args.model_base_url,
        provider=args.model_provider,
        api_key=args.model_api_key,
        research_max_tokens=args.research_model_max_tokens,
        compression_max_tokens=args.compression_model_max_tokens,
        final_report_max_tokens=args.final_report_model_max_tokens,
        temperature=args.model_temperature,
        seed=args.model_seed,
    )
    processed = 0
    skipped = 0
    errors = 0
    consecutive_errors = 0
    started = time.perf_counter()

    for row in shard_rows:
        run_id = stable_run_id(
            model=args.model,
            dataset=args.dataset,
            condition=args.condition,
            example_id=row["example_id"],
        )
        if run_id in done:
            skipped += 1
            continue
        if STOP_REQUESTED:
            break
        branch_started = time.perf_counter()
        try:
            trace = await run_single_condition(
                V3RunRequest(
                    dataset=args.dataset,
                    example_id=row["example_id"],
                    question=row["question"],
                    condition=args.condition,
                    corpus_path=args.corpus,
                    evidence_path=args.evidence,
                    output_dir=args.out_root / "traces" / args.model,
                    model=model,
                    top_k=args.top_k,
                    retrieval_scope="per_example",
                    emergency_max_react_tool_calls=args.emergency_max_react_tool_calls,
                    max_researcher_iterations=args.max_researcher_iterations,
                    allow_production_ids=True,
                    run_id=run_id,
                )
            )
            validate_branch_trace(trace)
            answer_metrics = _score_answer(trace.get("extracted_answer") or "", row)
            final_status = "ok"
            error = ""
            consecutive_errors = 0
        except Exception as exc:
            trace = {}
            answer_metrics = {"exact_match": 0.0, "f1": 0.0, "valid_answer": False}
            final_status = "error"
            error = repr(exc)
            errors += 1
            consecutive_errors += 1

        result_row = {
            "schema_version": "v3_production_result_v1",
            "run_id": run_id,
            "model": args.model,
            "model_id": model_id,
            "dataset": args.dataset,
            "condition": args.condition,
            "example_id": row["example_id"],
            "sample_ordinal": row["sample_ordinal"],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "array_index": args.array_index,
            "final_status": final_status,
            "error": error,
            "runtime_seconds": time.perf_counter() - branch_started,
            "trace_path": str(_trace_path(args.out_root / "traces" / args.model, trace, row)),
            "extracted_answer": trace.get("extracted_answer") if isinstance(trace, Mapping) else "",
            "gold_answer": row.get("answer"),
            **answer_metrics,
            "frozen_config_hashes": {
                "manifest_sha256": _file_sha256(args.manifest),
                "corpus_sha256": _file_sha256(args.corpus),
                "evidence_sha256": _file_sha256(args.evidence),
            },
            "censored_or_timeout": final_status != "ok",
            "stop_requested": STOP_REQUESTED,
        }
        _append_validated_jsonl(result_path, result_row)
        done.add(run_id)
        processed += 1
        _atomic_write_json(
            progress_path,
            {
                "schema_version": "v3_shard_progress_v1",
                "model": args.model,
                "dataset": args.dataset,
                "condition": args.condition,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "consecutive_errors": consecutive_errors,
                "last_run_id": run_id,
                "stop_requested": STOP_REQUESTED,
            },
        )
        if args.abort_after_consecutive_errors and consecutive_errors >= args.abort_after_consecutive_errors:
            raise RuntimeError(f"aborting shard after {consecutive_errors} consecutive row errors")

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "condition": args.condition,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
        "result_path": str(result_path),
        "progress_path": str(progress_path),
    }
    _atomic_write_json(out_dir / f"summary_shard{args.shard_index:02d}of{args.num_shards:02d}.json", summary)
    if args.fail_on_error and errors:
        raise RuntimeError(f"shard completed with {errors} errors")
    return summary


def _load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [row.get("example_id") for row in rows if str(row.get("example_id") or "").startswith("dev_")]
    if bad:
        raise ValueError(f"production manifest contains dev ids: {bad[:10]}")
    return rows


def _load_done_run_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("final_status") == "ok":
            done.add(str(row["run_id"]))
    return done


def _append_validated_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(row, sort_keys=True, ensure_ascii=False)
    json.loads(raw)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(raw + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
        tmp_name = fh.name
    os.replace(tmp_name, path)


def _score_answer(extracted: str, row: Mapping[str, Any]) -> dict[str, Any]:
    aliases = [str(row.get("answer") or ""), *[str(alias) for alias in row.get("answer_aliases") or []]]
    aliases = [alias for alias in aliases if alias.strip()]
    if not str(extracted).strip():
        return {"exact_match": 0.0, "f1": 0.0, "valid_answer": False}
    scores = [em_f1(str(extracted), alias) for alias in aliases] or [(0.0, 0.0)]
    em = max(score[0] for score in scores)
    f1 = max(score[1] for score in scores)
    return {"exact_match": em, "f1": f1, "valid_answer": bool(str(extracted).strip())}


def _trace_path(root: Path, trace: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    if trace and trace.get("dataset") and trace.get("example_id") and trace.get("condition"):
        return root / "branches" / str(trace["dataset"]) / str(trace["example_id"]) / str(trace["condition"]) / "trace.json"
    return root / "branches" / str(row["dataset"]) / str(row["example_id"]) / "missing_trace.json"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one production model/dataset/condition shard.")
    parser.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--dataset", choices=("hotpot", "2wiki", "musique", "multihoprag"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-api-key", default="EMPTY")
    parser.add_argument("--model-temperature", type=float, default=0.0)
    parser.add_argument("--model-seed", type=int, default=20260820)
    parser.add_argument("--research-model-max-tokens", type=int, default=1000)
    parser.add_argument("--compression-model-max-tokens", type=int, default=2000)
    parser.add_argument("--final-report-model-max-tokens", type=int, default=1000)
    parser.add_argument("--emergency-max-react-tool-calls", type=int, default=30)
    parser.add_argument("--max-researcher-iterations", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--abort-after-consecutive-errors", type=int, default=3)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = asyncio.run(run_shard(parse_args()))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
