"""Embedded runtime gate for queued production arrays."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from prematurestop.cues.fixed.registry import suffix_for_condition
from prematurestop.system.models import FINAL_PRODUCTION_MODELS, MODEL_TOOL_CALL_PARSERS
from prematurestop.system.runner import V3ModelConfig, V3RunRequest, run_paired_replay


GATE_CONDITIONS = ("NaturalClean", "Natural_PreRegisteredSufficiencySaturation")
GATE_CONTRACT_VERSION = "v3_paired_replay_anchor_20260826"


async def run_runtime_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema_version": "v3_runtime_gate_v1",
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "model": args.model,
        "model_id": args.model_id,
        "dataset": args.dataset,
        "gate_example_id": "",
        "passed": False,
        "checks": {},
        "errors": [],
    }
    try:
        model_identity = _endpoint_model_identity(args.model_base_url)
        report["checks"]["endpoint_available"] = True
        report["checks"]["model_identity"] = model_identity
        report["checks"]["model_identity_matches"] = args.model_id in json.dumps(model_identity)
        expected_parser = MODEL_TOOL_CALL_PARSERS.get(args.model)
        report["checks"]["expected_tool_call_parser"] = expected_parser
        report["checks"]["tool_parser_configured"] = args.tool_call_parser == expected_parser
        tokenizer_report = _tokenizer_report(args.model_id, tokenizer_python=args.tokenizer_python)
        report["checks"]["tokenizer_python"] = str(args.tokenizer_python)
        report["checks"]["tokenizer_has_chat_template"] = tokenizer_report["has_chat_template"]
        report["checks"]["tokenizer_hash"] = tokenizer_report["hash"]
        report["checks"]["tokenizer_vocab_size"] = tokenizer_report["vocab_size"]
        if args.max_model_len:
            report["checks"]["max_model_len"] = args.max_model_len
            long_context_probe = _long_context_probe(args.model_base_url, args.model_id, args.max_model_len)
            report["long_context_probe"] = long_context_probe
            report["checks"]["long_context_probe_passed"] = long_context_probe["passed"]

        example = _load_gate_example(args.examples, args.dataset)
        report["gate_example_id"] = example["example_id"]
        model = V3ModelConfig(
            model_name=args.model_id,
            base_url=args.model_base_url,
            provider="openai",
            api_key=args.model_api_key,
            research_max_tokens=args.research_model_max_tokens,
            compression_max_tokens=args.compression_model_max_tokens,
            final_report_max_tokens=args.final_report_model_max_tokens,
            temperature=0.0,
            seed=args.model_seed,
            reasoning_effort=args.model_reasoning_effort,
        )
        requests = [
            V3RunRequest(
                dataset=args.dataset,
                example_id=example["example_id"],
                question=example["question"],
                condition=condition,
                corpus_path=args.corpus,
                evidence_path=args.evidence,
                output_dir=args.gate_dir / "odr_trace",
                model=model,
                top_k=args.top_k,
                allow_production_ids=False,
                run_id=f"dev_runtime_gate_{args.model}_{args.dataset}_{condition}",
            )
            for condition in GATE_CONDITIONS
        ]
        traces = await run_paired_replay(requests)
        raw_text = json.dumps(traces, sort_keys=True)
        search_events = [
            event
            for trace in traces
            for event in trace.get("events", [])
            if isinstance(event, dict) and event.get("event_type") == "benchmark_search"
        ]
        attack_events = [event for event in search_events if event.get("attacked_observation")]
        replay_anchors = [
            trace.get("paired_replay_anchor")
            for trace in traces
            if isinstance(trace.get("paired_replay_anchor"), dict)
        ]
        report["checks"].update(
            {
                "actual_supervisor_node_executed": "research_supervisor" in raw_text or "supervisor" in raw_text,
                "actual_researcher_subgraph_executed": "researcher" in raw_text,
                "benchmark_local_bm25_executed": bool(search_events),
                "canonical_suffix_injection_count": len(attack_events),
                "canonical_suffix_injection_valid": len(attack_events) == 1
                and attack_events[0].get("suffix_hash")
                == suffix_for_condition("Natural_PreRegisteredSufficiencySaturation").suffix_hash,
                "real_odr_terminal_routing_reached": "compress_research" in raw_text,
                "compression_executed": "compress_research" in raw_text,
                "research_notes_forwarded": all(_has_forwarded_research_notes(trace) for trace in traces),
                "final_report_generation_executed": all(bool(trace.get("final_report")) for trace in traces),
                "valid_answer_extracted": all(str(trace.get("extracted_answer") or "").strip() for trace in traces),
                "paired_replay_anchor_verified": len(replay_anchors) == len(traces)
                and len({anchor.get("source_shared_prefix_hash") for anchor in replay_anchors}) == 1
                and all(
                    trace.get("events")
                    and next(
                        (
                            event
                            for event in trace.get("events", [])
                            if isinstance(event, dict) and event.get("event_type") == "benchmark_search"
                        ),
                        {},
                    ).get("shared_prefix_hash")
                    == trace.get("paired_replay_anchor", {}).get("source_shared_prefix_hash")
                    for trace in traces
                ),
                "complete_trace_validated": True,
            }
        )
        report["passed"] = all(
            bool(value)
            for key, value in report["checks"].items()
            if key not in {"model_identity", "tokenizer_hash", "tokenizer_python", "tokenizer_vocab_size"}
        )
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["passed"] = False

    report["elapsed_seconds"] = time.perf_counter() - started
    out_name = "RUNTIME_GATE_PASSED.json" if report["passed"] else "RUNTIME_GATE_FAILED.json"
    (args.gate_dir / out_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _endpoint_model_identity(base_url: str) -> Any:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokenizer_report(model_id: str, *, tokenizer_python: Path) -> dict[str, Any]:
    if not tokenizer_python.exists():
        raise FileNotFoundError(f"tokenizer Python does not exist: {tokenizer_python}")
    code = """
import json
import sys
from transformers import AutoTokenizer
from prematurestop.infra.hashing import stable_hash

model_id = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
chat_template = getattr(tokenizer, "chat_template", "")
payload = {"model_id": model_id, "vocab_size": len(tokenizer), "chat_template": chat_template}
print(json.dumps({"has_chat_template": bool(chat_template), "hash": stable_hash(payload, length=None), "vocab_size": len(tokenizer)}, sort_keys=True))
"""
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run(
        [str(tokenizer_python), "-B", "-c", code, model_id],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tokenizer check failed: stdout={result.stdout[-1000:]!r} stderr={result.stderr[-1000:]!r}")
    return json.loads(result.stdout)


def _long_context_probe(base_url: str, model_id: str, max_model_len: int) -> dict[str, Any]:
    prompt_words = min(max(17_000, max_model_len // 2 + 1), max_model_len - 1024)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Repeat the final word only.\n" + ("token " * prompt_words) + "done",
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {
            "passed": True,
            "prompt_words": prompt_words,
            "max_model_len": max_model_len,
            "response_id": body.get("id"),
        }
    except urllib.error.HTTPError as exc:
        return {
            "passed": False,
            "prompt_words": prompt_words,
            "max_model_len": max_model_len,
            "status": exc.code,
            "body": exc.read().decode("utf-8", errors="replace")[-1000:],
        }


def _load_gate_example(path: Path, dataset: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") == dataset and str(row.get("example_id") or "").startswith("dev_"):
            return row
    raise ValueError(f"no dev runtime-gate example found for {dataset} in {path}")


def _has_forwarded_research_notes(trace: dict[str, Any]) -> bool:
    final_state = trace.get("final_state")
    if isinstance(final_state, Mapping) and _state_has_forwarded_research_notes(final_state):
        return True
    return _final_report_input_had_forwarded_research_notes(trace)


def _state_has_forwarded_research_notes(state: Mapping[str, Any]) -> bool:
    raw_notes = state.get("raw_notes")
    if _nonempty_note_list(raw_notes):
        return True
    notes = state.get("notes")
    return _nonempty_note_list(notes)


def _nonempty_note_list(value: Any) -> bool:
    return isinstance(value, list) and any(str(note).strip() for note in value)


def _final_report_input_had_forwarded_research_notes(trace: Mapping[str, Any]) -> bool:
    raw_events = trace.get("raw_odr_events")
    if not isinstance(raw_events, list):
        return False
    for event in raw_events:
        if not isinstance(event, Mapping):
            continue
        if _raw_event_node_name(event) != "final_report_generation":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        node_input = data.get("input")
        if isinstance(node_input, Mapping) and _state_has_forwarded_research_notes(node_input):
            return True
    return False


def _raw_event_node_name(event: Mapping[str, Any]) -> str:
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        node_name = str(metadata.get("langgraph_node") or "")
        if node_name:
            return node_name
    return str(event.get("name") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one embedded production runtime gate.")
    parser.add_argument("--model", choices=FINAL_PRODUCTION_MODELS, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-api-key", default="EMPTY")
    parser.add_argument("--dataset", choices=("hotpot", "2wiki", "musique", "multihoprag"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-python", type=Path, default=Path(os.environ.get("PREMATURESTOP_ANALYSIS_PYTHON", Path.home() / ".conda/envs/whitebox/bin/python")))
    parser.add_argument("--tool-call-parser", default="hermes")
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=20260820)
    parser.add_argument("--model-reasoning-effort", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--research-model-max-tokens", type=int, default=1000)
    parser.add_argument("--compression-model-max-tokens", type=int, default=2000)
    parser.add_argument("--final-report-model-max-tokens", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    report = asyncio.run(run_runtime_gate(parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))
    # The sbatch wrapper checks RUNTIME_GATE_PASSED.json explicitly. Returning
    # zero here lets older queued wrappers reach their own hold/cancel block
    # instead of exiting immediately under `set -e` and cascading failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
