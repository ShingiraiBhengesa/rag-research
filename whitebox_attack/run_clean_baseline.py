#!/usr/bin/env python3
"""
Clean baseline rollouts for the paper PCR rerun.

This script runs the unmodified first-hop observation for a frozen sample
manifest and writes one clean JSONL row per example. The downstream global
eligible manifest is built from these rows across all target models.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_retrieval_control_pool
import jsonl_logging as jl
from search_utils import ddg_search
import victim_scaffold as vs


MODEL_ID = os.environ.get("WB_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda"
DEVICE_MAP = os.environ.get("WB_DEVICE_MAP", DEVICE)
MAX_STEPS = int(os.environ.get("WB_MAX_STEPS", "40"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "450"))
DATASET_SRC = os.environ.get("DATASET_SRC", "all")
SAMPLE_MANIFEST = os.environ.get("WB_SAMPLE_MANIFEST")

RUN_ID = os.environ.get("WB_RUN_ID", "default")
STAGE = os.environ.get("WB_STAGE", "PAPER_CLEAN_BASELINE")
LOG_PATH = os.environ.get(
    "WB_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "whitebox_results", "clean_baseline.jsonl"),
)
REPO_COMMIT = jl.get_repo_commit()

PREMATURE = vs.PREMATURE
CONTINUE = {"SEARCH_CONTINUE", "THINK_TOOL"}


def _completion_step(decisions):
    for i, decision in enumerate(decisions):
        if decision in PREMATURE:
            return i + 1
    return None


def _n_search(decisions):
    return sum(1 for decision in decisions if decision == "SEARCH_CONTINUE")


def load_done_keys(log_path):
    done = set()
    if not os.path.exists(log_path):
        return done
    for row in jl.read_rows(log_path):
        if row.get("condition") == "clean":
            done.add((row.get("dataset"), row.get("example_id"), "clean"))
    return done


if __name__ == "__main__":
    print(
        f"=== loading baseline examples: DATASET_SRC={DATASET_SRC} "
        f"N_SAMPLE={N_SAMPLE} WB_SAMPLE_MANIFEST={SAMPLE_MANIFEST} ===",
        flush=True,
    )
    examples = load_retrieval_control_pool(
        n_sample=N_SAMPLE,
        dataset_src=DATASET_SRC,
        sample_manifest_path=SAMPLE_MANIFEST,
    )
    print(f"=== loaded {len(examples)} examples ===", flush=True)

    done_keys = load_done_keys(LOG_PATH)
    remaining = [
        ex for ex in examples
        if (ex["source"], ex["name"], "clean") not in done_keys
    ]
    print(f"=== resume: {len(done_keys)} done, {len(remaining)} remaining ===", flush=True)

    if not remaining:
        print("=== all clean baseline rows already written; exiting before model load ===",
              flush=True)
        sys.exit(0)

    print(f"=== loading {MODEL_ID} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE_MAP)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print("=== model loaded ===", flush=True)

    def log_row(**kw):
        return jl.write_row(
            LOG_PATH,
            run_id=RUN_ID,
            stage=STAGE,
            model=MODEL_ID,
            repo_commit=REPO_COMMIT,
            attack="WrongGapAssessment",
            method="WrongGapAssessment_CleanBaseline",
            **kw,
        )

    for idx, ex in enumerate(examples):
        dataset, example_id = ex["source"], ex["name"]
        resume_key = (dataset, example_id, "clean")
        if resume_key in done_keys:
            print(f"[{idx+1}/{len(examples)}] {dataset}/{example_id}: already logged",
                  flush=True)
            continue

        topic, body = ex["topic"], ex["body"]
        t0 = time.time()
        decisions, generations, _, step_trace = vs.rollout_n_steps(
            model, tok, DEVICE, topic, body,
            n_steps=MAX_STEPS, search_fn=ddg_search,
        )

        clean_step1 = decisions[0]
        completion_step = _completion_step(decisions)
        n_search = _n_search(decisions)
        eligible = clean_step1 in CONTINUE
        ineligibility_reason = None if eligible else "clean_already_premature"

        print(
            f"[{idx+1}/{len(examples)}] {dataset}/{example_id}: "
            f"{' -> '.join(decisions)} completion_step={completion_step} "
            f"searches={n_search} eligible={eligible}",
            flush=True,
        )

        log_row(
            dataset=dataset,
            example_id=example_id,
            question=topic,
            gold_answer=ex.get("answer"),
            condition="clean",
            perturbation_family=None,
            eligible=eligible,
            ineligibility_reason=ineligibility_reason,
            clean_observation=body,
            attacked_observation=None,
            perturbation=None,
            perturbation_tokens=None,
            forbidden_word_flag=None,
            gold_answer_in_injection=None,
            clean_next_action=clean_step1,
            attack_next_action=None,
            strict_success=None,
            inclusive_success=None,
            step_trace_clean=step_trace,
            step_trace_attack=None,
            tool_trace_clean=decisions,
            tool_trace_attack=None,
            completion_step_clean=completion_step,
            completion_step_attack=None,
            n_search_steps_clean=n_search,
            n_search_steps_attack=None,
            raw_generation_clean=generations[0],
            raw_generation_attack=None,
            parse_error=(clean_step1 == "OTHER"),
            wallclock_sec=round(time.time() - t0, 2),
        )
        done_keys.add(resume_key)
