"""
wrong_gap_assessment_cue_only.py — Wrong Gap Assessment Cue-Only experiment.

Evaluates the pure rhetorical effect of five cue suffixes appended to D1 alone,
with no second-hop bridge sentence (F2s).  This isolates the Rhetorical Effect from
the Information Effect captured by the FactOnly condition.

Observation forms
-----------------
  Clean          : D1                                (canonical, from WGA output)
  FactOnly       : D1 + first_sentence(D2)           (wrong_gap_assessment_factonly.py)
  Cue Only       : D1 + suffix                       [this script]
  Full WGA       : D1 + first_sentence(D2) + suffix  (wrong_gap_assessment.py)

Effect decomposition
--------------------
  Information Effect = PCR@k(FactOnly) − PCR@k(Clean)
  Rhetorical Effect  = PCR@k(CueOnly)  − PCR@k(Clean)

Manifest-driven design
----------------------
This script does NOT re-run the clean rollout.  It loads the canonical WGA output
(WB_WGA_PATH) and uses its clean rows as the ground truth for:
  - example manifest (exact dataset/example_id pairs, in WGA order)
  - clean eligibility
  - all clean trace fields (step_trace_clean, tool_trace_clean, etc.)

This ensures all four conditions share the same clean baseline with zero stochastic
drift across conditions.  The cue-only model is not launched before WB_WGA_PATH
for that model is complete.

Resumption
----------
On startup, any rows already written to WB_LOG_PATH are read and their
(dataset, example_id, condition) triples are added to a done_keys set.  The main
loop skips any triple already in done_keys, making SLURM preemption safe.

Condition keys written
----------------------
  cue_only_epistemic_confirmation
  cue_only_search_match_cue
  cue_only_source_coverage
  cue_only_epistemic_confirmation_source
  cue_only_search_match_source

Environment variables
---------------------
  WB_MODEL        HuggingFace model ID  (must match the model used for WB_WGA_PATH)
  WB_WGA_PATH     Path to the completed final_wga_{model}.jsonl  (required)
  WB_DEVICE_MAP   "cuda" or "auto"  (default: cuda)
  WB_MAX_STEPS    Max rollout steps per trajectory  (default: 40)
  N_SAMPLE        Examples per dataset — must match WGA job  (default: 450)
  DATASET_SRC     "all" or comma-separated datasets — must match WGA job  (default: all)
  WB_LOG_PATH     Output JSONL path
  WB_RUN_ID       Run identifier written to every row
  WB_STAGE        Stage label  (default: S1)
  DDG_CACHE_DIR   Path for DuckDuckGo result cache
  DDG_SLEEP       Seconds between uncached DDG calls  (default: 1.5)
  SMOKE_IDS       Optional: path to file of newline-separated example_ids for smoke test
"""

import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cue_definitions import PERTURBATION_FAMILIES
from data_utils import (
    load_retrieval_control_pool, is_ascii, phrase_valid,
    UNIVERSAL_PATTERNS, FACTONLY_PATTERNS,
)
import victim_scaffold as vs
import jsonl_logging as jl
from search_utils import ddg_search

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = os.environ.get("WB_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE      = "cuda"
DEVICE_MAP  = os.environ.get("WB_DEVICE_MAP", DEVICE)
MAX_STEPS   = int(os.environ.get("WB_MAX_STEPS", "40"))
N_SAMPLE    = int(os.environ.get("N_SAMPLE", "450"))
DATASET_SRC = os.environ.get("DATASET_SRC", "all")

RUN_ID      = os.environ.get("WB_RUN_ID", "default")
STAGE       = os.environ.get("WB_STAGE", "S1")
WGA_PATH    = os.environ.get("WB_WGA_PATH")
LOG_PATH    = os.environ.get(
    "WB_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "whitebox_results", "wrong_gap_assessment_cue_only.jsonl"),
)
SMOKE_IDS_FILE = os.environ.get("SMOKE_IDS")
REPO_COMMIT    = jl.get_repo_commit()

PREMATURE = vs.PREMATURE
CONTINUE  = {"SEARCH_CONTINUE", "THINK_TOOL"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _completion_step(decisions):
    for i, d in enumerate(decisions):
        if d in PREMATURE:
            return i + 1
    return None


def _n_search(decisions):
    return sum(1 for d in decisions if d == "SEARCH_CONTINUE")


def normalize_spacing(s):
    return re.sub(r"\s+", " ", s).strip()


# ── Canonical manifest loading ────────────────────────────────────────────────

def load_wga_manifest(wga_path):
    """Load eligible clean rows from the canonical WGA JSONL.

    Returns:
      manifest  : list[(dataset, example_id)] — eligible examples in WGA order
      wga_index : {(dataset, example_id): clean_row}
    """
    if not wga_path or not os.path.exists(wga_path):
        raise FileNotFoundError(
            f"WB_WGA_PATH={wga_path!r} not found. "
            "Set WB_WGA_PATH to the completed final_wga_{{model}}.jsonl for this model."
        )
    wga_index = {}
    manifest = []
    seen = set()
    with open(wga_path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("condition") != "clean":
                continue
            key = (row["dataset"], row["example_id"])
            wga_index[key] = row
            if row.get("eligible") and key not in seen:
                manifest.append(key)
                seen.add(key)
    print(f"  WGA manifest: {len(wga_index)} clean rows, {len(manifest)} eligible", flush=True)
    return manifest, wga_index


# ── Dataset lookup ────────────────────────────────────────────────────────────

def build_dataset_lookup(examples):
    return {(ex["source"], ex["name"]): ex for ex in examples}


# ── Resumption ────────────────────────────────────────────────────────────────

def load_done_keys(log_path):
    done = set()
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    r = json.loads(raw)
                    done.add((r["dataset"], r["example_id"], r["condition"]))
                except Exception:
                    pass
        print(f"  Resumption: {len(done)} rows already written", flush=True)
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load canonical WGA manifest ──────────────────────────────────────────
    print(f"=== loading WGA manifest from {WGA_PATH} ===", flush=True)
    manifest, wga_index = load_wga_manifest(WGA_PATH)

    # Optional smoke-test filter: restrict to listed example IDs only
    if SMOKE_IDS_FILE:
        with open(SMOKE_IDS_FILE) as fh:
            smoke_ids = {line.strip() for line in fh if line.strip()}
        manifest = [(ds, eid) for ds, eid in manifest if eid in smoke_ids]
        print(f"  SMOKE_IDS filter: {len(manifest)} examples", flush=True)

    # ── Load dataset for body/second_para/topic/gold_answer ─────────────────
    print(f"=== loading dataset: DATASET_SRC={DATASET_SRC} N_SAMPLE={N_SAMPLE} ===", flush=True)
    examples = load_retrieval_control_pool(n_sample=N_SAMPLE, dataset_src=DATASET_SRC)
    ds_lookup = build_dataset_lookup(examples)
    print(f"  dataset loaded: {len(examples)} examples", flush=True)

    # Cross-reference check
    missing = [(ds, eid) for ds, eid in manifest if (ds, eid) not in ds_lookup]
    if missing:
        print(f"  WARNING: {len(missing)} manifest entries not found in dataset lookup", flush=True)
        for ds, eid in missing[:5]:
            print(f"    missing: ({ds}, {eid})", flush=True)

    # ── Resumption ───────────────────────────────────────────────────────────
    done_keys = load_done_keys(LOG_PATH)

    remaining = sum(
        1 for ds, eid in manifest
        for fam in PERTURBATION_FAMILIES
        if (ds, eid, f"cue_only_{fam}") not in done_keys
    )
    print(f"  Remaining: {remaining} cue-only rollouts to run", flush=True)

    if remaining == 0:
        print("=== All rows already written. Nothing to do. ===", flush=True)
        import sys
        sys.exit(0)

    # ── Model loading ────────────────────────────────────────────────────────
    print(f"=== loading {MODEL_ID} ===", flush=True)
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE_MAP)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("=== model loaded ===", flush=True)

    def log_row(**kw):
        jl.write_row(LOG_PATH, run_id=RUN_ID, stage=STAGE, model=MODEL_ID,
                     repo_commit=REPO_COMMIT, attack="WrongGapAssessment",
                     method="WrongGapAssessment_CueOnly_NaturalRollout", **kw)

    n_families   = len(PERTURBATION_FAMILIES)
    n_total      = len(manifest)
    rows_written = 0

    print(f"\n=== WrongGapAssessment CueOnly — {n_total} examples × {n_families} cues "
          f"× up to {MAX_STEPS} steps ===", flush=True)

    # ── Main loop ────────────────────────────────────────────────────────────
    for idx, (ds, eid) in enumerate(manifest):
        clean_row = wga_index[(ds, eid)]
        ex        = ds_lookup.get((ds, eid))

        if ex is None:
            print(f"  [{idx+1}/{n_total}] SKIP ({ds},{eid}): not in dataset lookup", flush=True)
            continue

        # Use the canonical body from the WGA clean row (zero drift)
        body  = clean_row["clean_observation"]
        topic = ex["topic"]
        gold  = ex.get("answer")

        # Soft integrity check: warn if dataset body diverges from WGA body
        if normalize_spacing(ex.get("body", "")) != normalize_spacing(body):
            print(f"  WARNING: body mismatch ({ds},{eid}) — using WGA canonical", flush=True)

        print(f"\n{'='*80}\n[{idx+1}/{n_total}] [{ds}] {eid}: {topic!r}", flush=True)

        for family_name, suffix in PERTURBATION_FAMILIES.items():
            cond_key   = f"cue_only_{family_name}"
            resume_key = (ds, eid, cond_key)
            if resume_key in done_keys:
                continue

            # Suffix validation (defensive — hardcoded suffixes always pass)
            if (not is_ascii(suffix)
                    or not phrase_valid(suffix, UNIVERSAL_PATTERNS)
                    or not phrase_valid(suffix, FACTONLY_PATTERNS)):
                print(f"  SKIP {cond_key}: suffix failed validation", flush=True)
                log_row(
                    dataset=ds, example_id=eid, question=topic, gold_answer=gold,
                    perturbation_family=family_name, condition=cond_key,
                    eligible=False, ineligibility_reason="banned_vocabulary",
                    clean_observation=body, attacked_observation=None,
                    perturbation=suffix, perturbation_tokens=None, forbidden_word_flag=True,
                    gold_answer_in_injection=None,
                    clean_next_action=clean_row["clean_next_action"],
                    attack_next_action=None, strict_success=None, inclusive_success=None,
                    step_trace_clean=clean_row["step_trace_clean"], step_trace_attack=None,
                    tool_trace_clean=clean_row["tool_trace_clean"], tool_trace_attack=None,
                    completion_step_clean=clean_row["completion_step_clean"],
                    completion_step_attack=None,
                    n_search_steps_clean=clean_row["n_search_steps_clean"],
                    n_search_steps_attack=None,
                    raw_generation_clean=clean_row["raw_generation_clean"],
                    raw_generation_attack=None,
                    parse_error=False, wallclock_sec=0.0,
                )
                done_keys.add(resume_key)
                continue

            patched  = body + " " + suffix
            gold_in  = gold.lower() in patched.lower() if gold else None

            t0 = time.time()
            atk_decisions, atk_gens, _, atk_trace = vs.rollout_n_steps(
                model, tok, DEVICE, topic, patched,
                n_steps=MAX_STEPS, search_fn=ddg_search)

            atk_step1     = atk_decisions[0]
            atk_comp_step = _completion_step(atk_decisions)
            atk_n_search  = _n_search(atk_decisions)
            pcr1          = atk_comp_step == 1
            strict_s      = atk_step1 == "RESEARCH_COMPLETE"
            atk_trace_str = " → ".join(atk_decisions)

            print(f"  {cond_key:<48} ({len(atk_decisions)} steps): {atk_trace_str}"
                  f"  PCR@1={pcr1}  completion_step={atk_comp_step}"
                  f"  searches={atk_n_search}  gold_in={gold_in}", flush=True)

            log_row(
                dataset=ds, example_id=eid, question=topic, gold_answer=gold,
                perturbation_family=family_name, condition=cond_key,
                eligible=True, ineligibility_reason=None,
                clean_observation=body,
                attacked_observation=patched,
                perturbation=suffix,
                perturbation_tokens=len(tok(suffix, add_special_tokens=False).input_ids),
                forbidden_word_flag=False,
                gold_answer_in_injection=gold_in,
                clean_next_action=clean_row["clean_next_action"],
                attack_next_action=atk_step1,
                strict_success=strict_s,
                inclusive_success=pcr1,
                step_trace_clean=clean_row["step_trace_clean"],
                step_trace_attack=atk_trace,
                tool_trace_clean=clean_row["tool_trace_clean"],
                tool_trace_attack=atk_decisions,
                completion_step_clean=clean_row["completion_step_clean"],
                completion_step_attack=atk_comp_step,
                n_search_steps_clean=clean_row["n_search_steps_clean"],
                n_search_steps_attack=atk_n_search,
                raw_generation_clean=clean_row["raw_generation_clean"],
                raw_generation_attack=atk_gens[0],
                parse_error=(atk_step1 == "OTHER"),
                wallclock_sec=round(time.time() - t0, 2),
            )
            done_keys.add(resume_key)
            rows_written += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"=== DONE — {rows_written} rows written to {LOG_PATH} ===", flush=True)

    def _pcr_at(comp_steps, k):
        if not comp_steps:
            return 0.0
        return 100.0 * sum(s is not None and s <= k for s in comp_steps) / len(comp_steps)

    # Re-read for summary
    if os.path.exists(LOG_PATH):
        all_rows = []
        with open(LOG_PATH) as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    all_rows.append(json.loads(raw))
        KS = [1, 2, 3, 5, 10, 20]
        hdr = "  ".join(f"PCR@{k}" for k in KS)
        print(f"\n  {'CONDITION':<52} n    {hdr}", flush=True)
        for fam_name in PERTURBATION_FAMILIES:
            cond_key = f"cue_only_{fam_name}"
            comp = [r.get("completion_step_attack")
                    for r in all_rows
                    if r.get("condition") == cond_key and r.get("eligible")]
            n = len(comp)
            if n == 0:
                print(f"  {cond_key:<52} n=0", flush=True)
                continue
            vals = "  ".join(f"{_pcr_at(comp, k):5.1f}%" for k in KS)
            print(f"  {cond_key:<52} n={n:<4} {vals}", flush=True)
