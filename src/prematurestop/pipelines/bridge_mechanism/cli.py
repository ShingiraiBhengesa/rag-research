"""Phase driver for the dynamic bridge-cue pilot. Every phase is resumable.

Phases (run in this order per split):
  decompose   attacker   step 1, H per question; shared by both victims
  anchor      victim     unforced NaturalClean anchor run -> clean d1
  cues        attacker   steps 2-5 per (victim, question): bridge + K variants
  select      CPU        step 6 with the lambda file (frozen for held-out)
  arms        victim     the six replayed arms
  analyze     CPU        labels, PCR, contrasts, trajectories, figures, audit
Support:
  audit-export / fit-lambda / freeze / verify-freeze

Each unit is appended to a JSONL under --state-dir with fsync before the next
unit starts; a restarted pod skips completed keys.

Sequence: dev decompose/anchor/cues -> audit-export, rate, fit-lambda ->
freeze -> dev select/arms (end-to-end check) -> full (canonical n450).
`select`/`arms` on dev and full, and every phase on full, refuse to run
unless verify-freeze passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from prematurestop.pipelines.bridge_mechanism import config as C
from prematurestop.pipelines.bridge_mechanism import freeze as F
from prematurestop.pipelines.bridge_mechanism.paths import DEFAULT_SPLITS, resolve_state_path, split_paths
from prematurestop.paths import EXPERIMENT_E2, RESULTS_E1_QWEN_LLAMA
from prematurestop.infra.store import append_jsonl, done_keys, index_by, read_jsonl

EXP_DIR = EXPERIMENT_E2
FROZEN_LAMBDA = EXP_DIR / "lambda" / "lambda_frozen.json"
FREEZE_MANIFEST = EXP_DIR / "freeze_manifest.json"


def _log(**kw) -> None:
    print(json.dumps({"ts": round(time.time(), 1), **kw}), flush=True)


def _questions(args, dataset: str, *, sharded: bool = True) -> list[dict]:
    """Manifest questions, optionally one shard of them. Shard i of N takes every
    N-th question in manifest order (a fixed partition, never outcome-based)."""
    qs = read_jsonl(split_paths(Path(args.split_dir), dataset, args.split)["manifest"])
    qs = qs[: args.max_questions] if args.max_questions else qs
    if sharded and args.num_shards > 1:
        qs = qs[args.shard:: args.num_shards]
    return qs


def _state(args, kind: str, name: str) -> Path:
    """This process's checkpoint file. Each shard writes its own file, so
    concurrent shard pods never append to the same file on the shared PVC.
    Decompositions are produced by the single unsharded decompose job."""
    tag = f".s{args.shard:03d}of{args.num_shards:03d}" if args.num_shards > 1 and kind != "decompositions" else ""
    return Path(args.state_dir) / args.split / kind / f"{name}{tag}.jsonl"


def _read_all(args, kind: str, name: str) -> list[dict]:
    """All records of a (kind, name) across every shard file, de-duplicated by key."""
    base = Path(args.state_dir) / args.split / kind
    files = sorted([base / f"{name}.jsonl", *base.glob(f"{name}.s[0-9]*of[0-9]*.jsonl")])
    return list(index_by(r for f in files for r in read_jsonl(f)).values())


def _victim_ctx(args, victim: str):
    from prematurestop.pipelines.bridge_mechanism.runner import VictimContext
    from prematurestop.system.runner import V3ModelConfig

    spec = C.VICTIM_MODELS[victim]
    model = V3ModelConfig(model_name=spec["model_id"], base_url=args.victim_base_url, provider="openai",
                          api_key=args.api_key, research_max_tokens=C.RESEARCH_MAX_TOKENS,
                          compression_max_tokens=C.COMPRESSION_MAX_TOKENS,
                          final_report_max_tokens=C.FINAL_REPORT_MAX_TOKENS,
                          temperature=C.VICTIM_TEMPERATURE, seed=C.VICTIM_SEED)
    return VictimContext(victim_key=victim, model=model, traces_root=Path(args.state_dir) / args.split / "traces")


def _attacker(args):
    from prematurestop.cues.dynamic.llm import OpenAICompatClient

    return OpenAICompatClient(base_url=args.attacker_base_url, model=C.VICTIM_MODELS[C.ATTACKER_MODEL_KEY]["model_id"])


def _token_counter(args):
    from prematurestop.cues.dynamic.text_checks import LocalTokenizerCounter, VllmTokenCounter

    if args.tokenizer_json:
        return LocalTokenizerCounter(Path(args.tokenizer_json))
    return VllmTokenCounter(args.attacker_base_url, C.VICTIM_MODELS[C.ATTACKER_MODEL_KEY]["model_id"])


# Deterministic VICTIM-side failures (temperature 0: a retry reproduces them).
# Added during the full run (2026-10-12) after 9 llama8b anchor runs hit the
# victim's output-length limit inside ODR. They are recorded once as a question-
# level victim failure (excluded from comparisons, counted in reports), as the
# canonical analysis treats EXECUTION_FAILURE; real infrastructure errors still
# retry and block the phase.
VICTIM_FAILURE_ERRORS = ("LengthFinishReasonError",)
VICTIM_FAILURE = "victim_failure"


class VictimFailure:
    def __init__(self, exc: Exception):
        self.error = repr(exc)


def _retry(fn, *, unit: str, failures: Path):
    last = None
    for attempt in range(C.MAX_TECHNICAL_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # infrastructure only; attack failures are records, not exceptions
            if type(exc).__name__ in VICTIM_FAILURE_ERRORS:
                _log(event=VICTIM_FAILURE, unit=unit, error=repr(exc)[:300])
                return VictimFailure(exc)
            last = exc
            _log(event="retry", unit=unit, attempt=attempt, error=repr(exc)[:300])
            time.sleep(min(2 ** attempt * 5, 60))
    append_jsonl(failures, {"key": unit, "error": repr(last), "ts": time.time()})
    return None


def _split_manifests(args) -> list[Path]:
    return [split_paths(Path(args.split_dir), ds, sp)["manifest"] for sp in C.FROZEN_SPLITS for ds in C.PILOT_DATASETS]


def _require_freeze(args) -> None:
    if args.split not in C.FROZEN_SPLITS:
        return
    problems = F.verify(FREEZE_MANIFEST, lambda_path=FROZEN_LAMBDA, split_manifests=_split_manifests(args))
    if problems:
        sys.exit(f"{args.split} refused, freeze verification failed:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------- phases
def _require_freeze_for_full(args) -> None:
    if args.split == "full":
        _require_freeze(args)


def phase_decompose(args) -> None:
    from prematurestop.cues.dynamic.bridge import decompose

    _require_freeze_for_full(args)
    client = _attacker(args)
    for ds in args.datasets:
        out = _state(args, "decompositions", ds)
        done = done_keys(out)
        for q in _questions(args, ds, sharded=False):
            key = f"{ds}|{q['example_id']}"
            if key in done:
                continue
            rec = _retry(lambda: decompose(client, q["question"]), unit=key, failures=_state(args, "failures", "decompose"))
            if rec is not None:
                append_jsonl(out, {"key": key, "dataset": ds, "example_id": q["example_id"], **rec})
                _log(phase="decompose", key=key, valid=rec["graph"]["valid"])


def phase_anchor(args) -> None:
    from prematurestop.pipelines.bridge_mechanism.runner import run_anchor

    _require_freeze_for_full(args)
    ctx = _victim_ctx(args, args.victim)
    for ds in args.datasets:
        paths = split_paths(Path(args.split_dir), ds, args.split)
        out = _state(args, "anchors", f"{args.victim}_{ds}")
        done = done_keys(out)
        for q in _questions(args, ds):
            key = f"{args.victim}|{ds}|{q['example_id']}"
            if key in done:
                continue
            rec = _retry(lambda: run_anchor(ctx, dataset=ds, q=q, paths=paths), unit=key,
                         failures=_state(args, "failures", f"anchor_{args.victim}"))
            if isinstance(rec, VictimFailure):
                append_jsonl(out, {"key": key, "victim": args.victim, "dataset": ds, "example_id": q["example_id"],
                                   "question": q["question"], "status": VICTIM_FAILURE, "error": rec.error})
                continue
            if rec is not None:
                append_jsonl(out, {"key": key, **rec})
                _log(phase="anchor", key=key, prefix_ok=rec["clean_observation_matches_anchor"])


def phase_cues(args) -> None:
    from prematurestop.cues.dynamic.bridge import build_bridge
    from prematurestop.cues.dynamic.generation import STATUS_NOT_ATTEMPTED, generate_dynamic_cue
    from prematurestop.cues.dynamic.entail import NliEntailer
    from prematurestop.cues.dynamic.unsupported import build_unsupported

    _require_freeze_for_full(args)
    client, counter, entailer = _attacker(args), _token_counter(args), NliEntailer()
    for ds in args.datasets:
        decomp = index_by(read_jsonl(_state(args, "decompositions", ds)))
        anchors = read_jsonl(_state(args, "anchors", f"{args.victim}_{ds}"))
        manifest = {q["example_id"]: q for q in _questions(args, ds)}
        b_out = _state(args, "bridge_selection", f"{args.victim}_{ds}")
        c_out = _state(args, "generated_cues", f"{args.victim}_{ds}")
        b_done, c_done = done_keys(b_out), done_keys(c_out)
        for a in anchors:
            if a["example_id"] not in manifest:
                continue
            d = decomp.get(f"{ds}|{a['example_id']}")
            if d is None:
                sys.exit(f"missing decomposition for {ds}|{a['example_id']}; run the decompose phase first")
            q = manifest[a["example_id"]]
            bkey = a["key"]
            if bkey not in b_done and a.get("status") == VICTIM_FAILURE:
                append_jsonl(b_out, {"key": bkey, "victim": args.victim, "dataset": ds, "example_id": a["example_id"],
                                     "question": q["question"], "status": "ANCHOR_VICTIM_FAILURE", "graph": {},
                                     "facts": [], "candidates": [], "anchor": None, "documents": []})
                b_done.add(bkey)
            if bkey not in b_done:
                b = _retry(lambda: build_bridge(client, entailer, question=q["question"],
                                                observation=a["clean_observation"], decomposition=d),
                           unit=bkey, failures=_state(args, "failures", f"bridge_{args.victim}"))
                if b is None:
                    continue
                append_jsonl(b_out, {"key": bkey, "victim": args.victim, "dataset": ds,
                                     "example_id": a["example_id"], "question": q["question"], **b})
                b_done.add(bkey)
                _log(phase="bridge", key=bkey, status=b["status"])
            b = index_by(read_jsonl(b_out))[bkey]
            gen_cfg = {"client": client, "token_counter": counter, "entailer": entailer,
                       "answer_slot": b["graph"].get("answer_slot_description", ""),
                       "gold_answers": [q["answer"], *q.get("answer_aliases", [])]}
            for unit, (mode, cond) in C.UNITS.items():
                ckey = f"{bkey}|{unit}"
                if ckey in c_done:
                    continue
                base = {"key": ckey, "victim": args.victim, "dataset": ds, "example_id": a["example_id"],
                        "unit": unit, "mode": mode, "condition": cond,
                        "generator_model": C.VICTIM_MODELS[C.ATTACKER_MODEL_KEY]}
                if b["status"] != "OK":
                    append_jsonl(c_out, {**base, "status": STATUS_NOT_ATTEMPTED, "bridge_status": b["status"],
                                         "variants": []})
                    c_done.add(ckey)
                    continue
                if mode == "unsupported":
                    src = index_by(read_jsonl(c_out)).get(f"{bkey}|bridge_suff")
                    if src is None:
                        continue   # bridge_suff failed this round; retried on rerun
                    if src["status"] != "OK":
                        append_jsonl(c_out, {**base, "status": "SOURCE_POOL_INVALID", "variants": []})
                        c_done.add(ckey)
                        continue
                    fn = lambda: build_unsupported(client=client, entailer=entailer, token_counter=counter,
                                                   question=q["question"], observation=a["clean_observation"],
                                                   bridge_rec=b, bridge_cue_rec=src, generation_config=gen_cfg)
                else:
                    fn = lambda: generate_dynamic_cue(q["question"], a["clean_observation"], b, cond, gen_cfg,
                                                      mode=mode)
                rec = _retry(fn, unit=ckey, failures=_state(args, "failures", f"cues_{args.victim}"))
                if rec is not None:
                    append_jsonl(c_out, {**base, **rec, "bridge_entity": b["anchor"]["bridge_entity_tail"],
                                         "bridge_fact": b["anchor"]["bridge_fact"]})
                    c_done.add(ckey)
                    _log(phase="cues", key=ckey, status=rec["status"], n_valid=rec.get("n_valid"))


def _lambda(args) -> dict:
    from prematurestop.cues.dynamic.lambda_fit import EQUAL_LAMBDA

    if args.split in C.FROZEN_SPLITS or args.lambda_file:
        path = Path(args.lambda_file) if args.lambda_file else FROZEN_LAMBDA
        return json.loads(path.read_text(encoding="utf-8"))
    if args.split == "smoke" and args.provisional_equal_lambda:
        return {"lambda": EQUAL_LAMBDA, "lambda_hash": "provisional_equal_smoke_only", "source": "provisional"}
    sys.exit("select needs --lambda-file or --provisional-equal-lambda (smoke only)")


def select_for_question(cue_recs_by_unit: dict, lambdas: dict) -> dict:
    """{arm: {"selected": variant|None, "reason": str}} for every dynamic arm of one question.
    Pure function of stored cue records and lambda; never reads victim output."""
    from prematurestop.cues.dynamic.generation import select, select_first_valid

    keep = ("index", "seed", "cue", "features", "fixed_similarity", "token_count", "checks")
    out: dict = {}
    dynamic = [(arm, spec) for arm, spec in C.ARMS.items() if spec["suffix_source"] == "dynamic"]
    for arm, spec in sorted(dynamic, key=lambda x: x[1]["selection"].startswith("counterpart_of:")):
        rec = cue_recs_by_unit.get(spec["unit"])
        if rec is None or rec.get("status") not in ("OK",):
            out[arm] = {"selected": None, "reason": (rec or {}).get("status", "NO_CUE_RECORD")}
            continue
        rule = spec["selection"]
        variants = rec.get("variants", [])
        if rule == "lambda":
            u = select(variants, lambdas)
        elif rule == "first_valid":
            u = select_first_valid(variants)
        elif rule.startswith("counterpart_of:"):
            parent = out.get(rule.split(":", 1)[1], {}).get("selected")
            u = None if parent is None else next(
                (v for v in variants if v["index"] == parent["index"] and v["valid"]), None)
            if u is None:
                out[arm] = {"selected": None, "reason": "PARENT_UNSELECTED" if parent is None
                            else "COUNTERPART_OF_SELECTED_VARIANT_INVALID"}
                continue
        else:
            raise ValueError(f"unknown selection rule {rule}")
        out[arm] = {"selected": None if u is None else {k: u.get(k) for k in keep},
                    "reason": "" if u is not None else "INVALID_CUE"}
    return out


def phase_select(args) -> None:
    _require_freeze(args)
    lam = _lambda(args)
    for ds in args.datasets:
        out = _state(args, "selections", f"{args.victim}_{ds}")
        done = done_keys(out)
        by_q: dict = {}
        for c in read_jsonl(_state(args, "generated_cues", f"{args.victim}_{ds}")):
            by_q.setdefault(c["example_id"], {})[c["unit"]] = c
        for ex, units in by_q.items():
            for arm, sel in select_for_question(units, lam["lambda"]).items():
                key = f"{args.victim}|{ds}|{ex}|{arm}"
                if key in done:
                    continue
                append_jsonl(out, {"key": key, "victim": args.victim, "dataset": ds, "example_id": ex, "arm": arm,
                                   "lambda": lam["lambda"], "lambda_hash": lam["lambda_hash"], **sel})


def phase_arms(args) -> None:
    from prematurestop.pipelines.bridge_mechanism.runner import plan_arms, run_arm

    _require_freeze(args)
    ctx = _victim_ctx(args, args.victim)
    dynamic_arms = {a for a, sp in C.ARMS.items() if sp["suffix_source"] == "dynamic"}
    for ds in args.datasets:
        paths = split_paths(Path(args.split_dir), ds, args.split)
        manifest = {q["example_id"]: q for q in _questions(args, ds)}
        bridges = index_by(read_jsonl(_state(args, "bridge_selection", f"{args.victim}_{ds}")))
        sels = read_jsonl(_state(args, "selections", f"{args.victim}_{ds}"))
        out = _state(args, "arms", f"{args.victim}_{ds}")
        done = done_keys(out)
        for a in read_jsonl(_state(args, "anchors", f"{args.victim}_{ds}")):
            if a["example_id"] not in manifest:
                continue
            if a.get("status") == VICTIM_FAILURE:
                for arm, spec in C.ARMS.items():
                    key = f"{a['key']}|{arm}"
                    if key not in done:
                        append_jsonl(out, {"key": key, "victim": args.victim, "dataset": ds,
                                           "example_id": a["example_id"], "arm": arm,
                                           "runtime_condition": spec["runtime_condition"],
                                           "suffix_source": spec["suffix_source"], "status": "not_run",
                                           "skip_reason": "ANCHOR_VICTIM_FAILURE", "trace_path": "", "suffix": ""})
                continue
            b = bridges.get(a["key"])
            if b is None:
                sys.exit(f"no bridge record for {a['key']}; run cues first")
            selected = {s["arm"]: s for s in sels if s["example_id"] == a["example_id"]}
            if b["status"] == "OK" and set(selected) != dynamic_arms:
                sys.exit(f"incomplete selection for {a['key']}; run select first")
            for plan in plan_arms(b, selected):
                key = f"{a['key']}|{plan.arm}"
                if key in done:
                    continue
                rec = _retry(lambda: run_arm(ctx, plan, dataset=ds, q=manifest[a["example_id"]], paths=paths,
                                             anchor_rec=a),
                             unit=key, failures=_state(args, "failures", f"arms_{args.victim}"))
                if isinstance(rec, VictimFailure):
                    rec = {"victim": args.victim, "dataset": ds, "example_id": a["example_id"], "arm": plan.arm,
                           "runtime_condition": plan.runtime_condition, "suffix_source": plan.suffix_source,
                           "status": VICTIM_FAILURE, "error": rec.error, "trace_path": "", "suffix": plan.suffix or ""}
                if rec is not None:
                    append_jsonl(out, {"key": key, **rec})
                    _log(phase="arms", key=key, status=rec["status"], injection_ok=rec.get("injection_ok"))


def phase_analyze(args) -> None:
    from prematurestop.pipelines.bridge_mechanism import analysis as A
    from prematurestop.pipelines.bridge_mechanism.outcomes import label_run, trajectory_rows

    res = Path(args.results_dir) / args.split if args.results_dir else EXP_DIR
    sub = (lambda name: res / name) if args.results_dir else (lambda name: EXP_DIR / name / args.split)
    victims = args.victims or list(C.PILOT_VICTIMS)
    labels, traj, tools, bridges, cues, arms, sels = [], [], [], [], [], [], {}
    for victim in victims:
        for ds in args.datasets:
            manifest = {q["example_id"]: q for q in read_jsonl(split_paths(Path(args.split_dir), ds, args.split)["manifest"])}
            bridges += _read_all(args, "bridge_selection", f"{victim}_{ds}")
            cues += _read_all(args, "generated_cues", f"{victim}_{ds}")
            for s in _read_all(args, "selections", f"{victim}_{ds}"):
                sels[(s["victim"], s["dataset"], s["example_id"], s["arm"])] = s["selected"]
            for r in _read_all(args, "arms", f"{victim}_{ds}"):
                arms.append(r)
                lab = label_run(r, manifest[r["example_id"]])
                labels.append(lab)
                rows = trajectory_rows(r, lab)
                traj += rows
                tools += [x for x in rows if any(t.get("queries") for t in x["tool_arguments"])]
    A.write_csv(sub("trajectories") / "run_labels.csv", labels)
    A.write_csv(sub("trajectories") / "trajectory_steps.csv", traj)
    A.write_csv(sub("tool_traces") / "search_calls.csv", tools)
    curves = A.pcr_curves(labels)
    contrasts = A.paired_contrasts(labels)
    premature = A.primary_outcome_contrasts(labels)
    stopped_k = A.stopped_by_k_contrasts(labels)
    gate = A.baseline_gate(_anchor_labels(args, victims, args.datasets))
    primary = A.primary_by_cell(gate, premature, stopped_k)
    secondary = A.secondary_by_arm(labels)
    A.write_csv(sub("diagnostics") / "baseline_gate.csv", gate)
    A.write_csv(sub("pcr") / "premature_stop_contrasts.csv", premature)
    A.write_csv(sub("pcr") / f"stopped_by_step_{C.GATE_FALLBACK_PCR_K}_contrasts.csv", stopped_k)
    A.write_csv(sub("pcr") / "primary_by_cell.csv", primary)
    A.write_csv(sub("diagnostics") / "outcomes_by_arm.csv", secondary)
    A.write_csv(sub("pcr") / "pcr_curves.csv", curves)
    A.write_csv(sub("pcr") / "pcr_paired_contrasts.csv", contrasts)
    A.write_csv(sub("diagnostics") / "outcome_labels_by_arm.csv", A.outcome_table(labels))
    A.write_csv(sub("diagnostics") / "answer_accuracy_by_arm.csv", A.answer_accuracy_table(labels))
    A.write_csv(sub("diagnostics") / "evidence_coverage_by_arm.csv", A.evidence_summary_table(labels))
    A.write_csv(sub("diagnostics") / "paired_answer_and_evidence.csv", A.paired_answer_and_evidence(labels))
    if args.split == "full":
        A.write_csv(sub("diagnostics") / "replication_vs_canonical.csv", A.replication_vs_canonical(
            labels, RESULTS_E1_QWEN_LLAMA / "data" / "pcr_run_level.csv"))
    A.write_csv(sub("diagnostics") / "bridge_and_cue_coverage.csv", A.coverage_table(bridges, cues, arms, {
        ds: len(read_jsonl(split_paths(Path(args.split_dir), ds, args.split)["manifest"])) for ds in args.datasets}))
    A.write_csv(sub("diagnostics") / "evidence_coverage.csv",
                [{**{k: l[k] for k in ("victim", "dataset", "example_id", "arm")}, **(l.get("evidence") or {})}
                 for l in labels if l.get("evidence")])
    A.write_csv(sub("generated_cues") / "cue_audit.csv", A.cue_audit_rows(bridges, cues, sels))
    A.write_csv(sub("bridge_selection") / "bridge_provenance.csv", [
        {"victim": b["victim"], "dataset": b["dataset"], "example_id": b["example_id"], "question": b["question"],
         "status": b["status"], **{k: (b.get("anchor") or {}).get(k) for k in (
             "bridge_entity_head", "bridge_entity_tail", "bridge_fact", "relation", "source_document_id",
             "source_document_title", "bridge_span", "reasoning_chain_position", "edge_tail_is_answer_slot",
             "entail_prob")}} for b in bridges])
    A.plot_pcr(curves, sub("figures"))
    sub("summaries").mkdir(parents=True, exist_ok=True)
    (sub("summaries") / "PILOT_SUMMARY.md").write_text(A.summarize(
        labels, contrasts, A.answer_accuracy_table(labels), A.paired_answer_and_evidence(labels),
        A.evidence_summary_table(labels), primary, secondary), encoding="utf-8")
    _log(phase="analyze", runs=len(labels), out=str(res))


def phase_audit_export(args) -> None:
    """Rating sheet for lambda fitting: every VALID dev variant, blank rating column."""
    from prematurestop.pipelines.bridge_mechanism import analysis as A
    from prematurestop.cues.dynamic.lambda_fit import flatten_variants

    rows = []
    for victim in args.victims or C.PILOT_VICTIMS:
        for ds in args.datasets:
            bridges = index_by(_read_all(args, "bridge_selection", f"{victim}_{ds}"))
            for v in flatten_variants(_read_all(args, "generated_cues", f"{victim}_{ds}")):
                # lambda only ever selects within the bridge/entity pools; unsupported
                # counterparts inherit the bridge choice and are never rated
                if not v["valid"] or v.get("unit") == "unsupported_suff":
                    continue
                b = bridges.get(f"{victim}|{ds}|{v['example_id']}", {})
                rows.append({"victim": victim, "dataset": ds, "example_id": v["example_id"],
                             "unit": v.get("unit"), "condition": v["condition"], "variant_index": v["index"],
                             "question": b.get("question"), "bridge_fact": (b.get("anchor") or {}).get("bridge_fact"),
                             "cue_text": v["cue"], "rating": "", "rater": ""})
    out = Path(args.out)
    A.write_csv(out, rows)
    _log(phase="audit-export", n=len(rows), out=str(out))


def phase_fit_lambda(args) -> None:
    from prematurestop.cues.dynamic import lambda_fit as L

    variants = []
    for victim in args.victims or C.PILOT_VICTIMS:
        for ds in args.datasets:
            variants += L.flatten_variants(_read_all(args, "generated_cues", f"{victim}_{ds}"))
    ratings, raters = L.load_ratings(Path(args.ratings))
    res = L.fit(variants, ratings)
    payload = L.write_lambda(FROZEN_LAMBDA, res, source=f"manual audit ratings ({args.ratings})", raters=raters)
    _log(phase="fit-lambda", **{k: payload[k] for k in ("lambda", "lambda_hash", "n_groups", "pairwise_concordance")})


def phase_freeze(args) -> None:
    m = F.build_manifest(lambda_path=FROZEN_LAMBDA, split_manifests=_split_manifests(args))
    F.write(FREEZE_MANIFEST, m)
    _log(phase="freeze", manifest_hash=m["manifest_hash"], path=str(FREEZE_MANIFEST))


def phase_verify_freeze(args) -> None:
    problems = F.verify(FREEZE_MANIFEST, lambda_path=FROZEN_LAMBDA, split_manifests=_split_manifests(args))
    _log(phase="verify-freeze", ok=not problems, problems=problems)
    if problems:
        sys.exit(1)


def _anchor_labels(args, victims, datasets) -> list[dict]:
    from prematurestop.pipelines.bridge_mechanism.outcomes import label_run

    labels = []
    for victim in victims:
        for ds in datasets:
            manifest = {q["example_id"]: q for q in read_jsonl(split_paths(Path(args.split_dir), ds, args.split)["manifest"])}
            for a in _read_all(args, "anchors", f"{victim}_{ds}"):
                failed = a.get("status") == VICTIM_FAILURE
                rec = {"victim": victim, "dataset": ds, "example_id": a["example_id"], "arm": "anchor_clean",
                       "runtime_condition": "NaturalClean", "status": VICTIM_FAILURE if failed else "ran",
                       "trace_path": "" if failed else a["anchor_trace_path"],
                       "run_id": a.get("anchor_run_id", ""), "injection_ok": None}
                labels.append(label_run(rec, manifest[a["example_id"]]))
    return labels


def phase_baseline_gate(args) -> None:
    """Zhen point 4: is there an effect to measure? Clean anchor runs only."""
    from prematurestop.pipelines.bridge_mechanism import analysis as A

    rows = A.baseline_gate(_anchor_labels(args, args.victims or list(C.PILOT_VICTIMS), args.datasets))
    out = (Path(args.results_dir) / args.split if args.results_dir else EXP_DIR / "diagnostics" / args.split)
    A.write_csv(out / "baseline_gate.csv", rows)
    for r in rows:
        _log(phase="baseline-gate", **r)
    if any(r["gate"] != "PASS" for r in rows):
        sys.exit(2)


def phase_judge_audit_export(args) -> None:
    from prematurestop.pipelines.bridge_mechanism import audit

    bridges, cues = [], []
    for victim in args.victims or C.PILOT_VICTIMS:
        for ds in args.datasets:
            bridges += _read_all(args, "bridge_selection", f"{victim}_{ds}")
            cues += _read_all(args, "generated_cues", f"{victim}_{ds}")
    paths = audit.export_sheets(bridges, cues, Path(args.out))
    _log(phase="judge-audit-export", **{k: str(v) for k, v in paths.items()})


def phase_judge_agreement(args) -> None:
    from prematurestop.pipelines.bridge_mechanism import audit

    rows = audit.agreement(Path(args.out))
    from prematurestop.pipelines.bridge_mechanism.analysis import write_csv
    write_csv(Path(args.out) / "judge_agreement.csv", rows)
    for r in rows:
        _log(phase="judge-agreement", **r)


def phase_fidelity_check(args) -> None:
    """Does a fresh neutral arm reproduce its anchor run exactly? Same first observation
    (d1 hash), same episode-0 decision sequence, same stopping step and route."""
    from prematurestop.evaluation.completion import build_canonical_record
    from prematurestop.evaluation.completion_inventory import RowRecord
    from prematurestop.evaluation.trace_parser import parse_trace_file
    from prematurestop.pipelines.bridge_mechanism.analysis import write_csv

    def summary(tp, victim, ds, ex):
        rec = build_canonical_record(RowRecord(model=victim, dataset=ds, condition="NaturalClean", shard_index=0,
                                               example_id=ex, final_status="ok", error="", trace_path=tp,
                                               trace_exists=True, valid_answer=None, row_index_in_shard=0,
                                               run_id="x"))
        ep = parse_trace_file(Path(tp)).episodes
        seq = [tuple(d.tool_call_names) for d in ep[0].decisions] if ep else []
        return rec.terminal_route, rec.first_terminal_post_observation_step, seq

    rows = []
    for victim in args.victims or C.PILOT_VICTIMS:
        for ds in args.datasets:
            anchors = index_by(_read_all(args, "anchors", f"{victim}_{ds}"))
            for r in _read_all(args, "arms", f"{victim}_{ds}"):
                if r["arm"] != "neutral" or r["status"] not in ("ran", "d1_mismatch"):
                    continue
                a = anchors.get(f"{victim}|{ds}|{r['example_id']}")
                if not a or a.get("status") == VICTIM_FAILURE:
                    continue
                fa, fn = summary(str(resolve_state_path(a["anchor_trace_path"])), victim, ds, r["example_id"]), \
                    summary(str(resolve_state_path(r["trace_path"])), victim, ds, r["example_id"])
                rows.append({"victim": victim, "dataset": ds, "example_id": r["example_id"],
                             "d1_identical": r["status"] == "ran", "route_identical": fa[0] == fn[0],
                             "stop_step_identical": fa[1] == fn[1], "decisions_identical": fa[2] == fn[2]})
    out = (Path(args.results_dir) / args.split if args.results_dir else EXP_DIR / "diagnostics" / args.split)
    write_csv(out / "fidelity_check.csv", rows)
    for victim in args.victims or C.PILOT_VICTIMS:
        for ds in args.datasets:
            rr = [x for x in rows if (x["victim"], x["dataset"]) == (victim, ds)]
            if rr:
                _log(phase="fidelity-check", victim=victim, dataset=ds, n=len(rr),
                     **{k: round(sum(x[k] for x in rr) / len(rr), 4) for k in
                        ("d1_identical", "decisions_identical", "stop_step_identical", "route_identical")})


PHASES = {"decompose": phase_decompose, "anchor": phase_anchor, "cues": phase_cues, "select": phase_select,
          "arms": phase_arms, "analyze": phase_analyze, "audit-export": phase_audit_export,
          "fit-lambda": phase_fit_lambda, "baseline-gate": phase_baseline_gate,
          "fidelity-check": phase_fidelity_check,
          "judge-audit-export": phase_judge_audit_export, "judge-agreement": phase_judge_agreement, "freeze": phase_freeze, "verify-freeze": phase_verify_freeze}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dynamic bridge-cue pilot driver")
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--split", choices=C.SPLITS, default="dev")
    ap.add_argument("--split-dir", default=str(DEFAULT_SPLITS))
    ap.add_argument("--state-dir", default=os.environ.get("PILOT_STATE_DIR", str(EXP_DIR / "state")),
                    help="checkpoints: phase JSONLs + victim traces (on NRP: /pvc/checkpoints/...)")
    ap.add_argument("--results-dir", default=os.environ.get("PILOT_RESULTS_DIR"),
                    help="analysis outputs (default: experiments/whitebox/e2_bridge_mechanism/<category>/<split>)")
    ap.add_argument("--victim", choices=C.PILOT_VICTIMS)
    ap.add_argument("--victims", nargs="*", choices=C.PILOT_VICTIMS)
    ap.add_argument("--datasets", nargs="*", default=list(C.PILOT_DATASETS), choices=C.PILOT_DATASETS)
    ap.add_argument("--victim-base-url")
    ap.add_argument("--attacker-base-url")
    ap.add_argument("--api-key", default=os.environ.get("PILOT_API_KEY", "EMPTY"))
    ap.add_argument("--tokenizer-json")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("PILOT_SHARD", "0")),
                    help="this process's shard index (NRP Indexed Job: JOB_COMPLETION_INDEX)")
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("PILOT_NUM_SHARDS", "1")))
    ap.add_argument("--lambda-file")
    ap.add_argument("--provisional-equal-lambda", action="store_true")
    ap.add_argument("--ratings")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.phase in ("anchor", "arms") and not (args.victim and args.victim_base_url):
        ap.error(f"{args.phase} needs --victim and --victim-base-url")
    if args.phase in ("cues", "select") and not args.victim:
        ap.error(f"{args.phase} needs --victim")
    if args.phase in ("decompose", "cues") and not args.attacker_base_url:
        ap.error(f"{args.phase} needs --attacker-base-url")
    if not 0 <= args.shard < max(args.num_shards, 1):
        ap.error("--shard must be in [0, --num-shards)")
    if args.phase in ("analyze", "audit-export", "fit-lambda", "freeze", "verify-freeze", "decompose",
                      "baseline-gate", "judge-audit-export", "judge-agreement", "fidelity-check") and args.num_shards > 1:
        ap.error(f"{args.phase} is not sharded; it reads/writes every shard")
    if args.split in C.FROZEN_SPLITS and args.max_questions:
        ap.error(f"--max-questions is not allowed on {args.split} (no subsetting of frozen splits)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    PHASES[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
