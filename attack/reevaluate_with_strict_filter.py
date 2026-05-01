"""Apply strict filters retroactively to existing JSONL results."""
import argparse, json, time
from collections import Counter
from pathlib import Path
import spacy
from semantic_filter import (EntityPreservationFilter, NLIBidirectionalFilter,
                             QuestionFormFilter, QualifierBlocklistFilter)

DEFAULT_INPUT = "results/attacks/whitebox_complexity_groq/attack_complexity_groq_n107.jsonl"
DEFAULT_OUTPUT = "results/attacks/whitebox_complexity_groq/strict_filter_v2_retroactive.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-entity", action="store_true")
    ap.add_argument("--no-qform", action="store_true")
    ap.add_argument("--no-qual", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records", flush=True)

    nlp = spacy.load("en_core_web_sm")
    entity_filter = EntityPreservationFilter(nlp) if not args.no_entity else None
    nli_filter = (NLIBidirectionalFilter(device=args.device,
                                         entailment_threshold=args.threshold)
                  if not args.no_nli else None)
    qform_filter = QuestionFormFilter() if not args.no_qform else None
    qual_filter = QualifierBlocklistFilter() if not args.no_qual else None

    def evaluate(original, candidate):
        ent = entity_filter.evaluate(original, candidate) if entity_filter else None
        nli = nli_filter.evaluate(original, candidate) if nli_filter else None
        qf = qform_filter.evaluate(original, candidate) if qform_filter else None
        ql = qual_filter.evaluate(original, candidate) if qual_filter else None
        ent_pass = ent is None or ent["passed"]
        nli_pass = nli is None or nli["passed"]
        qf_pass = qf is None or qf["passed"]
        ql_pass = ql is None or ql["passed"]
        return {"passed_strict": ent_pass and nli_pass and qf_pass and ql_pass,
                "entity": ent, "nli": nli, "qform": qf, "qual": ql}

    n_victims = n_old_pass = n_old_flip = 0
    n_new_pass = n_new_flip = 0
    rejection_reasons = Counter()
    flip_changes = {"kept_clean": 0, "lost_to_entity": 0,
                    "lost_to_nli_forward": 0, "lost_to_nli_reverse": 0,
                    "lost_to_qform": 0, "lost_to_qual": 0,
                    "lost_to_multiple": 0}
    out = []
    t_start = time.time()

    for i, rec in enumerate(records, 1):
        n_victims += 1
        original = rec["question"]
        if rec["n_candidates_passed_filters"] > 0: n_old_pass += 1
        if rec["n_route_flips"] > 0: n_old_flip += 1
        had_pass = had_flip = False
        first_success = None
        cands = []
        for c in rec.get("all_candidates", []):
            if not c.get("passed"): continue
            v = evaluate(original, c["rephrase"])
            new_passed = v["passed_strict"]
            new_flip = new_passed and c.get("flip", False)
            cands.append({
                "rephrase": c["rephrase"], "style": c.get("style"),
                "old_route_after": c.get("route_after"),
                "old_logit_drop": c.get("logit_drop_correct_route"),
                "old_flip": c.get("flip", False),
                "strict_passed": new_passed, "strict_flip": new_flip,
                "entity_filter": v["entity"], "nli_filter": v["nli"],
                "qform_filter": v["qform"], "qual_filter": v["qual"],
            })
            if new_passed: had_pass = True
            if new_flip and first_success is None:
                first_success = {"rephrase": c["rephrase"], "style": c.get("style"),
                                 "route_before": rec["baseline_route_predicted"],
                                 "route_after": c.get("route_after"),
                                 "logit_drop": c.get("logit_drop_correct_route")}
                had_flip = True
            if c.get("flip") and not new_passed:
                fails = []
                if v["entity"] and not v["entity"]["passed"]:
                    fails.append("entity")
                    rejection_reasons[f"entity:{v['entity']['reason'][:40]}"] += 1
                if v["nli"] and not v["nli"]["passed"]:
                    if "forward" in v["nli"]["reason"]: fails.append("nli_fwd")
                    else: fails.append("nli_rev")
                    rejection_reasons[f"nli:{v['nli']['reason']}"] += 1
                if v["qform"] and not v["qform"]["passed"]:
                    fails.append("qform")
                    rejection_reasons[f"qform:{v['qform']['reason']}"] += 1
                if v["qual"] and not v["qual"]["passed"]:
                    fails.append("qual")
                    rejection_reasons[f"qual:{v['qual']['reason'][:60]}"] += 1
                if len(fails) > 1:
                    flip_changes["lost_to_multiple"] += 1
                elif fails == ["entity"]:
                    flip_changes["lost_to_entity"] += 1
                elif fails == ["nli_fwd"]:
                    flip_changes["lost_to_nli_forward"] += 1
                elif fails == ["nli_rev"]:
                    flip_changes["lost_to_nli_reverse"] += 1
                elif fails == ["qform"]:
                    flip_changes["lost_to_qform"] += 1
                elif fails == ["qual"]:
                    flip_changes["lost_to_qual"] += 1
            elif c.get("flip") and new_passed:
                flip_changes["kept_clean"] += 1
        if had_pass: n_new_pass += 1
        if had_flip: n_new_flip += 1
        out.append({
            "qid": rec["qid"], "question": original,
            "actual_route": rec.get("actual_route"),
            "baseline_route_predicted": rec["baseline_route_predicted"],
            "old_n_filter_pass": rec["n_candidates_passed_filters"],
            "old_n_flip": rec["n_route_flips"],
            "strict_n_filter_pass": sum(1 for c in cands if c["strict_passed"]),
            "strict_n_flip": sum(1 for c in cands if c["strict_flip"]),
            "first_strict_success": first_success, "candidates": cands,
        })
        if i % 10 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}]  old_flips={n_old_flip}  "
                  f"strict_flips={n_new_flip}  "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")

    print()
    print("=" * 78)
    print("  STRICT-FILTER v2 RETROACTIVE ANALYSIS")
    print("=" * 78)
    print(f"  n_victims:                 {n_victims}")
    print(f"  old filter pass rate:      {n_old_pass}/{n_victims} "
          f"({100*n_old_pass/n_victims:.1f}%)")
    print(f"  old route flip rate:       {n_old_flip}/{n_victims} "
          f"({100*n_old_flip/n_victims:.1f}%)")
    print(f"  strict v2 filter pass:     {n_new_pass}/{n_victims} "
          f"({100*n_new_pass/n_victims:.1f}%)")
    print(f"  strict v2 route flip:      {n_new_flip}/{n_victims} "
          f"({100*n_new_flip/n_victims:.1f}%)")
    print()
    print(f"  Filters enabled:")
    print(f"    entity (4):       {'OFF' if args.no_entity else 'ON'}")
    print(f"    NLI (5):          {'OFF' if args.no_nli else f'ON (threshold={args.threshold})'}")
    print(f"    qform (6):        {'OFF' if args.no_qform else 'ON'}")
    print(f"    qualifier (7):    {'OFF' if args.no_qual else 'ON'}")
    print()
    print(f"  Flip outcomes (out of {n_old_flip} original flips):")
    print(f"    kept clean:                            {flip_changes['kept_clean']}")
    print(f"    lost to entity ONLY:                   {flip_changes['lost_to_entity']}")
    print(f"    lost to NLI forward ONLY (halluc):     {flip_changes['lost_to_nli_forward']}")
    print(f"    lost to NLI reverse ONLY (drop):       {flip_changes['lost_to_nli_reverse']}")
    print(f"    lost to qform ONLY (uncert/declar):    {flip_changes['lost_to_qform']}")
    print(f"    lost to qualifier ONLY (puffy adj):    {flip_changes['lost_to_qual']}")
    print(f"    lost to multiple filters:              {flip_changes['lost_to_multiple']}")
    print()
    if rejection_reasons:
        print(f"  Top rejection reasons:")
        for reason, count in rejection_reasons.most_common(8):
            print(f"    {count:3d}  {reason}")
    print()
    if n_old_flip > 0:
        clean_rate = 100 * flip_changes["kept_clean"] / n_old_flip
        print(f"  Clean fraction of original flips:  {clean_rate:.1f}%")
    print()
    print(f"  Retroactive output: {args.output}")


if __name__ == "__main__":
    main()
