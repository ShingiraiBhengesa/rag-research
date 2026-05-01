"""Post-run sanity check for complexity-attack flips."""
import argparse, json, os, random, sys
from pathlib import Path

DEFAULT_INPUT = "results/attacks/whitebox_complexity_groq/attack_complexity_groq_n107.jsonl"
DEFAULT_TAGS = "results/attacks/whitebox_complexity_groq/flip_tags.jsonl"
TAG_KEYS = {"c": "clean", "h": "hallucinated", "q": "question_changed",
            "u": "uncertain", "s": "skip", "?": "show_help", "x": "exit"}
HELP_TEXT = """
Tagging guide:
  c = clean             Same question, different surface form, gold answer still works
  h = hallucinated      Rephrasing introduces an entity/fact NOT in the original
  q = question_changed  Rephrasing genuinely asks something different
  u = uncertain         Borderline; flag for second look
  s = skip              Don't tag this one (won't appear in stats)
  ? = show this help
  x = exit and save
"""

def load_flips(p):
    out = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("first_success"):
                out.append({"qid": r["qid"], "original": r["question"],
                            "actual_route": r.get("actual_route"),
                            "baseline_route_predicted": r["baseline_route_predicted"],
                            "rephrase": r["first_success"]["rephrase"],
                            "style": r["first_success"]["style"],
                            "route_after": r["first_success"]["route_after"],
                            "logit_drop": r["first_success"]["logit_drop"]})
    return out

def load_tags(p):
    if not os.path.exists(p): return {}
    t = {}
    with open(p) as f:
        for line in f:
            try: r = json.loads(line); t[r["qid"]] = r["tag"]
            except: continue
    return t

def save_tag(p, qid, flip, tag):
    rec = {"qid": qid, "tag": tag, "rephrase": flip["rephrase"],
           "original": flip["original"], "style": flip["style"]}
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f: f.write(json.dumps(rec) + "\n")

def show(flip, i, n):
    print(); print("=" * 78)
    print(f"  [{i}/{n}]  qid: {flip['qid']}")
    print(f"  style:    {flip['style']}")
    print(f"  route:    {flip['baseline_route_predicted']} -> {flip['route_after']}"
          f"  (logit drop: {flip['logit_drop']:.2f})")
    print("=" * 78); print()
    print("ORIGINAL QUESTION:"); print(f"  {flip['original']}"); print()
    print("REPHRASING THAT FLIPPED:"); print(f"  {flip['rephrase']}"); print()

def prompt():
    while True:
        c = input("Tag (c/h/q/u/s, ? for help, x to exit): ").strip().lower()
        if c in TAG_KEYS: return TAG_KEYS[c]
        print(f"  unrecognized: {c!r}")

def summary(p):
    if not os.path.exists(p): print("No tags."); return
    counts, examples = {}, {}
    with open(p) as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            t = r["tag"]; counts[t] = counts.get(t, 0) + 1
            if t not in examples: examples[t] = r
    total = sum(v for k, v in counts.items() if k != "skip")
    print(); print("=" * 78); print(f"  TAG SUMMARY  (n={total} tagged)"); print("=" * 78)
    for t in ["clean", "hallucinated", "question_changed", "uncertain", "skip"]:
        n = counts.get(t, 0)
        if t != "skip":
            pct = round(100 * n / max(1, total), 1)
            print(f"  {t:18s}  {n:3d}  ({pct}%)")
        else:
            print(f"  {t:18s}  {n:3d}")
    print()
    for t in ["clean", "hallucinated", "question_changed"]:
        if t in examples:
            ex = examples[t]
            print(f"Example {t}:")
            print(f"  qid:       {ex['qid']}")
            print(f"  original:  {ex['original']}")
            print(f"  rephrase:  {ex['rephrase']}"); print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--tags-file", default=DEFAULT_TAGS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-sample", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.summary: summary(args.tags_file); return
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found"); sys.exit(1)
    flips = load_flips(args.input)
    print(f"Loaded {len(flips)} successful flips")
    existing = load_tags(args.tags_file)
    if existing: print(f"Found {len(existing)} previously tagged flips")
    if args.resume:
        flips = [f for f in flips if f["qid"] not in existing]
        print(f"After resume: {len(flips)} untagged")
    if not args.all:
        random.seed(args.seed)
        flips = random.sample(flips, min(args.n_sample, len(flips)))
        print(f"Sampling {len(flips)} for inspection")
    print(HELP_TEXT)
    input("Press Enter to begin...")
    n = 0
    for i, flip in enumerate(flips, 1):
        show(flip, i, len(flips))
        t = prompt()
        if t == "exit":
            print(f"\nExiting after {n} tags. Use --resume to continue."); break
        if t == "show_help":
            print(HELP_TEXT); t = prompt()
            if t in ("exit", "show_help"): break
        if t == "skip": print("  [skipped]"); continue
        save_tag(args.tags_file, flip["qid"], flip, t)
        n += 1
        print(f"  -> tagged as: {t}")
    print()
    summary(args.tags_file)

if __name__ == "__main__": main()
