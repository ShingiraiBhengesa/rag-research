"""White-box complexity attack on the Adaptive-RAG router via FLAN-T5-XL.

Three structural prompts targeting features the router measures:
  - inflate (add relative clauses)
  - deflate (more concise)
  - embed (indirect interrogative)
"""
import argparse, json, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

CLASSIFIER_PATH = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                   "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
                   "flan_t5_xxl/epoch/30")
REPHRASER_NAME = "google/flan-t5-xl"
SIM_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
PPL_MODEL_NAME = "gpt2"
LABELS = ["A", "B", "C"]
LABEL_TOKEN_IDS = {"A": 71, "B": 272, "C": 205}
POS_KEEP = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")

PROMPTS = {
    "inflate": ("Rewrite the following question by adding a relative clause or "
                "qualifying phrase. Keep the same answer. The rewritten question "
                "should be longer and more complex.\n"
                "Original: {q}\nRewritten:"),
    "deflate": ("Rewrite the following question more concisely while keeping "
                "the same answer.\n"
                "Original: {q}\nRewritten:"),
    "embed": ("Rephrase the following question as an indirect or embedded "
              "interrogative, keeping the same answer.\n"
              "Original: {q}\nRephrased:"),
}


class Rephraser:
    def __init__(self, model_name=REPHRASER_NAME, device="cpu"):
        print(f"[rephraser] loading {model_name} on {device}...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, q, n_per_prompt=4, max_new_tokens=80):
        all_rephrases = []
        for style, template in PROMPTS.items():
            prompt = template.format(q=q)
            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
            inp = {kk: v.to(self.device) for kk, v in inp.items()}
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                      do_sample=True, top_p=0.92, temperature=0.7,
                                      num_return_sequences=n_per_prompt)
            for seq in out:
                txt = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                if txt:
                    all_rephrases.append({"text": txt, "style": style})
        seen, dedup = set(), []
        for r in all_rephrases:
            if r["text"] not in seen:
                seen.add(r["text"])
                dedup.append(r)
        return dedup


class ClassifierLogits:
    def __init__(self, model_path=CLASSIFIER_PATH, device="cpu"):
        print(f"[classifier] loading from {model_path} on {device}...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def logits_abc(self, text):
        inp = self.tokenizer(text, return_tensors="pt", max_length=384,
                             truncation=True, padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model.generate(**inp, max_new_tokens=1, output_scores=True,
                                  return_dict_in_generate=True)
        L = out.scores[0][0]
        return np.array([float(L[LABEL_TOKEN_IDS[l]]) for l in LABELS])


class FilterStack:
    def __init__(self, sim_threshold=0.85, ppl_threshold=200.0, device="cpu"):
        print(f"[filters] loading MiniLM and GPT-2 on {device}...", flush=True)
        self.sim_model = SentenceTransformer(SIM_MODEL_NAME, device=device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model.to(device).eval()
        self.device = device
        self.sim_threshold = sim_threshold
        self.ppl_threshold = ppl_threshold
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def cosine(self, a, b):
        e = self.sim_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(e[0] @ e[1])

    @torch.no_grad()
    def perplexity(self, text):
        enc = self.ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 2:
            return 1.0
        out = self.ppl_model(**enc, labels=enc["input_ids"])
        return float(torch.exp(out.loss).item())

    def has_lexical_novelty(self, original, candidate):
        oc = {t.lemma_.lower() for t in self.nlp(original)
              if t.pos_ in POS_KEEP and not t.is_stop}
        cc = {t.lemma_.lower() for t in self.nlp(candidate)
              if t.pos_ in POS_KEEP and not t.is_stop}
        return len(oc.symmetric_difference(cc)) > 0

    def evaluate(self, original, candidate):
        sim = self.cosine(original, candidate)
        ppl = self.perplexity(candidate)
        novel = self.has_lexical_novelty(original, candidate)
        return {"sim": sim, "ppl": ppl, "lexical_novel": novel,
                "passed": (sim >= self.sim_threshold and ppl <= self.ppl_threshold and novel)}


def attack_one(rec, rephraser, cls, filters, n_per_prompt):
    qid, q = rec["qid"], rec["question"]
    actual = rec.get("actual_route") or rec.get("baseline_actual_route")
    base = cls.logits_abc(q)
    base_route = LABELS[int(np.argmax(base))]
    base_idx = LABELS.index(base_route)
    cands = rephraser.generate(q, n_per_prompt=n_per_prompt)
    results, success = [], None
    for c in cands:
        cand, style = c["text"], c["style"]
        f = filters.evaluate(q, cand)
        if not f["passed"]:
            results.append({"rephrase": cand, "style": style, **f,
                            "route_after": None, "logit_drop_correct_route": None,
                            "flip": False})
            continue
        cand_logits = cls.logits_abc(cand)
        new_route = LABELS[int(np.argmax(cand_logits))]
        drop = float(base[base_idx] - cand_logits[base_idx])
        flip = (new_route != base_route)
        results.append({"rephrase": cand, "style": style, **f,
                        "route_after": new_route,
                        "logit_drop_correct_route": drop,
                        "logits_abc": cand_logits.tolist(), "flip": flip})
        if flip and success is None:
            success = {"rephrase": cand, "style": style,
                       "route_before": base_route, "route_after": new_route,
                       "logit_drop": drop}
    return {"qid": qid, "question": q, "actual_route": actual,
            "baseline_route_predicted": base_route, "baseline_logits_abc": base.tolist(),
            "n_candidates_generated": len(cands),
            "n_candidates_passed_filters": sum(1 for r in results if r["passed"]),
            "n_route_flips": sum(1 for r in results if r["flip"]),
            "first_success": success, "all_candidates": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victims", default="results/attacks/whitebox/victims_107.jsonl")
    ap.add_argument("--out-dir", default="results/attacks/whitebox_complexity")
    ap.add_argument("--n-victims", type=int, default=None)
    ap.add_argument("--n-per-prompt", type=int, default=4,
                    help="rephrasings per prompt style (3 styles total)")
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--ppl", type=float, default=200.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rephraser = Rephraser(device=device)
    cls = ClassifierLogits(device=device)
    filters = FilterStack(sim_threshold=args.sim, ppl_threshold=args.ppl, device=device)

    with open(args.victims) as f:
        victims = [json.loads(l) for l in f]
    if args.n_victims:
        victims = victims[:args.n_victims]
    print(f"\n[run] {len(victims)} victims, n_per_prompt={args.n_per_prompt} "
          f"(3 styles), device={device}, sim>={args.sim}, ppl<={args.ppl}", flush=True)

    out_path = out / f"attack_complexity_whitebox_n{len(victims)}.jsonl"
    summary_path = out / f"summary_n{len(victims)}.json"

    n_total = n_filter = n_flip = 0
    flips_by_route = {"A": 0, "B": 0, "C": 0}
    flips_by_style = {"inflate": 0, "deflate": 0, "embed": 0}
    t_start = time.time()

    with open(out_path, "w") as fo:
        for i, rec in enumerate(victims, 1):
            t0 = time.time()
            try:
                r = attack_one(rec, rephraser, cls, filters, args.n_per_prompt)
            except Exception as e:
                print(f"  [{i}/{len(victims)}] qid={rec['qid']} ERROR: {e}", flush=True)
                continue
            fo.write(json.dumps(r) + "\n"); fo.flush()
            n_total += 1
            if r["n_candidates_passed_filters"] > 0: n_filter += 1
            if r["n_route_flips"] > 0:
                n_flip += 1
                ar = r.get("actual_route") or "?"
                if ar in flips_by_route: flips_by_route[ar] += 1
                if r["first_success"]:
                    flips_by_style[r["first_success"]["style"]] += 1
            print(f"  [{i}/{len(victims)}] qid={r['qid']}  base={r['baseline_route_predicted']}"
                  f"  gen={r['n_candidates_generated']}"
                  f"  pass={r['n_candidates_passed_filters']}"
                  f"  flips={r['n_route_flips']}  wall={time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t_start
    summary = {"n_victims": n_total, "n_any_filter_pass": n_filter, "n_any_flip": n_flip,
               "flip_rate": round(n_flip / max(1, n_total), 4),
               "filter_pass_rate": round(n_filter / max(1, n_total), 4),
               "flips_by_actual_route": flips_by_route,
               "flips_by_style": flips_by_style,
               "n_per_prompt": args.n_per_prompt,
               "sim_threshold": args.sim, "ppl_threshold": args.ppl, "device": device,
               "wall_total_s": round(elapsed, 1)}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
