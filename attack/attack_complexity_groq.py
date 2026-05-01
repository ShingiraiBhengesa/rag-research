"""White-box complexity attack on Adaptive-RAG router via Groq + Llama-3.3-70B.

Three structural prompts targeting features the router measures:
  - inflate (add relative clauses / qualifiers)
  - deflate (more concise structure)
  - embed (indirect / embedded interrogative)

Reuses the local FilterStack and ClassifierLogits from the FLAN-T5 attempt;
only the rephraser changes (FLAN-T5-XL -> Llama-3.3-70B via Groq).
"""
import argparse, json, os, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

CLASSIFIER_PATH = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                   "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
                   "flan_t5_xxl/epoch/30")
SIM_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
PPL_MODEL_NAME = "gpt2"
GROQ_MODEL = "llama-3.3-70b-versatile"
LABELS = ["A", "B", "C"]
LABEL_TOKEN_IDS = {"A": 71, "B": 272, "C": 205}
POS_KEEP = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")

SYSTEM_PROMPT = (
    "You are a precise question paraphraser. Given an original question, you "
    "produce alternative versions of it that have the SAME intended meaning "
    "and SAME correct answer, but differ in surface form (length, structure, "
    "syntax, level of explicitness). Never invent facts. Never change the "
    "subject of the question. Output only the rephrased question, nothing else."
)

PROMPTS = {
    "inflate": (
        "Rewrite the following question by adding a relative clause or "
        "qualifying phrase that makes it look more complex, while keeping the "
        "same answer. Do not change what is being asked.\n\n"
        "Question: {q}\n\n"
        "Rewritten question:"
    ),
    "deflate": (
        "Rewrite the following question more concisely or directly, while "
        "keeping the same answer. Do not change what is being asked.\n\n"
        "Question: {q}\n\n"
        "Rewritten question:"
    ),
    "embed": (
        "Rephrase the following question as an indirect or embedded "
        "interrogative form, while keeping the same answer. Do not change "
        "what is being asked.\n\n"
        "Question: {q}\n\n"
        "Rephrased question:"
    ),
}


class GroqRephraser:
    def __init__(self, api_key: str, model: str = GROQ_MODEL,
                 sleep_per_request: float = 2.1):
        print(f"[rephraser] Groq client, model={model}", flush=True)
        self.client = Groq(api_key=api_key)
        self.model = model
        self.sleep = sleep_per_request

    def generate(self, q: str, n_per_prompt: int = 4) -> List[Dict[str, str]]:
        all_rephrases = []
        for style, template in PROMPTS.items():
            user_msg = template.format(q=q)
            for _ in range(n_per_prompt):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.8,
                        top_p=0.9,
                        max_tokens=120,
                    )
                    txt = resp.choices[0].message.content.strip()
                    txt = txt.strip("\"'").strip()
                    if txt:
                        all_rephrases.append({"text": txt, "style": style})
                except Exception as e:
                    print(f"  [groq] error on style={style}: {e}", flush=True)
                time.sleep(self.sleep)
        seen, dedup = set(), []
        for r in all_rephrases:
            if r["text"] not in seen:
                seen.add(r["text"])
                dedup.append(r)
        return dedup


class ClassifierLogits:
    def __init__(self, model_path: str = CLASSIFIER_PATH, device: str = "cpu"):
        print(f"[classifier] loading on {device}...", flush=True)
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
        print(f"[filters] loading on {device}...", flush=True)
        self.sim_model = SentenceTransformer(SIM_MODEL_NAME, device=device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model.to(device).eval()
        self.device = device
        self.sim_threshold = sim_threshold
        self.ppl_threshold = ppl_threshold
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def cosine(self, a, b):
        e = self.sim_model.encode([a, b], convert_to_numpy=True,
                                  normalize_embeddings=True)
        return float(e[0] @ e[1])

    @torch.no_grad()
    def perplexity(self, text):
        enc = self.ppl_tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=256)
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
                "passed": (sim >= self.sim_threshold
                           and ppl <= self.ppl_threshold and novel)}


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
                            "route_after": None,
                            "logit_drop_correct_route": None, "flip": False})
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
            "baseline_route_predicted": base_route,
            "baseline_logits_abc": base.tolist(),
            "n_candidates_generated": len(cands),
            "n_candidates_passed_filters": sum(1 for r in results if r["passed"]),
            "n_route_flips": sum(1 for r in results if r["flip"]),
            "first_success": success, "all_candidates": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victims",
                    default="results/attacks/whitebox/victims_107.jsonl")
    ap.add_argument("--out-dir",
                    default="results/attacks/whitebox_complexity_groq")
    ap.add_argument("--n-victims", type=int, default=None)
    ap.add_argument("--n-per-prompt", type=int, default=4)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--ppl", type=float, default=200.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sleep", type=float, default=2.1,
                    help="seconds between Groq API calls (rate limit safety)")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set GROQ_API_KEY environment variable")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rephraser = GroqRephraser(api_key=api_key, sleep_per_request=args.sleep)
    cls = ClassifierLogits(device=device)
    filters = FilterStack(sim_threshold=args.sim, ppl_threshold=args.ppl,
                          device=device)

    with open(args.victims) as f:
        victims = [json.loads(l) for l in f]
    if args.n_victims:
        victims = victims[: args.n_victims]
    print(f"\n[run] {len(victims)} victims, n_per_prompt={args.n_per_prompt} "
          f"(3 styles), device={device}, sim>={args.sim}, ppl<={args.ppl}, "
          f"groq_sleep={args.sleep}s", flush=True)

    out_path = out / f"attack_complexity_groq_n{len(victims)}.jsonl"
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
                print(f"  [{i}/{len(victims)}] qid={rec['qid']} ERROR: {e}",
                      flush=True)
                continue
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            n_total += 1
            if r["n_candidates_passed_filters"] > 0:
                n_filter += 1
            if r["n_route_flips"] > 0:
                n_flip += 1
                ar = r.get("actual_route") or "?"
                if ar in flips_by_route:
                    flips_by_route[ar] += 1
                if r["first_success"]:
                    flips_by_style[r["first_success"]["style"]] += 1
            print(f"  [{i}/{len(victims)}] qid={r['qid']}"
                  f"  base={r['baseline_route_predicted']}"
                  f"  gen={r['n_candidates_generated']}"
                  f"  pass={r['n_candidates_passed_filters']}"
                  f"  flips={r['n_route_flips']}"
                  f"  wall={time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t_start
    summary = {"n_victims": n_total, "n_any_filter_pass": n_filter,
               "n_any_flip": n_flip,
               "flip_rate": round(n_flip / max(1, n_total), 4),
               "filter_pass_rate": round(n_filter / max(1, n_total), 4),
               "flips_by_actual_route": flips_by_route,
               "flips_by_style": flips_by_style,
               "n_per_prompt": args.n_per_prompt,
               "sim_threshold": args.sim, "ppl_threshold": args.ppl,
               "device": device, "rephraser": GROQ_MODEL,
               "wall_total_s": round(elapsed, 1)}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
