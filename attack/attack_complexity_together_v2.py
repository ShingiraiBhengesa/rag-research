"""White-box complexity attack v2 via Together AI + Llama-3.3-70B."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import spacy, torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from together import Together
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)
from semantic_filter import (EntityPreservationFilter, NLIBidirectionalFilter,
                             QuestionFormFilter, QualifierBlocklistFilter,
                             CategoricalDescriptorFilter)

CLASSIFIER_PATH = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                   "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
                   "flan_t5_xxl/epoch/30")
SIM_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
PPL_MODEL_NAME = "gpt2"
TOGETHER_MODEL_PAID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
LABELS = ["A", "B", "C"]
LABEL_TOKEN_IDS = {"A": 71, "B": 272, "C": 205}
POS_KEEP = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")

SYSTEM_PROMPT = ("You are a precise question paraphraser. Output ONLY the "
                 "paraphrased question and nothing else - no explanation, "
                 "no commentary, no quotes.")

INFLATE_PROMPT = """You are paraphrasing a question. The paraphrase MUST:
- Preserve the EXACT same answer
- Be phrased as a question (end with "?")
- NOT introduce any entity, place, person, date, fact, or property not in the original
- NOT add adjectives like "renowned", "famous", "iconic", "notable", "specific"
- NOT change uncertainty or confidence (do not say "it is not known" or "one might wonder")

Make the paraphrase longer by adding a SEMANTICALLY EMPTY relative clause or qualifier.
Empty means: adds words but no new information.

EXAMPLE:
Original: Where was Walt Disney born?
GOOD paraphrase: Where was Walt Disney, the person who invented him, born?
BAD paraphrase: Where was the renowned animator Walt Disney born?  (adds "renowned")
BAD paraphrase: It is not clear where Walt Disney was born.  (changes to assertion)
BAD paraphrase: Where was Mickey Mouse's creator Walt Disney born?  (adds "Mickey Mouse")

Now paraphrase this question. Output ONLY the paraphrase, no explanation.

Original: {q}
Paraphrase:"""

EMBED_PROMPT = """You are rephrasing a question into an indirect-style question. The rephrasing MUST:
- Preserve the EXACT same answer
- Be phrased as a question (end with "?")
- NOT introduce any entity, place, person, date, fact, or property not in the original
- NOT add adjectives like "renowned", "famous", "iconic", "notable"
- NOT use "it is not known", "is unclear", or other uncertainty markers

Rephrase using indirect interrogative structure: "What is the answer to..." or "Can you tell me..." form.

EXAMPLE:
Original: Who founded Microsoft?
GOOD rephrase: Who is the person that founded Microsoft?
GOOD rephrase: Can you identify who founded Microsoft?
BAD rephrase: It is not clear who founded Microsoft.  (changes to assertion)
BAD rephrase: Who is the famous tech entrepreneur who founded Microsoft?  (adds "famous tech entrepreneur")

Now rephrase this question. Output ONLY the rephrasing, no explanation.

Original: {q}
Rephrase:"""

PROMPTS = {"embed": EMBED_PROMPT}  # dropped inflate (produced tautological + hallucinated rephrases)


class TogetherRephraser:
    def __init__(self, api_key, model=TOGETHER_MODEL_PAID, sleep_per_request=0.5):
        self.client = Together(api_key=api_key)
        self.model = model
        self.sleep = sleep_per_request
        print(f"[rephraser] Together AI, model={self.model}, sleep={self.sleep}s", flush=True)

    def generate(self, q, n_per_prompt=5):
        rephrases = []
        for style, template in PROMPTS.items():
            user_msg = template.format(q=q)
            for _ in range(n_per_prompt):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user_msg}],
                        temperature=0.95, top_p=0.95, max_tokens=80)
                    txt = resp.choices[0].message.content.strip().strip("\"'").strip()
                    for prefix in ["Paraphrase:", "Rephrase:", "Rephrased:",
                                   "Question:", "Output:"]:
                        if txt.startswith(prefix): txt = txt[len(prefix):].strip()
                    if txt: rephrases.append({"text": txt, "style": style})
                except Exception as e:
                    print(f"  [together] error: {str(e)[:120]}", flush=True)
                if self.sleep > 0: time.sleep(self.sleep)
        seen, dedup = set(), []
        for r in rephrases:
            if r["text"] not in seen: seen.add(r["text"]); dedup.append(r)
        return dedup


class ClassifierLogits:
    def __init__(self, model_path=CLASSIFIER_PATH, device="cpu"):
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


class SurfaceFilters:
    def __init__(self, sim_threshold=0.85, ppl_threshold=200.0, device="cpu", nlp=None):
        print(f"[surface] loading on {device}...", flush=True)
        self.sim_model = SentenceTransformer(SIM_MODEL_NAME, device=device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME)
        self.ppl_model.to(device).eval()
        self.device = device
        self.sim_threshold = sim_threshold
        self.ppl_threshold = ppl_threshold
        self.nlp = nlp or spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def cosine(self, a, b):
        e = self.sim_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(e[0] @ e[1])

    @torch.no_grad()
    def perplexity(self, text):
        enc = self.ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 2: return 1.0
        out = self.ppl_model(**enc, labels=enc["input_ids"])
        return float(torch.exp(out.loss).item())

    def has_lexical_novelty(self, original, candidate):
        # Loosened: any normalized text difference counts (was: content-lemma diff).
        # Complexity attacks operate on structure, not content words, so stop-word
        # and word-order changes are valid perturbations.
        import re
        norm_o = re.sub(r"\s+", " ", original.lower().strip())
        norm_c = re.sub(r"\s+", " ", candidate.lower().strip())
        return norm_o != norm_c

    def evaluate(self, original, candidate):
        sim = self.cosine(original, candidate)
        ppl = self.perplexity(candidate)
        novel = self.has_lexical_novelty(original, candidate)
        return {"sim": sim, "ppl": ppl, "lexical_novel": novel,
                "passed": sim >= self.sim_threshold
                          and ppl <= self.ppl_threshold and novel}


class StrictFilterStack:
    def __init__(self, surface, entity, nli, qform, qual, cat_desc):
        self.surface = surface; self.entity = entity; self.nli = nli
        self.qform = qform; self.qual = qual; self.cat_desc = cat_desc

    def evaluate(self, original, candidate):
        results = {}
        s = self.surface.evaluate(original, candidate); results["surface"] = s
        if not s["passed"]:
            r = []
            if s["sim"] < self.surface.sim_threshold: r.append(f"sim={s['sim']:.2f}")
            if s["ppl"] > self.surface.ppl_threshold: r.append(f"ppl={s['ppl']:.0f}")
            if not s["lexical_novel"]: r.append("not_novel")
            return {"passed": False, "first_failure": "surface",
                    "first_failure_reason": " ".join(r), "details": results}
        qf = self.qform.evaluate(original, candidate); results["qform"] = qf
        if not qf["passed"]:
            return {"passed": False, "first_failure": "qform",
                    "first_failure_reason": qf["reason"], "details": results}
        ql = self.qual.evaluate(original, candidate); results["qual"] = ql
        if not ql["passed"]:
            return {"passed": False, "first_failure": "qual",
                    "first_failure_reason": ql["reason"], "details": results}
        e = self.entity.evaluate(original, candidate); results["entity"] = e
        if not e["passed"]:
            return {"passed": False, "first_failure": "entity",
                    "first_failure_reason": e["reason"], "details": results}
        cd = self.cat_desc.evaluate(original, candidate); results["cat_desc"] = cd
        if not cd["passed"]:
            return {"passed": False, "first_failure": "cat_desc",
                    "first_failure_reason": cd["reason"], "details": results}
        n = self.nli.evaluate(original, candidate); results["nli"] = n
        if not n["passed"]:
            return {"passed": False, "first_failure": "nli",
                    "first_failure_reason": n["reason"], "details": results}
        return {"passed": True, "first_failure": None,
                "first_failure_reason": None, "details": results}


def attack_one(rec, rephraser, cls, filters, n_per_prompt):
    qid, q = rec["qid"], rec["question"]
    actual = rec.get("actual_route") or rec.get("baseline_actual_route")
    base = cls.logits_abc(q)
    base_route = LABELS[int(np.argmax(base))]
    base_idx = LABELS.index(base_route)
    cands = rephraser.generate(q, n_per_prompt=n_per_prompt)
    results, success = [], None
    n_passed = 0
    rej = {"surface": 0, "qform": 0, "qual": 0, "entity": 0, "cat_desc": 0, "nli": 0}
    for c in cands:
        cand_text, style = c["text"], c["style"]
        v = filters.evaluate(q, cand_text)
        if not v["passed"]:
            rej[v["first_failure"]] += 1
            results.append({"rephrase": cand_text, "style": style,
                            "passed_strict": False,
                            "first_failure": v["first_failure"],
                            "failure_reason": v["first_failure_reason"],
                            "filter_details": v["details"],
                            "route_after": None,
                            "logit_drop_correct_route": None, "flip": False})
            continue
        n_passed += 1
        cand_logits = cls.logits_abc(cand_text)
        new_route = LABELS[int(np.argmax(cand_logits))]
        drop = float(base[base_idx] - cand_logits[base_idx])
        flip = new_route != base_route
        results.append({"rephrase": cand_text, "style": style,
                        "passed_strict": True,
                        "first_failure": None, "failure_reason": None,
                        "filter_details": v["details"],
                        "route_after": new_route,
                        "logits_abc": cand_logits.tolist(),
                        "logit_drop_correct_route": drop, "flip": flip})
        if flip and success is None:
            success = {"rephrase": cand_text, "style": style,
                       "route_before": base_route, "route_after": new_route,
                       "logit_drop": drop}
    return {"qid": qid, "question": q, "actual_route": actual,
            "baseline_route_predicted": base_route,
            "baseline_logits_abc": base.tolist(),
            "n_candidates_generated": len(cands),
            "n_candidates_passed_strict": n_passed,
            "n_route_flips": sum(1 for r in results if r["flip"]),
            "rejection_counts": rej, "first_success": success,
            "all_candidates": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victims", default="results/attacks/whitebox/victims_107.jsonl")
    ap.add_argument("--out-dir", default="results/attacks/whitebox_complexity_together_v2")
    ap.add_argument("--n-victims", type=int, default=None)
    ap.add_argument("--n-per-prompt", type=int, default=10)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--ppl", type=float, default=300.0)
    ap.add_argument("--nli-threshold", type=float, default=0.65)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--model", default=TOGETHER_MODEL_PAID)
    args = ap.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set TOGETHER_API_KEY")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rephraser = TogetherRephraser(api_key=api_key, model=args.model,
                                  sleep_per_request=args.sleep)
    cls = ClassifierLogits(device=device)
    nlp_full = spacy.load("en_core_web_sm")
    nlp_light = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    surface = SurfaceFilters(sim_threshold=args.sim, ppl_threshold=args.ppl,
                             device=device, nlp=nlp_light)
    entity = EntityPreservationFilter(nlp_full)
    nli = NLIBidirectionalFilter(device=device,
                                 entailment_threshold=args.nli_threshold)
    qform = QuestionFormFilter()
    qual = QualifierBlocklistFilter()
    cat_desc = CategoricalDescriptorFilter(nlp_full)
    filters = StrictFilterStack(surface, entity, nli, qform, qual, cat_desc)

    with open(args.victims) as f:
        victims = [json.loads(l) for l in f]
    if args.n_victims: victims = victims[: args.n_victims]
    print(f"\n[run] {len(victims)} victims, n_per_prompt={args.n_per_prompt}, "
          f"device={device}, sim>={args.sim}, ppl<={args.ppl}, "
          f"nli>={args.nli_threshold}, model={args.model}", flush=True)

    out_path = out / f"attack_v2_n{len(victims)}.jsonl"
    summary_path = out / f"summary_n{len(victims)}.json"
    n_total = n_strict_pass = n_flip = 0
    flips_by_route = {"A": 0, "B": 0, "C": 0}
    flips_by_style = {"embed": 0}
    agg_rej = {"surface": 0, "qform": 0, "qual": 0, "entity": 0, "cat_desc": 0, "nli": 0}
    t_start = time.time()

    with open(out_path, "w") as fo:
        for i, rec in enumerate(victims, 1):
            t0 = time.time()
            try:
                r = attack_one(rec, rephraser, cls, filters, args.n_per_prompt)
            except Exception as e:
                print(f"  [{i}/{len(victims)}] ERROR: {e}", flush=True); continue
            fo.write(json.dumps(r) + "\n"); fo.flush()
            n_total += 1
            if r["n_candidates_passed_strict"] > 0: n_strict_pass += 1
            if r["n_route_flips"] > 0:
                n_flip += 1
                ar = r.get("actual_route") or "?"
                if ar in flips_by_route: flips_by_route[ar] += 1
                if r["first_success"]:
                    flips_by_style[r["first_success"]["style"]] += 1
            for k, v in r["rejection_counts"].items(): agg_rej[k] += v
            print(f"  [{i}/{len(victims)}] qid={r['qid']}  "
                  f"base={r['baseline_route_predicted']}  "
                  f"gen={r['n_candidates_generated']}  "
                  f"pass_strict={r['n_candidates_passed_strict']}  "
                  f"flips={r['n_route_flips']}  "
                  f"wall={time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t_start
    summary = {"n_victims": n_total, "n_any_strict_pass": n_strict_pass,
               "n_any_flip": n_flip,
               "flip_rate": round(n_flip / max(1, n_total), 4),
               "strict_pass_rate": round(n_strict_pass / max(1, n_total), 4),
               "flips_by_actual_route": flips_by_route,
               "flips_by_style": flips_by_style,
               "aggregate_rejection_counts": agg_rej,
               "n_per_prompt": args.n_per_prompt,
               "sim_threshold": args.sim, "ppl_threshold": args.ppl,
               "nli_threshold": args.nli_threshold,
               "device": device, "rephraser": args.model,
               "wall_total_s": round(elapsed, 1)}
    with open(summary_path, "w") as f: json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
