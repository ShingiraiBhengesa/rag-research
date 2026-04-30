"""
attack_whitebox_v1.py - White-box adversarial attack on the Adaptive-RAG
routing classifier (T5-Large), TextFooler-style.

Threat model:
  Attacker has access to the classifier weights and can read logits over
  routes {A, B, C}. They cannot see retrieval results or LLM outputs.
  Goal: flip the predicted route via single-word substitutions while
  preserving question semantics and fluency.

Differences from attack_blackbox_v3.py:
  - Goal function: argmax(logits) != original_route (not answer-text change)
  - Search signal: classifier logit drop on word masking (vs success/fail)
  - Search method: TextFooler-style word-importance ranking, then greedy swap
  - Multiple swaps may be needed per attack (black-box succeeded with 1)

Reuses from black-box pipeline:
  - WordNet synonym generation
  - spaCy POS tagging
  - sentence-transformers similarity filter (>= 0.90)
  - GPT-2 fluency filter (PPL <= 200)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import spacy
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


# ---------------------------------------------------------------------------
# WordNet helpers (same as black-box v3)
# ---------------------------------------------------------------------------

WN_POS_MAP = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}


def wordnet_synonyms(word, spacy_pos, max_syns=8):
    """Return up to max_syns synonyms for `word` with the given spaCy POS."""
    wn_pos = WN_POS_MAP.get(spacy_pos)
    if wn_pos is None:
        return []
    seen = set()
    out = []
    target = word.lower()
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            cand = lemma.name().replace("_", " ")
            if cand.lower() == target or " " in cand:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
            if len(out) >= max_syns:
                return out
    return out


def is_content_word(token):
    """spaCy tokens we will consider for swap candidates."""
    if token.is_stop or token.is_punct or token.is_space:
        return False
    if token.pos_ not in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
        return False
    if not token.text.strip():
        return False
    return True


def replace_token(tokens, idx, new_word):
    """Rebuild question text replacing the token at `idx` with new_word."""
    out = list(tokens)
    out[idx] = new_word
    return " ".join(out)


# ---------------------------------------------------------------------------
# Classifier wrapper - extends classifier_wrapper.py to expose logits cleanly
# ---------------------------------------------------------------------------

class ClassifierLogitWrapper:
    """T5-Large complexity classifier exposing logits over {A, B, C}."""

    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        print("Loading tokenizer from " + str(model_path), flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        print("Loading classifier model from " + str(model_path), flush=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.model.eval()

        self.labels = ["A", "B", "C"]
        self.label_token_ids = {
            lab: self.tokenizer.encode(lab, add_special_tokens=False)[0]
            for lab in self.labels
        }
        print("Label token IDs: " + str(self.label_token_ids), flush=True)

    @torch.no_grad()
    def logits(self, text):
        """Return raw logits (not softmaxed) over [A, B, C]."""
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=384,
            truncation=True, padding=True,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=1,
            output_scores=True, return_dict_in_generate=True,
        )
        first_token_logits = outputs.scores[0][0]
        abc = torch.tensor([
            first_token_logits[self.label_token_ids["A"]].item(),
            first_token_logits[self.label_token_ids["B"]].item(),
            first_token_logits[self.label_token_ids["C"]].item(),
        ])
        return abc.numpy()

    def predict_route(self, text):
        """Return (predicted_route_label, logits_array)."""
        lg = self.logits(text)
        return self.labels[int(np.argmax(lg))], lg


# ---------------------------------------------------------------------------
# Filter components (sim + PPL) - reused from black-box v3
# ---------------------------------------------------------------------------

class FilterStack:
    def __init__(
        self,
        sim_model_name="paraphrase-MiniLM-L6-v2",
        min_sentence_sim=0.90,
        max_fluency_ppl=200.0,
        ppl_device="cpu",
    ):
        print("Loading sentence-similarity model: " + sim_model_name, flush=True)
        self.sim_model = SentenceTransformer(sim_model_name)
        self.min_sentence_sim = min_sentence_sim

        print("Loading GPT-2 small for fluency", flush=True)
        self.lm_tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(ppl_device)
        self.lm.eval()
        self.ppl_device = ppl_device
        self.max_fluency_ppl = max_fluency_ppl

    def sentence_similarity(self, a, b):
        embs = self.sim_model.encode([a, b], show_progress_bar=False)
        v1, v2 = embs[0], embs[1]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    @torch.no_grad()
    def fluency_perplexity(self, text):
        inputs = self.lm_tok(text, return_tensors="pt").to(self.ppl_device)
        out = self.lm(**inputs, labels=inputs["input_ids"])
        return float(torch.exp(out.loss).item())

    def passes(self, original, candidate):
        """Return (passed, reason, sim, ppl). reason='' if passed."""
        sim = self.sentence_similarity(original, candidate)
        if sim < self.min_sentence_sim:
            return (False, "sim", sim, -1.0)
        ppl = self.fluency_perplexity(candidate)
        if ppl > self.max_fluency_ppl:
            return (False, "ppl", sim, ppl)
        return (True, "", sim, ppl)


# ---------------------------------------------------------------------------
# Attack core - TextFooler-style greedy
# ---------------------------------------------------------------------------

class WhiteBoxAttacker:
    def __init__(
        self,
        classifier,
        filters,
        spacy_model="en_core_web_sm",
        max_synonyms_per_word=8,
        unk_token="<unk>",
    ):
        self.cls = classifier
        self.filters = filters
        self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        self.max_synonyms_per_word = max_synonyms_per_word
        self.unk_token = unk_token

    def compute_word_importance(self, question, original_route, original_logits):
        """For each content word, compute logit drop if it is masked."""
        doc = self.nlp(question)
        tokens = [t.text for t in doc]
        candidates = []
        route_idx = self.cls.labels.index(original_route)
        for i, tok in enumerate(doc):
            if not is_content_word(tok):
                continue
            masked = replace_token(tokens, i, self.unk_token)
            masked_logits = self.cls.logits(masked)
            drop = float(original_logits[route_idx] - masked_logits[route_idx])
            candidates.append((i, tok.text, tok.pos_, drop))
        candidates.sort(key=lambda x: x[3], reverse=True)
        return candidates

    def attack(self, question, gold_spans=None, max_queries=100):
        oracle_log = []
        original_route, original_logits = self.cls.predict_route(question)
        oracle_log.append({
            "call_idx": 1,
            "type": "baseline",
            "text": question,
            "logits": original_logits.tolist(),
            "predicted_route": original_route,
        })
        queries_used = 1

        importance = self.compute_word_importance(
            question, original_route, original_logits,
        )
        queries_used += len(importance)
        for k, (i, w, pos, score) in enumerate(importance):
            oracle_log.append({
                "call_idx": queries_used + 1 - len(importance) + k,
                "type": "importance_mask",
                "token_idx": i,
                "word": w,
                "pos": pos,
                "importance": score,
            })

        if not importance:
            return {
                "attack_success": False,
                "attack_error": "no content words",
                "original_route": original_route,
                "perturbed_route": original_route,
                "queries_used": queries_used,
                "successful_swap": None,
                "swap_chain": [],
                "oracle_log": oracle_log,
            }

        doc = self.nlp(question)
        current_tokens = [t.text for t in doc]
        current_question = question
        current_logits = original_logits
        swap_chain = []
        n_filtered_sim = 0
        n_filtered_ppl = 0
        route_idx = self.cls.labels.index(original_route)

        for (idx, original_word, pos, score) in importance:
            if queries_used >= max_queries:
                break

            cands = wordnet_synonyms(original_word, pos, self.max_synonyms_per_word)
            if not cands:
                continue

            best_candidate = None
            best_logits = None
            best_sim = None
            best_ppl = None
            best_drop = -float("inf")

            for syn in cands:
                if queries_used >= max_queries:
                    break
                trial_question = replace_token(current_tokens, idx, syn)
                if trial_question == current_question:
                    continue

                passed, reason, sim, ppl = self.filters.passes(question, trial_question)
                if not passed:
                    if reason == "sim":
                        n_filtered_sim += 1
                    elif reason == "ppl":
                        n_filtered_ppl += 1
                    continue

                trial_logits = self.cls.logits(trial_question)
                queries_used += 1
                trial_pred = self.cls.labels[int(np.argmax(trial_logits))]

                oracle_log.append({
                    "call_idx": queries_used,
                    "type": "candidate",
                    "token_idx": idx,
                    "original_word": original_word,
                    "replacement": syn,
                    "text": trial_question,
                    "logits": trial_logits.tolist(),
                    "predicted_route": trial_pred,
                    "sim": sim,
                    "ppl": ppl,
                })

                if trial_pred != original_route:
                    swap_chain.append({
                        "token_idx": idx,
                        "original_word": original_word,
                        "replacement": syn,
                        "sim": sim,
                        "ppl": ppl,
                    })
                    return {
                        "attack_success": True,
                        "attack_error": None,
                        "original_route": original_route,
                        "perturbed_route": trial_pred,
                        "queries_used": queries_used,
                        "successful_swap": swap_chain[-1],
                        "swap_chain": swap_chain,
                        "n_filtered_sim": n_filtered_sim,
                        "n_filtered_ppl": n_filtered_ppl,
                        "perturbed_question": trial_question,
                        "oracle_log": oracle_log,
                    }

                drop = float(current_logits[route_idx] - trial_logits[route_idx])
                if drop > best_drop:
                    best_drop = drop
                    best_candidate = syn
                    best_logits = trial_logits
                    best_sim = sim
                    best_ppl = ppl

            if best_candidate is not None and best_drop > 0:
                current_tokens[idx] = best_candidate
                current_question = " ".join(current_tokens)
                current_logits = best_logits
                swap_chain.append({
                    "token_idx": idx,
                    "original_word": original_word,
                    "replacement": best_candidate,
                    "sim": best_sim,
                    "ppl": best_ppl,
                    "logit_drop": best_drop,
                })

        final_pred = self.cls.labels[int(np.argmax(current_logits))]
        return {
            "attack_success": False,
            "attack_error": None,
            "original_route": original_route,
            "perturbed_route": final_pred,
            "queries_used": queries_used,
            "successful_swap": None,
            "swap_chain": swap_chain,
            "n_filtered_sim": n_filtered_sim,
            "n_filtered_ppl": n_filtered_ppl,
            "perturbed_question": current_question,
            "oracle_log": oracle_log,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument(
        "--model-path", type=str,
        default=(
            "/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
            "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
            "flan_t5_xxl/epoch/30"
        ),
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-oracle-queries", type=int, default=100)
    ap.add_argument("--max-synonyms-per-word", type=int, default=8)
    ap.add_argument("--min-sentence-sim", type=float, default=0.90)
    ap.add_argument("--max-fluency-ppl", type=float, default=200.0)
    args = ap.parse_args()

    with open(args.input_jsonl) as f:
        all_records = [json.loads(line) for line in f]
    if args.start:
        all_records = all_records[args.start:]
    if args.n is not None:
        all_records = all_records[: args.n]
    print("Will attack " + str(len(all_records)) + " victims from "
          + str(args.input_jsonl), flush=True)
    print("Budget: " + str(args.max_oracle_queries) + " queries / victim",
          flush=True)

    cls = ClassifierLogitWrapper(args.model_path, device=args.device)
    filters = FilterStack(
        min_sentence_sim=args.min_sentence_sim,
        max_fluency_ppl=args.max_fluency_ppl,
    )
    attacker = WhiteBoxAttacker(
        classifier=cls,
        filters=filters,
        max_synonyms_per_word=args.max_synonyms_per_word,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_success = 0
    t0 = time.time()
    with open(args.output_jsonl, "w") as fo:
        for i, rec in enumerate(all_records, 1):
            qid = rec["qid"]
            question = rec["question"]
            print("\n[" + str(i) + "/" + str(len(all_records)) + "] qid=" + str(qid), flush=True)
            print("  question: " + question[:90], flush=True)

            t_start = time.time()
            try:
                result = attacker.attack(
                    question=question,
                    gold_spans=rec.get("gold_answer_spans"),
                    max_queries=args.max_oracle_queries,
                )
            except Exception as e:
                result = {
                    "attack_success": False,
                    "attack_error": type(e).__name__ + ": " + str(e),
                    "original_route": rec.get("baseline_actual_route"),
                    "perturbed_route": None,
                    "queries_used": 0,
                    "successful_swap": None,
                    "swap_chain": [],
                    "oracle_log": [],
                }
            wall = time.time() - t_start

            result.update({
                "qid": qid,
                "question": question,
                "baseline_answer": rec.get("system_answer"),
                "gold_answer_spans": rec.get("gold_answer_spans"),
                "baseline_actual_route": rec.get("baseline_actual_route") or rec.get("actual_route"),
                "baseline_em": rec.get("exact_match"),
                "wall_clock_s": wall,
            })

            fo.write(json.dumps(result) + "\n")
            fo.flush()

            if result.get("attack_success"):
                n_success += 1
                sw = result.get("successful_swap") or {}
                print("  SUCCESS  queries=" + str(result["queries_used"])
                      + "  swaps=" + str(len(result.get("swap_chain", [])))
                      + "  wall=" + str(round(wall, 1)) + "s", flush=True)
                print("  swap: " + repr(sw.get("original_word"))
                      + " -> " + repr(sw.get("replacement")), flush=True)
                print("  route: " + str(result["original_route"])
                      + " -> " + str(result["perturbed_route"]), flush=True)
            else:
                err = result.get("attack_error") or "no flip found"
                print("  failed   queries=" + str(result["queries_used"])
                      + "  swaps=" + str(len(result.get("swap_chain", [])))
                      + "  wall=" + str(round(wall, 1)) + "s  (" + err + ")",
                      flush=True)

            elapsed = time.time() - t0
            rate = n_success / i
            eta = (elapsed / i) * (len(all_records) - i) / 60.0
            print("  running: " + str(n_success) + "/" + str(i)
                  + " successes (" + str(round(100 * rate, 1)) + "%)  "
                  + "ETA=" + str(round(eta, 1)) + "m", flush=True)

    print("\nDone: " + str(n_success) + "/" + str(len(all_records)) + " successes",
          flush=True)


if __name__ == "__main__":
    main()
