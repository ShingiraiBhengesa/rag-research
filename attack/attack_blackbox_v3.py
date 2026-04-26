"""
Pure black-box adversarial attack on deployed Adaptive-RAG.

Threat model: attacker observes ONLY answer text returned by oracle.
Algorithm: greedy WordNet-synonym word substitution.
            - spacy for POS tagging
            - WordNet for POS-matched synonym lookup
            - Custom greedy search (try each candidate, accept first that flips answer)

For each victim:
  1. Query oracle with original question -> baseline_answer
  2. POS-tag question, identify content words (noun/verb/adj/adv)
  3. For each content word, pull WordNet synonyms matching the POS
  4. Build candidate questions (one synonym swap each)
  5. Query oracle on each candidate within budget; first answer-change = SUCCESS

Every oracle query is logged with full observables for post-hoc analysis
using the trained route detector.
"""

import argparse
import json
import re
import string
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import spacy
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from deployed_adaptive_rag import DeployedAdaptiveRAG


# spacy POS -> WordNet POS
SPACY_TO_WN_POS = {
    "NOUN": wn.NOUN,
    "PROPN": wn.NOUN,  # proper nouns sometimes have synonyms (e.g. countries, titles)
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}
# Only swap these POS classes
CONTENT_POS = set(SPACY_TO_WN_POS.keys())

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "i", "you", "we", "they", "what", "which", "who", "when", "where",
    "why", "how", "do", "does", "did", "can", "could", "would", "should",
}


def normalize_answer(s: str) -> str:
    """MuSiQue official normalization."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def get_wordnet_synonyms(word: str, spacy_pos: str, max_per_word: int = 8) -> List[str]:
    """Return up to N WordNet synonyms for `word` matching the spacy POS tag."""
    wn_pos = SPACY_TO_WN_POS.get(spacy_pos)
    if wn_pos is None:
        return []
    word_lower = word.lower()
    synonyms: List[str] = []
    seen = set()
    for syn in wn.synsets(word_lower, pos=wn_pos):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            # Filter:
            #   - single word only (no multi-word expressions)
            #   - not the original word
            #   - not already collected
            #   - alphabetic only (skip "U.S." style)
            if " " in name:
                continue
            if name.lower() == word_lower:
                continue
            if name.lower() in seen:
                continue
            if not name.isalpha():
                continue
            seen.add(name.lower())
            synonyms.append(name)
            if len(synonyms) >= max_per_word:
                break
        if len(synonyms) >= max_per_word:
            break
    return synonyms


class BlackBoxAttacker:
    def __init__(
        self,
        rag: DeployedAdaptiveRAG,
        spacy_model: str = "en_core_web_sm",
        sim_model_name: str = "paraphrase-MiniLM-L6-v2",
        max_synonyms_per_word: int = 8,
        min_sentence_sim: float = 0.90,
        max_fluency_ppl: float = 200.0,
    ):
        self.rag = rag
        self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        self.sim_model = SentenceTransformer(sim_model_name)
        # Fluency model — GPT-2 small on CPU
        self.lm_tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self.lm.eval()
        self.max_synonyms_per_word = max_synonyms_per_word
        self.min_sentence_sim = min_sentence_sim
        self.max_fluency_ppl = max_fluency_ppl

    def get_swappable_tokens(self, question: str) -> List[Tuple[int, str, str]]:
        """Return [(token_idx, token_text, spacy_pos), ...] for content words."""
        doc = self.nlp(question)
        result = []
        for tok in doc:
            if tok.pos_ not in CONTENT_POS:
                continue
            if tok.text.lower() in STOPWORDS:
                continue
            if not tok.text.isalpha():
                continue
            result.append((tok.i, tok.text, tok.pos_))
        return result

    def build_perturbed_question(
        self, question: str, token_idx: int, replacement: str
    ) -> str:
        """Reconstruct the question with token at `token_idx` replaced.
        Preserves original spacing/punctuation by using spacy's tokens."""
        doc = self.nlp(question)
        parts: List[str] = []
        for tok in doc:
            text = replacement if tok.i == token_idx else tok.text
            # Use spacy's whitespace_ to preserve original spacing
            parts.append(text + tok.whitespace_)
        return "".join(parts)

    def fluency_perplexity(self, text: str) -> float:
        """GPT-2 perplexity. Lower = more fluent. Used as fluency filter."""
        inputs = self.lm_tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.lm(**inputs, labels=inputs["input_ids"])
        return float(torch.exp(out.loss).item())

    def sentence_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two sentence embeddings."""
        embs = self.sim_model.encode([text_a, text_b], show_progress_bar=False)
        a, b = embs[0], embs[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def query_oracle(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = self.rag.query(text)
            answer = result.get("answer", "")
            latency_ms = result.get("latency_ms", -1.0)
            num_sources = result.get("num_sources", -1)
            error = None
        except Exception as e:
            answer = "[ORACLE_ERROR]"
            latency_ms = -1.0
            num_sources = -1
            error = f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        return {
            "perturbed_text": text,
            "system_answer": answer,
            "latency_ms": latency_ms,
            "num_sources": num_sources,
            "wall_clock_s": elapsed,
            "error": error,
        }

    def attack(
        self,
        question: str,
        baseline_answer: str,
        gold_spans: List[str] = None,
        max_queries: int = 30,
    ) -> Dict[str, Any]:
        # use a local alias so the inline closure in #5 below sees it
        gold_spans_for_check = gold_spans or []
        baseline_norm = normalize_answer(baseline_answer)
        oracle_log: List[Dict[str, Any]] = []
        queries_used = 0

        # 1. Baseline query (we keep this as a sanity-check oracle call)
        baseline_record = self.query_oracle(question)
        baseline_record["call_idx"] = len(oracle_log) + 1
        baseline_record["is_baseline"] = True
        baseline_record["replacement"] = None
        baseline_record["original_word"] = None
        baseline_record["answer_norm"] = normalize_answer(baseline_record["system_answer"])
        baseline_record["answer_changed"] = (
            baseline_record["answer_norm"] != baseline_norm
        )
        oracle_log.append(baseline_record)
        queries_used += 1

        # 2. Identify swappable tokens
        swappable = self.get_swappable_tokens(question)
        if not swappable:
            return {
                "attack_success": False,
                "attack_error": "no_content_words",
                "queries_used": queries_used,
                "num_candidates_generated": 0,
                "perturbed_question": None,
                "perturbed_answer": None,
                "oracle_log": oracle_log,
            }

        # 3. Generate (token_idx, original_word, synonym) candidates
        candidates: List[Tuple[int, str, str]] = []
        for tok_idx, tok_text, spacy_pos in swappable:
            synonyms = get_wordnet_synonyms(
                tok_text, spacy_pos, max_per_word=self.max_synonyms_per_word
            )
            for syn in synonyms:
                candidates.append((tok_idx, tok_text, syn))

        n_candidates = len(candidates)
        if n_candidates == 0:
            return {
                "attack_success": False,
                "attack_error": "no_synonyms_found",
                "queries_used": queries_used,
                "num_candidates_generated": 0,
                "perturbed_question": None,
                "perturbed_answer": None,
                "oracle_log": oracle_log,
            }

        # 4. Compute baseline fluency for ratio reference
        baseline_ppl = self.fluency_perplexity(question)

        # Normalize gold spans for the strict-success check
        gold_spans = []  # filled in by caller via attack_kwargs
        # (We accept gold_spans as a method arg now — see signature change.)

        # 5. Greedy: try candidates passing similarity AND fluency filters.
        # Keep searching past first answer-change to find one that's also wrong-vs-gold.
        last_record = None
        first_change_record = None  # fallback: any answer change, even if it still matches gold
        n_filtered_sim = 0
        n_filtered_ppl = 0
        for tok_idx, original_word, synonym in candidates:
            if queries_used >= max_queries:
                break
            perturbed_q = self.build_perturbed_question(question, tok_idx, synonym)

            # Filter 1: sentence similarity
            sim = self.sentence_similarity(question, perturbed_q)
            if sim < self.min_sentence_sim:
                n_filtered_sim += 1
                continue

            # Filter 2: fluency (absolute PPL ceiling)
            ppl = self.fluency_perplexity(perturbed_q)
            if ppl > self.max_fluency_ppl:
                n_filtered_ppl += 1
                continue

            # Passed filters - query oracle
            record = self.query_oracle(perturbed_q)
            record["call_idx"] = len(oracle_log) + 1
            record["is_baseline"] = False
            record["original_word"] = original_word
            record["replacement"] = synonym
            record["sentence_similarity"] = sim
            record["fluency_ppl"] = ppl
            record["answer_norm"] = normalize_answer(record["system_answer"])
            record["answer_changed"] = record["answer_norm"] != baseline_norm
            # Strict check: does perturbed answer match any gold span?
            record["matches_gold"] = any(
                normalize_answer(g) == record["answer_norm"]
                for g in (gold_spans_for_check or [])
                if g
            )
            oracle_log.append(record)
            queries_used += 1

            # STRICT SUCCESS: answer changed AND new answer is wrong-vs-gold
            if record["answer_changed"] and not record["matches_gold"]:
                return {
                    "attack_success": True,
                    "attack_success_strict": True,
                    "attack_error": None,
                    "queries_used": queries_used,
                    "num_candidates_generated": n_candidates,
                    "num_candidates_filtered_sim": n_filtered_sim,
                    "num_candidates_filtered_ppl": n_filtered_ppl,
                    "perturbed_question": perturbed_q,
                    "perturbed_answer": record["system_answer"],
                    "successful_swap": {
                        "original": original_word,
                        "replacement": synonym,
                        "sentence_similarity": sim,
                        "fluency_ppl": ppl,
                    },
                    "oracle_log": oracle_log,
                }
            # Track first answer change for soft-success fallback
            if record["answer_changed"] and first_change_record is None:
                first_change_record = record
            last_record = record

        # No strict-success found. Report soft-success if we at least changed the answer.
        soft_success = first_change_record is not None
        return {
            "attack_success": False,  # primary metric = strict
            "attack_success_strict": False,
            "attack_success_soft": soft_success,
            "attack_error": None,
            "queries_used": queries_used,
            "num_candidates_generated": n_candidates,
            "num_candidates_filtered_sim": n_filtered_sim,
            "num_candidates_filtered_ppl": n_filtered_ppl,
            "perturbed_question": (
                first_change_record["perturbed_text"]
                if first_change_record else
                (last_record["perturbed_text"] if last_record else None)
            ),
            "perturbed_answer": (
                first_change_record["system_answer"]
                if first_change_record else
                (last_record["system_answer"] if last_record else None)
            ),
            "successful_swap": None,
            "oracle_log": oracle_log,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-oracle-queries", type=int, default=30)
    ap.add_argument("--max-synonyms-per-word", type=int, default=8)
    ap.add_argument("--min-sentence-sim", type=float, default=0.90)
    ap.add_argument("--max-fluency-ppl", type=float, default=200.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    victims = []
    with open(args.input_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if args.only_correct and not r.get("exact_match"):
                continue
            victims.append(r)
    if args.start:
        victims = victims[args.start:]
    if args.n:
        victims = victims[: args.n]

    print(f"Will attack {len(victims)} victims from {args.input_jsonl}")
    print(f"Budget: {args.max_oracle_queries} oracle queries per victim")

    rag = DeployedAdaptiveRAG()
    attacker = BlackBoxAttacker(
        rag=rag,
        max_synonyms_per_word=args.max_synonyms_per_word,
        min_sentence_sim=args.min_sentence_sim,
        max_fluency_ppl=args.max_fluency_ppl,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = open(args.output_jsonl, "a")

    successes = 0
    total_oracle_calls = 0
    t_start = time.time()
    for i, v in enumerate(victims):
        qid = v["qid"]
        question = v["question"]
        baseline_answer = v["system_answer"]
        print(f"\n[{i+1}/{len(victims)}] qid={qid}")
        print(f"  question: {question[:80]}")
        print(f"  baseline answer: {baseline_answer[:60]!r}")

        t0 = time.perf_counter()
        result = attacker.attack(
            question=question,
            baseline_answer=baseline_answer,
            gold_spans=v.get("gold_answer_spans") or [],
            max_queries=args.max_oracle_queries,
        )
        wall_clock = time.perf_counter() - t0

        record = {
            "qid": qid,
            "question": question,
            "baseline_answer": baseline_answer,
            "baseline_actual_route": v.get("actual_route"),
            "baseline_em": v.get("exact_match"),
            "gold_answer_spans": v.get("gold_answer_spans"),
            "wall_clock_s": wall_clock,
            **result,
        }

        out.write(json.dumps(record) + "\n")
        out.flush()

        if result["attack_success"]:
            successes += 1
        total_oracle_calls += result["queries_used"]
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(victims) - (i + 1)) / rate if rate > 0 else 0

        status = "SUCCESS" if result["attack_success"] else "failed"
        print(
            f"  {status:8s}  queries={result['queries_used']:3d}  "
            f"cand={result.get('num_candidates_generated', 0)}  "
            f"filt_sim={result.get('num_candidates_filtered_sim', 0)}  "
            f"filt_ppl={result.get('num_candidates_filtered_ppl', 0)}  "
            f"wall_s={wall_clock:5.1f}"
        )
        print(
            f"  running: {successes}/{i+1} successes ({100*successes/(i+1):.1f}%)  "
            f"avg_queries={total_oracle_calls/(i+1):.1f}  ETA={eta/60:.1f}m"
        )
        if result["attack_success"]:
            swap = result.get("successful_swap", {})
            print(f"  swap: {swap.get('original')!r} -> {swap.get('replacement')!r}")
            print(f"  perturbed: {result['perturbed_question'][:80]}")
            print(f"  new answer: {result['perturbed_answer'][:60]!r}")
        elif result.get("attack_error"):
            print(f"  ERROR: {result['attack_error']}")

    out.close()
    print(f"\n{'='*60}")
    print(f"Done: {successes}/{len(victims)} successes ({100*successes/len(victims):.1f}%)")
    print(f"Total oracle calls: {total_oracle_calls}")


if __name__ == "__main__":
    main()
