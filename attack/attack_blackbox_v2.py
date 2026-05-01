"""
Pure black-box adversarial attack on deployed Adaptive-RAG.

Threat model: attacker observes ONLY answer text returned by oracle.
Algorithm: greedy word substitution.
            - Use TextAttack's WordSwapEmbedding for synonym candidates
            - Use TextAttack's constraints for filtering
            - Custom greedy search loop (no TextAttack goal/search abstraction)
"""

import argparse
import json
import re
import string
import time
from pathlib import Path
from typing import Dict, Any, List

from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import (
    StopwordModification,
    RepeatModification,
)
from textattack.shared import AttackedText

from deployed_adaptive_rag import DeployedAdaptiveRAG


def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


class BlackBoxAttacker:
    def __init__(
        self,
        rag: DeployedAdaptiveRAG,
        max_candidates: int = 50,
        min_cos_sim: float = 0.5,
    ):
        self.rag = rag
        self.transformation = WordSwapEmbedding(max_candidates=max_candidates)
        self.constraints = [
            RepeatModification(),
            StopwordModification(),
            WordEmbeddingDistance(min_cos_sim=min_cos_sim),
        ]

    def get_candidates(self, attacked_text: AttackedText) -> List[AttackedText]:
        candidates = self.transformation(attacked_text, self.constraints)
        return list(candidates)

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
        max_queries: int = 30,
    ) -> Dict[str, Any]:
        baseline_norm = normalize_answer(baseline_answer)
        oracle_log: List[Dict[str, Any]] = []
        queries_used = 0

        # 1. Baseline query
        baseline_record = self.query_oracle(question)
        baseline_record["call_idx"] = len(oracle_log) + 1
        baseline_record["is_baseline"] = True
        baseline_record["answer_norm"] = normalize_answer(baseline_record["system_answer"])
        baseline_record["answer_changed"] = (
            baseline_record["answer_norm"] != baseline_norm
        )
        oracle_log.append(baseline_record)
        queries_used += 1

        # 2. Generate candidate perturbations
        attacked_text = AttackedText(question)
        try:
            candidates = self.get_candidates(attacked_text)
        except Exception as e:
            return {
                "attack_success": False,
                "attack_error": f"candidate_gen: {type(e).__name__}: {e}",
                "queries_used": queries_used,
                "num_candidates_generated": 0,
                "perturbed_question": None,
                "perturbed_answer": None,
                "oracle_log": oracle_log,
            }

        n_candidates = len(candidates)
        if not candidates:
            return {
                "attack_success": False,
                "attack_error": "no_candidates_generated",
                "queries_used": queries_used,
                "num_candidates_generated": 0,
                "perturbed_question": None,
                "perturbed_answer": None,
                "oracle_log": oracle_log,
            }

        # 3. Try each candidate, stop on success
        last_record = None
        for cand in candidates:
            if queries_used >= max_queries:
                break
            cand_text = cand.text
            record = self.query_oracle(cand_text)
            record["call_idx"] = len(oracle_log) + 1
            record["is_baseline"] = False
            record["answer_norm"] = normalize_answer(record["system_answer"])
            record["answer_changed"] = record["answer_norm"] != baseline_norm
            oracle_log.append(record)
            queries_used += 1

            if record["answer_changed"]:
                return {
                    "attack_success": True,
                    "attack_error": None,
                    "queries_used": queries_used,
                    "num_candidates_generated": n_candidates,
                    "perturbed_question": cand_text,
                    "perturbed_answer": record["system_answer"],
                    "oracle_log": oracle_log,
                }
            last_record = record

        return {
            "attack_success": False,
            "attack_error": None,
            "queries_used": queries_used,
            "num_candidates_generated": n_candidates,
            "perturbed_question": last_record["perturbed_text"] if last_record else None,
            "perturbed_answer": last_record["system_answer"] if last_record else None,
            "oracle_log": oracle_log,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-oracle-queries", type=int, default=30)
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
    attacker = BlackBoxAttacker(rag=rag, max_candidates=50, min_cos_sim=0.5)

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
        print(f"  {status:8s}  queries={result['queries_used']:3d}  "
              f"candidates={result.get('num_candidates_generated', 0)}  "
              f"wall_s={wall_clock:5.1f}")
        print(
            f"  running: {successes}/{i+1} successes ({100*successes/(i+1):.1f}%)  "
            f"avg_queries={total_oracle_calls/(i+1):.1f}  ETA={eta/60:.1f}m"
        )
        if result["attack_success"]:
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
