"""
Pure black-box adversarial attack on deployed Adaptive-RAG.

Threat model:
    Attacker observes ONLY the answer string returned by wrapper.query(text).
    No latency, no num_sources, no internal state.
    
Attack mechanism:
    TextFooler-style word substitution:
        - GreedyWordSwapWIR (greedy with word importance ranking)
        - WordSwapEmbedding (counter-fitted GloVe synonyms)
        - WordEmbeddingDistance constraint (cosine sim >= 0.5)
        - StopwordModification constraint (don't swap stopwords)
    
    Goal: produce a perturbation Q' such that 
          normalize(wrapper.query(Q').answer) != normalize(baseline_answer)
    
Every oracle query is logged with full observables for post-hoc analysis
using the trained route detector (route_detector.py).

Output JSONL contains one record per victim, with a sub-list of all queries
made during the attack search.
"""

import argparse
import json
import re
import string
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# TextAttack imports
from textattack import Attack
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import (
    GoalFunctionResult,
    GoalFunctionResultStatus,
)
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import (
    StopwordModification,
    RepeatModification,
)
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared import AttackedText
from textattack.attack_results import (
    SuccessfulAttackResult,
    FailedAttackResult,
)

# Wrapper
from deployed_adaptive_rag import DeployedAdaptiveRAG


# --- MuSiQue official normalization (for goal comparison) -----------------

def normalize_answer(s: str) -> str:
    """Verbatim from official_evaluation/musique/metrics/answer.py."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


# --- Oracle adapter (wrapper -> TextAttack-compatible interface) ----------

class BlackBoxOracle:
    """Wraps DeployedAdaptiveRAG.query() and logs every call."""

    def __init__(self, rag: DeployedAdaptiveRAG):
        self.rag = rag
        self.call_log: List[Dict[str, Any]] = []
        self.call_count = 0

    def query(self, text: str) -> Dict[str, Any]:
        """Execute one oracle query, log full observables, return them."""
        t0 = time.perf_counter()
        try:
            result = self.rag.query(text)
            answer = result["answer"]
            latency_ms = result["latency_ms"]
            num_sources = result["num_sources"]
        except Exception as e:
            # Treat oracle errors as a failed query — record but don't crash attack
            answer = "[ORACLE_ERROR]"
            latency_ms = -1.0
            num_sources = -1
        elapsed = time.perf_counter() - t0
        self.call_count += 1
        record = {
            "call_idx": self.call_count,
            "perturbed_text": text,
            "system_answer": answer,
            "latency_ms": latency_ms,
            "num_sources": num_sources,
            "wall_clock_s": elapsed,
        }
        self.call_log.append(record)
        return record

    def reset(self) -> None:
        self.call_log = []
        self.call_count = 0


# --- TextAttack goal function (pure black-box) ----------------------------

class AnswerChangedGoal(GoalFunction):
    """
    Goal function: succeed when the system answer is different from baseline.
    
    The goal is "non-classification" style — TextAttack normally returns logits,
    but we return a pseudo-score based on whether the answer differs from the
    target. Score = 1.0 if different (success), 0.0 if same (still searching).
    """

    def __init__(
        self,
        oracle: BlackBoxOracle,
        baseline_answer: str,
        max_oracle_queries: int = 30,
        **kwargs,
    ):
        # GoalFunction expects a model_wrapper; we pass None and override behavior
        self.oracle = oracle
        self.baseline_answer_norm = normalize_answer(baseline_answer)
        self.max_oracle_queries = max_oracle_queries
        # GoalFunction parent expects these attributes
        self.use_cache = False
        self.query_budget = max_oracle_queries
        self.num_queries = 0
        self.maximizable = False
        self.model = None
        self.model_wrapper = None
        self.tokenizer = None
        self.model_cache = None

    def init_attack_example(self, attacked_text, ground_truth_output):
        """Reset state for a new attack."""
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def _get_score(self, model_output, attacked_text):
        """Score this candidate: 1.0 if answer changed, 0.0 if same."""
        return float(model_output)  # model_output IS the score (0/1) we put there

    def _is_goal_complete(self, model_output, attacked_text):
        return bool(model_output)

    def _get_displayed_output(self, raw_output):
        return str(raw_output)

    def _process_model_outputs(self, inputs, outputs):
        """No-op: our 'model_output' is already the score we want."""
        return outputs

    @property
    def _goal_function_result_type(self):
        """Use the generic GoalFunctionResult class."""
        return GoalFunctionResult

    def _call_model_uncached(self, attacked_text_list):
        """Send candidate texts to the oracle, return per-candidate scores."""
        scores = []
        for at in attacked_text_list:
            text = at.text
            self.num_queries += 1
            record = self.oracle.query(text)
            # Annotate with attack metadata
            record["is_baseline"] = (text == self.initial_attacked_text.text)
            record["baseline_answer_norm"] = self.baseline_answer_norm
            ans_norm = normalize_answer(record["system_answer"])
            record["answer_norm"] = ans_norm
            record["answer_changed"] = (ans_norm != self.baseline_answer_norm)
            scores.append(1.0 if record["answer_changed"] else 0.0)
        return scores

    def _call_model(self, attacked_text_list):
        return self._call_model_uncached(attacked_text_list)

    def get_results(self, attacked_text_list, check_skip=False):
        results, search_over = [], False
        if self.num_queries >= self.query_budget:
            search_over = True
            return results, search_over
        # Filter to budget
        budget_left = self.query_budget - self.num_queries
        attacked_text_list = attacked_text_list[:budget_left]
        if not attacked_text_list:
            return [], True
        scores = self._call_model(attacked_text_list)
        for at, score in zip(attacked_text_list, scores):
            goal_status = (
                GoalFunctionResultStatus.SUCCEEDED
                if self._is_goal_complete(score, at)
                else GoalFunctionResultStatus.SEARCHING
            )
            result = GoalFunctionResult(
                attacked_text=at,
                raw_output=score,
                output=score,
                goal_status=goal_status,
                score=score,
                num_queries=self.num_queries,
                ground_truth_output=self.ground_truth_output,
            )
            results.append(result)
        if self.num_queries >= self.query_budget:
            search_over = True
        return results, search_over

    def get_result(self, attacked_text, check_skip=False):
        results, search_over = self.get_results([attacked_text], check_skip=check_skip)
        if results:
            return results[0], search_over
        return None, search_over


# --- Attack assembly --------------------------------------------------------

def build_attack(
    oracle: BlackBoxOracle,
    baseline_answer: str,
    max_oracle_queries: int = 30,
    min_cos_sim: float = 0.5,
    max_candidates: int = 50,
) -> Attack:
    """
    Assemble TextFooler-style attack against the black-box oracle.
    """
    goal = AnswerChangedGoal(
        oracle=oracle,
        baseline_answer=baseline_answer,
        max_oracle_queries=max_oracle_queries,
    )
    transformation = WordSwapEmbedding(max_candidates=max_candidates)
    constraints = [
        RepeatModification(),  # don't swap same word twice
        StopwordModification(),  # don't swap stopwords
        WordEmbeddingDistance(min_cos_sim=min_cos_sim),
    ]
    search_method = GreedyWordSwapWIR(wir_method="delete")
    return Attack(goal, constraints, transformation, search_method)


# --- Run attack on one victim -----------------------------------------------

def attack_one_question(
    rag: DeployedAdaptiveRAG,
    question: str,
    baseline_answer: str,
    qid: str,
    max_oracle_queries: int = 30,
) -> Dict[str, Any]:
    """Run black-box attack on a single victim question."""
    oracle = BlackBoxOracle(rag)
    attack = build_attack(
        oracle=oracle,
        baseline_answer=baseline_answer,
        max_oracle_queries=max_oracle_queries,
    )
    initial_text = AttackedText(question)
    t0 = time.perf_counter()
    try:
        # AttackedText goes in as the "input"; we don't have a true ground truth class
        # so we pass 0 as a placeholder
        initial_result, _ = attack.goal_function.init_attack_example(initial_text, 0)
        if initial_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            # Baseline already differs (shouldn't happen since we initialized w/ baseline)
            final_result = initial_result
        else:
            final_result = attack.search_method.perform_search(initial_result)
        elapsed = time.perf_counter() - t0
        success = final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "qid": qid,
            "question": question,
            "baseline_answer": baseline_answer,
            "attack_success": False,
            "attack_error": f"{type(e).__name__}: {e}",
            "num_oracle_queries": oracle.call_count,
            "wall_clock_s": elapsed,
            "perturbed_question": None,
            "perturbed_answer": None,
            "oracle_log": oracle.call_log,
        }

    perturbed_text = final_result.attacked_text.text if final_result else None
    perturbed_answer = None
    if oracle.call_log:
        # Find the matching log entry for the final perturbed text
        for entry in reversed(oracle.call_log):
            if entry["perturbed_text"] == perturbed_text:
                perturbed_answer = entry["system_answer"]
                break
        if perturbed_answer is None:
            perturbed_answer = oracle.call_log[-1]["system_answer"]

    return {
        "qid": qid,
        "question": question,
        "baseline_answer": baseline_answer,
        "attack_success": bool(success),
        "num_oracle_queries": oracle.call_count,
        "wall_clock_s": elapsed,
        "perturbed_question": perturbed_text,
        "perturbed_answer": perturbed_answer,
        "oracle_log": oracle.call_log,
    }


# --- Run attack over a dataset ---------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="baseline_musique_route_X.jsonl input",
    )
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-oracle-queries", type=int, default=30)
    ap.add_argument("--n", type=int, default=None, help="limit to N victims (None=all)")
    ap.add_argument(
        "--only-correct",
        action="store_true",
        help="only attack queries where baseline got the right answer",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="skip first N victims (for resuming)",
    )
    args = ap.parse_args()

    # Load victims
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

    # Init wrapper once
    rag = DeployedAdaptiveRAG()

    # Run attacks, append to output JSONL after each victim (for resumability)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = open(args.output_jsonl, "a")  # append mode

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
        result = attack_one_question(
            rag=rag,
            question=question,
            baseline_answer=baseline_answer,
            qid=qid,
            max_oracle_queries=args.max_oracle_queries,
        )
        # Add baseline metadata
        result["baseline_actual_route"] = v.get("actual_route")
        result["baseline_predicted_route"] = v.get("predicted_route")
        result["baseline_em"] = v.get("exact_match")
        result["baseline_loose"] = v.get("loose_match")
        result["gold_answer_spans"] = v.get("gold_answer_spans")

        out.write(json.dumps(result) + "\n")
        out.flush()
        if result["attack_success"]:
            successes += 1
        total_oracle_calls += result["num_oracle_queries"]
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(victims) - (i + 1)) / rate if rate > 0 else 0
        print(
            f"  {'SUCCESS' if result['attack_success'] else 'failed':8s}  "
            f"queries_used={result['num_oracle_queries']:3d}  "
            f"wall_s={result['wall_clock_s']:5.1f}"
        )
        print(
            f"  running: {successes}/{i+1} successes "
            f"({100*successes/(i+1):.1f}%)  "
            f"avg_queries={total_oracle_calls/(i+1):.1f}  "
            f"ETA={eta/60:.1f}m"
        )
        if result["attack_success"] and result["perturbed_question"]:
            print(f"  perturbed: {result['perturbed_question'][:80]}")
            print(f"  new answer: {result['perturbed_answer'][:60]!r}")

    out.close()
    print(f"\n{'='*60}")
    print(f"Done: {successes}/{len(victims)} successes "
          f"({100*successes/len(victims):.1f}%)")
    print(f"Total oracle calls: {total_oracle_calls}")
    print(f"Avg queries per victim: {total_oracle_calls/len(victims):.1f}")


if __name__ == "__main__":
    main()
