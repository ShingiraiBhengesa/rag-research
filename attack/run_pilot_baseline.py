"""
Pilot baseline: run N MuSiQue Route C queries through the deployed
system via DeployedAdaptiveRAG wrapper. Capture all black-box observables
and compute exact-match accuracy vs gold answers.

Usage:
    python run_pilot_baseline.py --n 20
    python run_pilot_baseline.py --n 50 --target-route C
    python run_pilot_baseline.py --n 100 --all-routes
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from deployed_adaptive_rag import DeployedAdaptiveRAG, ADAPTIVE_RAG_REPO


# Data paths
CLASSIFIER_PREDS_PATH = (
    ADAPTIVE_RAG_REPO
    / "classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/epoch/30/predict/dict_id_pred_results.json"
)
MUSIQUE_DEV_PATH = ADAPTIVE_RAG_REPO / "processed_data/musique/test_subsampled.jsonl"

# Output location
OUTPUT_DIR = Path.home() / "adaptive-rag/attack/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_musique_gold():
    """Load qid -> gold data (question, answer spans)."""
    gold = {}
    with open(MUSIQUE_DEV_PATH, "r") as f:
        for line in f:
            inst = json.loads(line)
            qid = inst["question_id"]
            spans = inst["answers_objects"][0]["spans"]
            gold[qid] = {
                "question": inst["question_text"],
                "answer_spans": spans,
                "primary_answer": spans[0] if spans else "",
            }
    return gold


def load_classifier_predictions():
    """Load qid -> predicted route."""
    with open(CLASSIFIER_PREDS_PATH, "r") as f:
        data = json.load(f)
    # Return qid -> route letter
    return {qid: v["prediction"] for qid, v in data.items()}


def exact_match(pred: str, gold_spans: list) -> bool:
    """MuSiQue official EM.
    Verbatim from official_evaluation/musique/metrics/answer.py"""
    import re, string
    if not pred or not gold_spans:
        return False

    def normalize_answer(s):
        # Lowercase, remove punctuation, remove articles, normalize whitespace.
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
        def remove_punc(text):
            return "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(remove_articles(remove_punc(s.lower())).split())

    pred_n = normalize_answer(pred)
    for span in gold_spans:
        if span and normalize_answer(span) == pred_n:
            return True
    return False


def loose_match(pred: str, gold_spans: list) -> bool:
    """Substring match — kept for secondary reporting."""
    if not pred or not gold_spans:
        return False
    pred_n = pred.strip().lower()
    for span in gold_spans:
        if not span:
            continue
        s = span.strip().lower()
        if s == pred_n or s in pred_n:
            return True
    return False


def run_pilot(n: int, target_route: str = "C", all_routes: bool = False):
    """Run the pilot."""
    print("=" * 70)
    print(f"Pilot baseline: n={n}, target_route={target_route if not all_routes else 'ALL'}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading MuSiQue gold answers...")
    gold = load_musique_gold()
    print(f"  Loaded {len(gold)} MuSiQue queries with gold answers")

    print("\n[2/4] Loading classifier predictions...")
    preds = load_classifier_predictions()
    # Filter to MuSiQue only
    musique_preds = {qid: route for qid, route in preds.items() if qid in gold}
    print(f"  Classifier predictions for {len(musique_preds)} MuSiQue qids")
    if not all_routes:
        filtered = {qid: r for qid, r in musique_preds.items() if r == target_route}
        print(f"  Filtered to Route {target_route}: {len(filtered)} qids")
    else:
        filtered = musique_preds
        print(f"  Using all routes")

    # Sample
    qids_to_test = list(filtered.keys()) if n is None else list(filtered.keys())[:n]
    print(f"\n[3/4] Will run {len(qids_to_test)} queries")

    # Set up the wrapper
    print("\n[4/4] Initializing DeployedAdaptiveRAG wrapper...")
    system = DeployedAdaptiveRAG(mock=False)

    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"pilot_baseline_{timestamp}.csv"
    jsonl_path = OUTPUT_DIR / f"pilot_baseline_{timestamp}.jsonl"
    print(f"\nWriting CSV to:   {csv_path}")
    print(f"Writing JSONL to: {jsonl_path}")

    # Run queries
    print("\n" + "=" * 70)
    print("Running queries...")
    print("=" * 70)

    results = []
    start_time = time.time()

    with open(csv_path, "w", newline="") as csvfile, open(jsonl_path, "w") as jsonlfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "idx", "qid", "predicted_route", "actual_route",
            "question", "gold_answer", "system_answer",
            "latency_ms", "num_sources", "exact_match", "loose_match",
        ])

        for i, qid in enumerate(qids_to_test, 1):
            g = gold[qid]
            predicted_route = filtered[qid]
            question = g["question"]
            gold_answer = g["primary_answer"]

            print(f"\n[Q{i}/{len(qids_to_test)}] qid={qid}")
            print(f"  predicted_route: {predicted_route}")
            print(f"  question: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"  gold: {gold_answer!r}")

            try:
                result = system.query(question, expose_route=True)
                actual_route = result.get("_route", "?")
                answer = result.get("answer", "[ERROR]")
                latency_ms = result.get("latency_ms", -1)
                num_sources = result.get("num_sources", -1)
            except Exception as e:
                print(f"  ERROR: {e}")
                actual_route = "?"
                answer = f"[EXCEPTION: {e}]"
                latency_ms = -1
                num_sources = -1

            em_strict = exact_match(answer, g["answer_spans"])
            em_loose = loose_match(answer, g["answer_spans"])

            print(f"  actual_route: {actual_route}")
            print(f"  answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"  latency_ms: {latency_ms:.1f}")
            print(f"  num_sources: {num_sources}")
            print(f"  exact_match (strict): {em_strict}")
            print(f"  loose_match:          {em_loose}")

            row = [
                i, qid, predicted_route, actual_route,
                question, gold_answer, answer,
                f"{latency_ms:.1f}", num_sources, em_strict, em_loose,
            ]
            writer.writerow(row)
            csvfile.flush()

            result_record = {
                "idx": i,
                "qid": qid,
                "predicted_route": predicted_route,
                "actual_route": actual_route,
                "question": question,
                "gold_answer_spans": g["answer_spans"],
                "system_answer": answer,
                "latency_ms": latency_ms,
                "num_sources": num_sources,
                "exact_match": em_strict,
                "loose_match": em_loose,
            }
            jsonlfile.write(json.dumps(result_record) + "\n")
            jsonlfile.flush()

            results.append(result_record)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/len(results):.1f}s per query avg)")
    print(f"Total queries: {len(results)}")

    # Classifier vs actual route agreement
    agree = sum(1 for r in results if r["predicted_route"] == r["actual_route"])
    print(f"\nClassifier route stability: {agree}/{len(results)} queries got the same route in pilot and wrapper")

    # Per-route stats
    routes = {}
    for r in results:
        route = r["actual_route"]
        if route not in routes:
            routes[route] = []
        routes[route].append(r)

    print(f"\nPer-route breakdown:")
    for route in sorted(routes.keys()):
        rs = routes[route]
        n_route = len(rs)
        correct_strict = sum(1 for r in rs if r["exact_match"])
        correct_loose = sum(1 for r in rs if r.get("loose_match", False))
        latencies = [r["latency_ms"] for r in rs if r["latency_ms"] > 0]
        sources = [r["num_sources"] for r in rs if r["num_sources"] >= 0]
        print(f"  Route {route}: n={n_route}")
        print(f"    accuracy (strict EM):  {correct_strict}/{n_route} = {100*correct_strict/n_route:.1f}%")
        print(f"    accuracy (loose):      {correct_loose}/{n_route} = {100*correct_loose/n_route:.1f}%")
        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            print(f"    latency_ms: mean={mean_lat:.0f}, min={min_lat:.0f}, max={max_lat:.0f}")
        if sources:
            mean_src = sum(sources) / len(sources)
            print(f"    num_sources: mean={mean_src:.1f}")

    strict_total = sum(1 for r in results if r["exact_match"])
    loose_total = sum(1 for r in results if r.get("loose_match", False))
    print(f"\nOverall strict EM: {strict_total}/{len(results)} = {100*strict_total/len(results):.1f}%")
    print(f"Overall loose:      {loose_total}/{len(results)} = {100*loose_total/len(results):.1f}%")

    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Number of queries to run (default: all)")
    parser.add_argument("--target-route", type=str, default="C", choices=["A", "B", "C"],
                        help="Route to filter to (default: C)")
    parser.add_argument("--all-routes", action="store_true",
                        help="Use all routes (ignores --target-route)")
    args = parser.parse_args()

    run_pilot(args.n, args.target_route, args.all_routes)
