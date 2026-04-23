"""
DeployedAdaptiveRAG — wrapper around the Adaptive-RAG pipeline that exposes a
clean black-box query interface.

Usage:
    system = DeployedAdaptiveRAG()
    result = system.query("Who founded the Danish Football Union?")
    # result = {
    #     "answer": "...",
    #     "latency_ms": 8240,
    #     "num_sources": 5,
    #     "_route": "C",   # internal, not used by black-box attack
    # }

The attacker only sees `answer`, `latency_ms`, and `num_sources`.
`_route` is exposed for analysis but would not be available to a real attacker.

Implementation:
    - Classifier runs in-process (our existing classifier_wrapper.py).
    - Based on the classifier's A/B/C decision, dispatches to one of three
      inference systems via the existing Adaptive-RAG predict.py:
          A -> nor_qa    (no retrieval)
          B -> oner_qa   (single-step retrieval)
          C -> ircot_qa  (multi-step retrieval)
    - Temp jsonl with one question fed to predict.py, output parsed.

Two modes:
    - mock=True: no cluster or servers needed. Fake responses for testing.
    - mock=False: shells out to the real Adaptive-RAG pipeline.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Path to the Adaptive-RAG repo — adjust if the layout changes
ADAPTIVE_RAG_REPO = Path("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG")

# Path to the trained classifier checkpoint (the attack target)
CLASSIFIER_PATH = (
    ADAPTIVE_RAG_REPO
    / "classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/epoch/30"
)

# The valid qids the base configs filter on. We reuse one so our query isn't
# silently dropped by the filter. This qid must exist in the valid_qids list
# inside the jsonnet config.
REUSABLE_MUSIQUE_QID = "2hop__323282_79175"

# Route → system name mapping. These correspond to the base_configs/*.jsonnet
# files we point predict.py at.
ROUTE_TO_SYSTEM = {
    "A": "nor_qa_flan_t5_xl_musique",       # no retrieval
    "B": "oner_qa_flan_t5_xl_musique_paper",  # single-step retrieval (paper config)
    "C": "ircot_qa_flan_t5_xl_musique_paper",  # multi-step retrieval (IRCoT, paper config)
}


@dataclass
class QueryResult:
    answer: str
    latency_ms: float
    num_sources: int
    route: str  # internal — _-prefixed when serialized to mimic black-box

    def to_public_dict(self) -> dict:
        """Return only the fields a black-box attacker would see."""
        return {
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "num_sources": self.num_sources,
        }

    def to_full_dict(self) -> dict:
        """Full result including internal route label (for analysis only)."""
        return {
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "num_sources": self.num_sources,
            "_route": self.route,
        }


class DeployedAdaptiveRAG:
    """Black-box interface to the full Adaptive-RAG pipeline.

    Parameters
    ----------
    mock : bool
        If True, don't touch the cluster or servers. Return fake responses
        based on query length and the classifier's prediction. Useful for
        local testing of attack code before the deployed system is up.
    classifier_device : str
        Which GPU to run the T5-Large classifier on. Ignored in mock mode.
    llm_port : int
        Port of the LLM server (default 8010 per .llm_server_address.jsonnet).
    retriever_port : int
        Port of the retriever server (default 8000).
    """

    def __init__(
        self,
        mock: bool = False,
        classifier_device: str = "cuda:0",
        llm_port: int = 8010,
        retriever_port: int = 8000,
    ):
        self.mock = mock
        self.llm_port = llm_port
        self.retriever_port = retriever_port

        if mock:
            self._classifier = None
            print("[DeployedAdaptiveRAG] mock mode — no models loaded")
        else:
            self._classifier = self._load_classifier(classifier_device)

        # Working directory for temp input/output files (per-instance)
        self._workdir = Path(tempfile.mkdtemp(prefix="deployed_rag_"))
        print(f"[DeployedAdaptiveRAG] workdir: {self._workdir}")

    # --------------------------------------------------------------------
    # Public API — this is what the black-box attack calls
    # --------------------------------------------------------------------
    def query(self, question: str, expose_route: bool = False) -> dict:
        """Run a question through the full pipeline and return observables.

        Parameters
        ----------
        question : str
            The question to ask.
        expose_route : bool
            If True, include the internal route label (for analysis).
            A real attacker would not have access to this.
        """
        t_start = time.perf_counter()

        # 1. Classify
        route, _ = self._classify(question)

        # 2. Run the appropriate system end-to-end
        if self.mock:
            answer, num_sources = self._mock_answer(question, route)
        else:
            answer, num_sources = self._run_system(question, route)

        latency_ms = (time.perf_counter() - t_start) * 1000

        result = QueryResult(
            answer=answer,
            latency_ms=latency_ms,
            num_sources=num_sources,
            route=route,
        )
        return result.to_full_dict() if expose_route else result.to_public_dict()

    # --------------------------------------------------------------------
    # Classifier (real + mock)
    # --------------------------------------------------------------------
    def _load_classifier(self, device: str):
        """Load the T5-Large complexity classifier."""
        import sys
        sys.path.insert(0, str(Path.home() / "adaptive-rag/attack"))
        from classifier_wrapper import AdaptiveRAGClassifierWrapper

        return AdaptiveRAGClassifierWrapper(str(CLASSIFIER_PATH), device=device)

    def _classify(self, question: str) -> tuple[str, np.ndarray]:
        """Return (route_letter, probability_vector)."""
        if self.mock:
            # Mock: simple heuristic — hop count keywords + length.
            q_lower = question.lower()
            if "and" in q_lower or "after" in q_lower or "before" in q_lower:
                probs = np.array([0.1, 0.2, 0.7])  # leans C
            elif len(question.split()) > 12:
                probs = np.array([0.15, 0.55, 0.30])  # leans B
            else:
                probs = np.array([0.6, 0.25, 0.15])  # leans A
        else:
            probs = self._classifier([question])[0]

        route = ["A", "B", "C"][int(np.argmax(probs))]
        return route, probs

    # --------------------------------------------------------------------
    # Real-mode system invocation via predict.py
    # --------------------------------------------------------------------
    def _run_system(self, question: str, route: str) -> tuple[str, int]:
        """Shell out to Adaptive-RAG's predict.py for a single question.

        Returns (answer_string, num_sources_count).

        num_sources is parsed from the chains file — the real count of
        distinct retrieved pids the LLM had in context when generating
        the answer. -1 is returned if parsing fails.
        """
        system = ROUTE_TO_SYSTEM[route]

        run_id = uuid.uuid4().hex[:10]
        set_name = f"attack_{run_id}"

        eval_dir = ADAPTIVE_RAG_REPO / "processed_data" / "musique"
        eval_filename = f"{set_name}_subsampled.jsonl"
        eval_path = eval_dir / eval_filename

        try:
            self._write_single_question_jsonl(question, eval_path)

            cmd = [
                "python", "predict.py",
                system,
                str(eval_path),
                "--set_name", set_name,
                "--llm_port_num", str(self.llm_port),
                "--silent",
                "--skip-evaluation", "skip_evaluation",
                "--force",
            ]

            completed = subprocess.run(
                cmd,
                cwd=str(ADAPTIVE_RAG_REPO),
                capture_output=True,
                text=True,
                timeout=300,
            )

            if completed.returncode != 0:
                print(f"[predict.py FAILED for route {route}]")
                print(f"stderr tail:\n{completed.stderr[-1500:]}")
                return "[ERROR]", -1

            expdir = self._find_experiment_dir(set_name)
            pred_file = self._find_prediction_file(expdir)
            chains_file = self._find_chains_file(expdir)

            if pred_file is None or not pred_file.exists():
                print(f"[WARN] prediction file not found for set_name={set_name}")
                return "[NO_PREDICTION]", -1

            # Retry loop for NFS eventual consistency / predict.py write atomicity
            predictions = None
            for attempt in range(5):
                try:
                    with open(pred_file, "r") as f:
                        content = f.read().strip()
                    if content:
                        predictions = json.loads(content)
                        break
                except (json.JSONDecodeError, OSError):
                    pass
                time.sleep(0.5)
            if predictions is None:
                print(f"[WARN] prediction file empty or invalid after retries: {pred_file}")
                return "[EMPTY_PREDICTION]", -1
            answer = predictions.get(REUSABLE_MUSIQUE_QID, "[QID_MISSING]")

            num_sources = self._count_sources_from_chains(
                chains_file, REUSABLE_MUSIQUE_QID, route
            )

            return answer, num_sources

        finally:
            try:
                eval_path.unlink()
            except FileNotFoundError:
                pass

    # --------------------------------------------------------------------
    # Input/output plumbing for predict.py
    # --------------------------------------------------------------------
    def _write_single_question_jsonl(self, question: str, path: Path):
        """Create a jsonl that predict.py can consume.

        We reuse REUSABLE_MUSIQUE_QID so the config's valid_qids filter lets
        our question through. `contexts: []` is safe — MultiParaRCReader's
        reader loops over contexts but all three add_* branches no-op on an
        empty list. Routes B and C retrieve fresh via Elasticsearch; Route A
        uses only parametric knowledge.
        """
        instance = {
            "question_id": REUSABLE_MUSIQUE_QID,
            "question_text": question,
            "contexts": [],
            "answers_objects": [{
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [""]
            }],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(instance) + "\n")

    def _find_experiment_dir(self, set_name: str) -> Optional[Path]:
        """Locate the single experiment subdir predict.py wrote to."""
        pred_dir = ADAPTIVE_RAG_REPO / "predictions" / set_name
        if not pred_dir.exists():
            return None
        for expdir in pred_dir.iterdir():
            if expdir.is_dir():
                return expdir
        return None

    def _find_prediction_file(self, expdir: Path) -> Optional[Path]:
        """Locate the main prediction JSON inside the experiment dir.
        
        predict.py writes several files: prediction__X.json (the real one),
        plus side files like prediction__X_variable_replacements.json and
        prediction__X_time_taken.txt. We need the bare one — no suffix
        between the dataset tag and .json.
        """
        if expdir is None or not expdir.exists():
            return None
        # Excluded suffixes — these are side files, not the real prediction
        EXCLUDED_SUFFIXES = (
            "_variable_replacements.json",
            "_chains.txt",
            "_time_taken.txt",
            "_full_eval_path.txt",
        )
        for f in expdir.iterdir():
            name = f.name
            if not name.startswith("prediction__"):
                continue
            if not name.endswith(".json"):
                continue
            if any(name.endswith(suf) for suf in EXCLUDED_SUFFIXES):
                continue
            return f
        return None

    def _find_chains_file(self, expdir: Path) -> Optional[Path]:
        """Locate prediction__*_chains.txt inside the experiment dir.

        The chains file holds the reasoning trace including retrieved pids.
        That's where we count real num_sources.
        """
        if expdir is None or not expdir.exists():
            return None
        for f in expdir.iterdir():
            if f.name.startswith("prediction__") and f.name.endswith("_chains.txt"):
                return f
        return None

    def _count_sources_from_chains(
        self, chains_file: Path, qid: str, route: str
    ) -> int:
        """Parse the chains file to count distinct retrieved paragraphs.

        Chains file format (per query):
            <blank line>
            <qid>
            <question_text>
            A: ["title1", "title2", ...]        # retrieval step
            A: <llm reasoning>                   # generation step
            A: Exit? No.
            ... repeat ...
            A: ["pid___...", "pid___...", ...]   # FINAL line with pids
            Q: <question>
            A: "<answer>"
            Q: [EOQ]
            S: <score>

        We find the query's block, locate the final "pid___" line,
        count unique pids.

        For Route A (nor_qa), no retrieval happens — return 0.
        """
        if route == "A":
            return 0
        if chains_file is None or not chains_file.exists():
            return -1
        try:
            text = chains_file.read_text()
        except Exception:
            return -1

        lines = text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip() == qid:
                start = i
                break
        if start is None:
            return -1

        # Find where this qid's block ends (next blank line or end of file).
        # Route C blocks can span hundreds of lines with many iterations.
        end = len(lines)
        for j in range(start + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped.startswith("S:") or stripped.startswith("Q: [EOQ]"):
                # Found the terminator. Scan a bit past to catch closing list.
                end = min(j + 5, len(lines))
                break

        # Collect all "A: [...]" list lines in the block. The LAST one is
        # the retrieval accumulated at the moment the LLM produced its final
        # answer. Route B has one such line. Route C has several across
        # iterations, and the last one is the full paragraph list.
        list_lines = []
        for j in range(start, end):
            line = lines[j].strip()
            if line.startswith('A: [') and line.endswith(']'):
                list_lines.append(line)

        if not list_lines:
            return -1

        # Take the last retrieval list
        last = list_lines[-1]
        arr_str = last[len("A: "):].strip()
        try:
            items = json.loads(arr_str)
            return len(set(items))
        except json.JSONDecodeError:
            # Fallback: count commas + 1 (rough estimate for malformed JSON)
            return max(1, arr_str.count(',') + 1)

    # --------------------------------------------------------------------
    # Mock mode — for local testing
    # --------------------------------------------------------------------
    def _mock_answer(self, question: str, route: str) -> tuple[str, int]:
        """Fake a plausible response. Simulates route-dependent latency.

        num_sources distributions match what we see empirically in real runs:
          Route A: always 0
          Route B: ~5 (matches bm25_retrieval_count in oner_qa config)
          Route C: 5-40, with a mean around 15, long-tailed (mirrors stepNum
                   distribution — most queries 2-4 iterations, some 10+)
        """
        fake_latency = {"A": 0.3, "B": 1.2, "C": 3.5}[route]
        fake_latency *= random.uniform(0.85, 1.15)
        time.sleep(fake_latency)

        if route == "A":
            num_sources = 0
        elif route == "B":
            num_sources = 5
        else:
            steps = max(2, int(random.lognormvariate(1.1, 0.5)))
            num_sources = steps * 5

        q_hash = abs(hash(question)) % 10000
        answer = f"[MOCK answer for route {route} · q={q_hash:04d}]"
        return answer, num_sources


# ====================================================================
# Smoke test
# ====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="run against real deployed servers (requires them running)")
    parser.add_argument("--test-parser", action="store_true",
                        help="unit-test the chains parser against an existing chains file")
    args = parser.parse_args()

    if args.test_parser:
        print("=" * 70)
        print("Parser test — _count_sources_from_chains against real data")
        print("=" * 70)
        system = DeployedAdaptiveRAG(mock=True)
        chains_file = (
            ADAPTIVE_RAG_REPO
            / "predictions/test/ircot_qa_flan_t5_xl_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1"
            / "prediction__musique_to_musique__test_subsampled_chains.txt"
        )
        test_qids = [
            "2hop__511176_22458",   # Danish Football Union
            "2hop__20059_236755",   # Intellectual property
        ]
        for qid in test_qids:
            count = system._count_sources_from_chains(chains_file, qid, route="C")
            print(f"  {qid}: parsed num_sources = {count}")
        exit(0)

    system = DeployedAdaptiveRAG(mock=not args.real)

    test_questions = [
        "What is the capital of France?",
        "Who directed the 2010 film Inception and what was its worldwide box office?",
        "Who founded the football team that won the championship in the country where the Eiffel Tower is located, and what year was that team founded?",
    ]

    print("=" * 70)
    print(f"Mode: {'REAL' if args.real else 'MOCK'}")
    print("=" * 70)

    for i, q in enumerate(test_questions, 1):
        print(f"\n[Q{i}] {q}")
        result = system.query(q, expose_route=True)
        print(f"  route: {result.get('_route', '?')}")
        print(f"  answer: {result['answer']}")
        print(f"  latency_ms: {result['latency_ms']:.1f}")
        print(f"  num_sources: {result['num_sources']}")
