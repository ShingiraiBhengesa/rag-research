import json
import tempfile
import unittest
from pathlib import Path

from attack.cue_definitions import PERTURBATION_FAMILIES
from attack.runtime.benchmark_search import BM25Index, BenchmarkDocument, run_benchmark_search
from attack.runtime.cues import suffix_for_condition
from attack.runtime.corpus import synthetic_dev_corpus, write_jsonl
from attack.runtime.evidence import (
    EVIDENCE_AMBIGUOUS,
    EVIDENCE_RELEVANT_INCOMPLETE,
    EvidenceDependency,
    coverage_from_retrieval_metadata,
    evaluate_coverage,
)
from attack.runtime.trace_schema import TraceValidationError, validate_paired_traces


class V3InfrastructureTests(unittest.TestCase):
    def test_bm25_is_deterministic_with_doc_id_tiebreak(self):
        docs = [
            BenchmarkDocument("devset", "dev_example_002", "b", "b", "Same", "identical text"),
            BenchmarkDocument("devset", "dev_example_002", "a", "a", "Same", "identical text"),
        ]
        index = BM25Index(docs)
        first = [result.document_id for result in index.search("identical", 2)]
        second = [result.document_id for result in index.search("identical", 2)]
        self.assertEqual(first, ["a", "b"])
        self.assertEqual(first, second)

    def test_corpus_rows_reject_gold_label_leakage(self):
        with self.assertRaises(ValueError):
            BenchmarkDocument.from_json(
                {
                    "dataset": "devset",
                    "example_id": "dev_example_003",
                    "corpus_id": "x",
                    "document_id": "x",
                    "title": "Leak",
                    "text": "Visible text",
                    "answer": "forbidden",
                }
            )

    def test_coverage_ignores_suffix_and_answer_text(self):
        deps = (
            EvidenceDependency("dep_alpha", "doc_alpha"),
            EvidenceDependency("dep_beta", "doc_beta"),
        )
        metadata = [{"document_id": "doc_alpha", "suffix": "Answer: doc_beta"}]
        coverage = coverage_from_retrieval_metadata(deps, metadata)
        self.assertEqual(coverage.status, EVIDENCE_RELEVANT_INCOMPLETE)
        self.assertEqual(coverage.remaining_dependency_ids, ("dep_beta",))

    def test_ambiguous_dependency_stays_ambiguous(self):
        deps = (EvidenceDependency("dep_unknown", "doc_unknown", ambiguous=True),)
        coverage = evaluate_coverage(deps, ["doc_unknown"])
        self.assertEqual(coverage.status, EVIDENCE_AMBIGUOUS)

    def test_benchmark_search_injects_only_first_target_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus, evidence = synthetic_dev_corpus()
            corpus_path = tmp_path / "corpus.jsonl"
            evidence_path = tmp_path / "evidence.jsonl"
            trace_path = tmp_path / "trace.jsonl"
            write_jsonl(corpus_path, [row.to_json() for row in corpus])
            write_jsonl(evidence_path, [row.to_json() for row in evidence])

            cfg = {
                "configurable": {
                    "benchmark_corpus_path": str(corpus_path),
                    "benchmark_evidence_path": str(evidence_path),
                    "benchmark_dataset": "devset",
                    "benchmark_example_id": "dev_example_001",
                    "benchmark_cache_dir": str(tmp_path / "cache"),
                    "benchmark_trace_path": str(trace_path),
                    "benchmark_top_k": 2,
                    "v3_run_id": "dev_run_inject_once",
                    "v3_condition": "Natural_epistemic_confirmation",
                    "v3_tool_context": {
                        "supervisor_delegation_index": 0,
                        "researcher_subgraph_id": "dev_researcher",
                        "tool_call_id": "dev_tool_call",
                        "tool_name": "tavily_search",
                    },
                }
            }

            first = run_benchmark_search(["score festival"], config=cfg)
            second = run_benchmark_search(["score festival"], config=cfg)
            self.assertIn(PERTURBATION_FAMILIES["epistemic_confirmation"], first)
            self.assertNotIn(PERTURBATION_FAMILIES["epistemic_confirmation"], second)

            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(sum(1 for event in events if event["attacked_observation"]), 1)

    def test_natural_clean_never_injects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus, _evidence = synthetic_dev_corpus()
            corpus_path = tmp_path / "corpus.jsonl"
            write_jsonl(corpus_path, [row.to_json() for row in corpus])
            cfg = {
                "configurable": {
                    "benchmark_corpus_path": str(corpus_path),
                    "benchmark_dataset": "devset",
                    "benchmark_example_id": "dev_example_001",
                    "benchmark_cache_dir": str(tmp_path / "cache"),
                    "v3_run_id": "dev_run_clean",
                    "v3_condition": "NaturalClean",
                }
            }
            output = run_benchmark_search(["score festival"], config=cfg)
            self.assertNotIn("This note is included", output)

    def test_paired_trace_validation_accepts_shared_base(self):
        base_event = {
            "event_type": "benchmark_search",
            "shared_prefix_hash": "same_prefix",
            "base_tool_message_hash": "same_base",
            "attacked_observation": False,
        }
        neutral_event = {
            "event_type": "benchmark_search",
            "shared_prefix_hash": "same_prefix",
            "base_tool_message_hash": "same_base",
            "attacked_observation": True,
            "suffix_hash": suffix_for_condition("NaturalNeutral").suffix_hash,
        }
        traces = [
            {
                "dataset": "devset",
                "example_id": "dev_example_001",
                "condition": "NaturalClean",
                "events": [{"step": 0, **base_event}],
                "final_report": "Answer: x",
                "extracted_answer": "x",
                "final_state": {"raw_notes": ["source observation"], "notes": []},
            },
            {
                "dataset": "devset",
                "example_id": "dev_example_001",
                "condition": "NaturalNeutral",
                "events": [{"step": 0, **neutral_event}],
                "final_report": "Answer: x",
                "extracted_answer": "x",
                "final_state": {"raw_notes": ["source observation"], "notes": []},
            },
        ]
        validate_paired_traces(traces)

    def test_paired_trace_validation_requires_replay_anchor_match(self):
        anchor = {
            "first_search_queries": ["score festival"],
            "source_base_tool_message_hash": "same_base",
            "source_shared_prefix_hash": "same_prefix",
        }
        traces = [
            {
                "dataset": "devset",
                "example_id": "dev_example_001",
                "condition": "NaturalClean",
                "paired_replay_anchor": anchor,
                "events": [
                    {
                        "step": 0,
                        "event_type": "benchmark_search",
                        "queries": ["score festival"],
                        "shared_prefix_hash": "same_prefix",
                        "base_tool_message_hash": "same_base",
                        "attacked_observation": False,
                    }
                ],
                "final_report": "Answer: x",
                "extracted_answer": "x",
                "final_state": {"raw_notes": ["source observation"], "notes": []},
            },
            {
                "dataset": "devset",
                "example_id": "dev_example_001",
                "condition": "NaturalNeutral",
                "paired_replay_anchor": anchor,
                "events": [
                    {
                        "step": 0,
                        "event_type": "benchmark_search",
                        "queries": ["different query"],
                        "shared_prefix_hash": "same_prefix",
                        "base_tool_message_hash": "same_base",
                        "attacked_observation": True,
                        "suffix_hash": suffix_for_condition("NaturalNeutral").suffix_hash,
                    }
                ],
                "final_report": "Answer: x",
                "extracted_answer": "x",
                "final_state": {"raw_notes": ["source observation"], "notes": []},
            },
        ]
        with self.assertRaisesRegex(TraceValidationError, "did not reuse anchor first-search queries"):
            validate_paired_traces(traces)

    def test_shared_prefix_hash_ignores_runtime_tool_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus, evidence = synthetic_dev_corpus()
            corpus_path = tmp_path / "corpus.jsonl"
            evidence_path = tmp_path / "evidence.jsonl"
            clean_trace_path = tmp_path / "clean.jsonl"
            attacked_trace_path = tmp_path / "attacked.jsonl"
            write_jsonl(corpus_path, [row.to_json() for row in corpus])
            write_jsonl(evidence_path, [row.to_json() for row in evidence])

            base_cfg = {
                "benchmark_corpus_path": str(corpus_path),
                "benchmark_evidence_path": str(evidence_path),
                "benchmark_dataset": "devset",
                "benchmark_example_id": "dev_example_001",
                "benchmark_cache_dir": str(tmp_path / "cache"),
                "benchmark_top_k": 2,
                "v3_target_supervisor_delegation_index": 0,
                "v3_target_search_call_index": 0,
            }
            clean_cfg = {
                "configurable": {
                    **base_cfg,
                    "benchmark_trace_path": str(clean_trace_path),
                    "v3_run_id": "dev_runtime_id_clean",
                    "v3_condition": "NaturalClean",
                    "v3_tool_context": {
                        "supervisor_delegation_index": 0,
                        "supervisor_tool_call_id": "chatcmpl-tool-clean",
                        "researcher_subgraph_id": "supervisor_iter_2:delegation_0:tool_call_chatcmpl-tool-clean",
                        "tool_call_id": "forced_benchmark_local_search_1",
                        "tool_name": "tavily_search",
                    },
                }
            }
            attacked_cfg = {
                "configurable": {
                    **base_cfg,
                    "benchmark_trace_path": str(attacked_trace_path),
                    "v3_run_id": "dev_runtime_id_attacked",
                    "v3_condition": "NaturalNeutral",
                    "v3_tool_context": {
                        "supervisor_delegation_index": 0,
                        "supervisor_tool_call_id": "chatcmpl-tool-attacked",
                        "researcher_subgraph_id": "supervisor_iter_2:delegation_0:tool_call_chatcmpl-tool-attacked",
                        "tool_call_id": "forced_benchmark_local_search_1",
                        "tool_name": "tavily_search",
                    },
                }
            }

            query = "score festival"
            run_benchmark_search([query], config=clean_cfg)
            run_benchmark_search([query], config=attacked_cfg)

            clean_event = json.loads(clean_trace_path.read_text().splitlines()[0])
            attacked_event = json.loads(attacked_trace_path.read_text().splitlines()[0])
            self.assertEqual(clean_event["base_tool_message_hash"], attacked_event["base_tool_message_hash"])
            self.assertEqual(clean_event["shared_prefix_hash"], attacked_event["shared_prefix_hash"])
            self.assertNotEqual(clean_event["researcher_subgraph_id"], attacked_event["researcher_subgraph_id"])
            self.assertNotEqual(clean_event["supervisor_tool_call_id"], attacked_event["supervisor_tool_call_id"])
            self.assertNotEqual(clean_event["tool_message_hash"], attacked_event["tool_message_hash"])
            validate_paired_traces(
                [
                    {
                        "dataset": "devset",
                        "example_id": "dev_example_001",
                        "condition": "NaturalClean",
                        "events": [{"step": 0, **clean_event}],
                        "final_report": "Answer: x",
                        "extracted_answer": "x",
                        "final_state": {"raw_notes": ["source observation"], "notes": []},
                    },
                    {
                        "dataset": "devset",
                        "example_id": "dev_example_001",
                        "condition": "NaturalNeutral",
                        "events": [{"step": 0, **attacked_event}],
                        "final_report": "Answer: x",
                        "extracted_answer": "x",
                        "final_state": {"raw_notes": ["source observation"], "notes": []},
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
