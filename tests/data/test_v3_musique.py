import json
import tempfile
import unittest
from pathlib import Path

from prematurestop.system.benchmark_search import run_benchmark_search
from prematurestop.data.corpus import build_musique_rows, write_jsonl
from prematurestop.data.evidence import (
    EVIDENCE_AMBIGUOUS,
    EVIDENCE_COMPLETE,
    EVIDENCE_NONE,
    EVIDENCE_RELEVANT_INCOMPLETE,
    coverage_from_retrieval_metadata,
)
from prematurestop.data.musique_scope_audit import canonical_rows_sha256


class V3MuSiQueGeneralizationTests(unittest.TestCase):
    def test_valid_2_3_4_hop_examples_are_not_filtered(self):
        for hop_count in (2, 3, 4):
            corpus, evidence = build_musique_rows(_musique_fixture(hop_count))
            self.assertEqual(evidence.hop_count, hop_count)
            self.assertEqual(len(evidence.dependencies), hop_count)
            self.assertGreaterEqual(len(corpus), hop_count + 1)

    def test_corpus_does_not_expose_support_labels_or_gold_answer(self):
        corpus, _evidence = build_musique_rows(_musique_fixture(3))
        for row in corpus:
            payload = row.to_json()
            self.assertNotIn("is_supporting", payload)
            self.assertNotIn("answer", payload)
            self.assertNotIn("gold_answer", payload)

    def test_dependency_fields_use_official_decomposition_order(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(4, support_order=(3, 1, 4, 0)))
        self.assertEqual([dep.hop_index for dep in evidence.dependencies], [0, 1, 2, 3])
        self.assertEqual([dep.decomposition_step for dep in evidence.dependencies], [10, 11, 12, 13])
        self.assertEqual([dep.supporting_paragraph_id for dep in evidence.dependencies], ["3", "1", "4", "0"])
        self.assertEqual(evidence.dependencies[2].decomposition_question, "Subquestion 3?")
        self.assertEqual(evidence.dependencies[2].intermediate_answer, "Intermediate 3")

    def test_exact_coverage_fractions_for_four_hops(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(4))
        deps = evidence.dependencies
        one = coverage_from_retrieval_metadata(deps, [{"document_id": deps[0].document_id}])
        three = coverage_from_retrieval_metadata(
            deps,
            [
                {"document_id": deps[0].document_id},
                {"document_id": deps[1].document_id},
                {"document_id": deps[2].document_id},
            ],
        )
        four = coverage_from_retrieval_metadata(deps, [{"document_id": dep.document_id} for dep in deps])
        self.assertEqual(one.status, EVIDENCE_RELEVANT_INCOMPLETE)
        self.assertEqual(one.coverage_fraction, 0.25)
        self.assertEqual(three.coverage_fraction, 0.75)
        self.assertEqual(four.status, EVIDENCE_COMPLETE)
        self.assertEqual(four.coverage_fraction, 1.0)

    def test_repeated_retrieval_does_not_increase_coverage(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(4))
        dep = evidence.dependencies[0]
        coverage = coverage_from_retrieval_metadata(
            evidence.dependencies,
            [{"document_id": dep.document_id}, {"document_id": dep.document_id}, {"document_id": dep.document_id}],
        )
        self.assertEqual(coverage.coverage_fraction, 0.25)

    def test_two_dependencies_can_map_to_same_paragraph(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(3, support_order=(0, 0, 2)))
        shared_paragraph = evidence.dependencies[0].supporting_paragraph_id
        coverage = coverage_from_retrieval_metadata(
            evidence.dependencies,
            [{"paragraph_id": shared_paragraph}],
        )
        self.assertEqual(coverage.required_paragraph_ids, ("0", "2"))
        self.assertEqual(coverage.coverage_fraction, 2 / 3)
        self.assertEqual(
            set(coverage.covered_dependency_ids),
            {evidence.dependencies[0].dependency_id, evidence.dependencies[1].dependency_id},
        )

    def test_one_paragraph_covers_one_of_several_dependencies(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(4))
        coverage = coverage_from_retrieval_metadata(
            evidence.dependencies,
            [{"paragraph_id": evidence.dependencies[2].supporting_paragraph_id}],
        )
        self.assertEqual(coverage.coverage_fraction, 0.25)
        self.assertEqual(coverage.covered_dependency_ids, (evidence.dependencies[2].dependency_id,))

    def test_all_paragraphs_retrieved_but_ambiguous_dependency_blocks_completion(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(3, missing_support_idx=1))
        coverage = coverage_from_retrieval_metadata(
            evidence.dependencies,
            [{"document_id": dep.document_id, "paragraph_id": dep.supporting_paragraph_id} for dep in evidence.dependencies],
        )
        self.assertEqual(coverage.status, EVIDENCE_AMBIGUOUS)
        self.assertEqual(coverage.ambiguous_dependency_ids, (evidence.dependencies[1].dependency_id,))

    def test_distractor_only_retrieval_is_none(self):
        corpus, evidence = build_musique_rows(_musique_fixture(3))
        distractor = next(row for row in corpus if row.title == "Distractor")
        coverage = coverage_from_retrieval_metadata(evidence.dependencies, [{"document_id": distractor.document_id}])
        self.assertEqual(coverage.status, EVIDENCE_NONE)
        self.assertEqual(coverage.coverage_fraction, 0.0)

    def test_all_supports_in_different_order_are_complete(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(4))
        metadata = [
            {"document_id": dep.document_id, "paragraph_id": dep.supporting_paragraph_id}
            for dep in reversed(evidence.dependencies)
        ]
        coverage = coverage_from_retrieval_metadata(evidence.dependencies, metadata)
        self.assertEqual(coverage.status, EVIDENCE_COMPLETE)
        self.assertEqual(coverage.coverage_fraction, 1.0)
        self.assertEqual(
            coverage.retrieved_paragraph_ids,
            tuple(dep.supporting_paragraph_id for dep in reversed(evidence.dependencies)),
        )

    def test_missing_required_dependency_is_ambiguous(self):
        _corpus, evidence = build_musique_rows(_musique_fixture(3, missing_support_idx=1))
        self.assertTrue(evidence.dependencies[1].ambiguous)
        coverage = coverage_from_retrieval_metadata(
            evidence.dependencies,
            [{"document_id": evidence.dependencies[0].document_id}],
        )
        self.assertEqual(coverage.status, EVIDENCE_AMBIGUOUS)

    def test_stable_dependency_ids(self):
        _corpus_a, evidence_a = build_musique_rows(_musique_fixture(4))
        _corpus_b, evidence_b = build_musique_rows(_musique_fixture(4))
        self.assertEqual(
            [dep.dependency_id for dep in evidence_a.dependencies],
            [dep.dependency_id for dep in evidence_b.dependencies],
        )

    def test_oracle_complete_supplies_all_supports_for_each_hop_length(self):
        for hop_count in (2, 3, 4):
            with self.subTest(hop_count=hop_count):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    corpus, evidence = build_musique_rows(_musique_fixture(hop_count))
                    corpus_path = tmp_path / "corpus.jsonl"
                    evidence_path = tmp_path / "evidence.jsonl"
                    trace_path = tmp_path / "trace.jsonl"
                    write_jsonl(corpus_path, [row.to_json() for row in corpus])
                    write_jsonl(evidence_path, [evidence.to_json()])
                    output = run_benchmark_search(
                        ["anything"],
                        config={
                            "configurable": {
                                "benchmark_corpus_path": str(corpus_path),
                                "benchmark_evidence_path": str(evidence_path),
                                "benchmark_dataset": "musique",
                                "benchmark_example_id": f"dev_musique_{hop_count}",
                                "benchmark_cache_dir": str(tmp_path / "cache"),
                                "benchmark_trace_path": str(trace_path),
                                "benchmark_top_k": 1,
                                "v3_run_id": f"dev_oracle_{hop_count}",
                                "v3_condition": "OracleComplete_Neutral",
                                "v3_tool_context": {
                                    "supervisor_delegation_index": 0,
                                    "researcher_subgraph_id": f"dev_researcher_{hop_count}",
                                    "tool_call_id": f"dev_tool_call_{hop_count}",
                                    "tool_name": "tavily_search",
                                },
                            }
                        },
                    )
                    event = json.loads(trace_path.read_text().splitlines()[0])
                    self.assertIn("Oracle Support", output)
                    self.assertTrue(event["oracle_complete_available"])
                    self.assertEqual(event["oracle_required_dependency_count"], hop_count)
                    self.assertEqual(event["oracle_supplied_dependency_count"], hop_count)
                    self.assertEqual(event["oracle_required_paragraph_count"], hop_count)
                    self.assertEqual(event["oracle_coverage_before_cue"]["coverage_fraction"], 1.0)

    def test_oracle_complete_counts_shared_paragraph_dependencies_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus, evidence = build_musique_rows(_musique_fixture(3, support_order=(0, 0, 2)))
            corpus_path = tmp_path / "corpus.jsonl"
            evidence_path = tmp_path / "evidence.jsonl"
            trace_path = tmp_path / "trace.jsonl"
            write_jsonl(corpus_path, [row.to_json() for row in corpus])
            write_jsonl(evidence_path, [evidence.to_json()])
            run_benchmark_search(
                ["anything"],
                config={
                    "configurable": {
                        "benchmark_corpus_path": str(corpus_path),
                        "benchmark_evidence_path": str(evidence_path),
                        "benchmark_dataset": "musique",
                        "benchmark_example_id": "dev_musique_3",
                        "benchmark_cache_dir": str(tmp_path / "cache"),
                        "benchmark_trace_path": str(trace_path),
                        "benchmark_top_k": 1,
                        "v3_run_id": "dev_oracle_shared_paragraph",
                        "v3_condition": "OracleComplete_Neutral",
                        "v3_tool_context": {
                            "supervisor_delegation_index": 0,
                            "researcher_subgraph_id": "dev_researcher_shared",
                            "tool_call_id": "dev_tool_call_shared",
                            "tool_name": "tavily_search",
                        },
                    }
                },
            )
            event = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(event["oracle_required_dependency_count"], 3)
            self.assertEqual(event["oracle_supplied_dependency_count"], 3)
            self.assertEqual(event["oracle_required_paragraph_count"], 2)
            self.assertEqual(event["oracle_supplied_paragraph_count"], 2)

    def test_candidate_hashing_uses_full_sha256_and_rejects_duplicate_ids(self):
        rows = [{"id": "b", "hop_count": 3}, {"id": "a", "hop_count": 2}]
        digest = canonical_rows_sha256(rows, dataset="bdsaglam/musique", dataset_version="0.0.0", split="validation")
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            canonical_rows_sha256(
                [{"id": "dup", "hop_count": 2}, {"id": "dup", "hop_count": 3}],
                dataset="bdsaglam/musique",
                dataset_version="0.0.0",
                split="validation",
            )


def _musique_fixture(hop_count: int, *, support_order=None, missing_support_idx: int | None = None):
    if support_order is None:
        support_order = tuple(range(hop_count))
    paragraphs = []
    max_index = max(max(support_order, default=0), hop_count) + 1
    for idx in range(max_index):
        title = "Distractor" if idx == hop_count else f"Oracle Support {idx}"
        text = (
            "Distractor paragraph with no required evidence."
            if idx == hop_count
            else f"Support paragraph {idx} contains evidence for hop {idx}."
        )
        if idx == support_order[-1]:
            text += " Final Answer Token."
        paragraphs.append(
            {
                "idx": idx,
                "title": title,
                "paragraph_text": text,
                "is_supporting": idx in set(support_order),
            }
        )

    decomposition = []
    for hop_index in range(hop_count):
        support_idx = support_order[hop_index]
        if missing_support_idx == hop_index:
            support_idx = len(paragraphs) + 10
        decomposition.append(
            {
                "question": f"Subquestion {hop_index + 1}?",
                "id": hop_index + 10,
                "answer": f"Intermediate {hop_index + 1}",
                "paragraph_support_idx": support_idx,
            }
        )

    return {
        "id": f"dev_musique_{hop_count}",
        "question": f"Composed {hop_count}-hop question?",
        "answer": "Final Answer Token",
        "question_decomposition": decomposition,
        "paragraphs": paragraphs,
    }


if __name__ == "__main__":
    unittest.main()
