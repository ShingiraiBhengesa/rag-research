import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prematurestop.system.benchmark_search import BenchmarkDocument, run_benchmark_search
from prematurestop.data.corpus import build_musique_rows, write_jsonl
from prematurestop.data.evidence import EVIDENCE_COMPLETE, coverage_from_retrieval_metadata
from prematurestop.data.musique_scope_audit import validate_musique_example


class V3MuSiQueRealSchemaTests(unittest.TestCase):
    def test_real_schema_examples_for_2_3_4_hops(self):
        for hop_count in (2, 3, 4):
            with self.subTest(hop_count=hop_count):
                ex = _real_musique_example(hop_count)
                corpus, evidence = build_musique_rows(ex)
                corpus_again, evidence_again = build_musique_rows(ex)

                self.assertEqual(len(corpus), len(ex["paragraphs"]))
                self.assertEqual(evidence.hop_count, hop_count)
                self.assertEqual(len(evidence.dependencies), hop_count)
                self.assertEqual(
                    [dep.dependency_id for dep in evidence.dependencies],
                    [dep.dependency_id for dep in evidence_again.dependencies],
                )
                self.assertEqual(
                    [row.document_id for row in corpus],
                    [row.document_id for row in corpus_again],
                )

                for dep, step in zip(evidence.dependencies, ex["question_decomposition"]):
                    support_idx = int(step["paragraph_support_idx"])
                    supporting_paragraph = ex["paragraphs"][support_idx]
                    self.assertEqual(dep.supporting_paragraph_id, str(supporting_paragraph["idx"]))
                    self.assertEqual(dep.hop_index, evidence.dependencies.index(dep))

                for row in corpus:
                    payload = row.to_json()
                    self.assertNotIn("answer", payload)
                    self.assertNotIn("is_supporting", payload)
                    BenchmarkDocument.from_json(payload)

                first_dep = evidence.dependencies[0]
                coverage = coverage_from_retrieval_metadata(
                    evidence.dependencies,
                    [{"paragraph_id": first_dep.supporting_paragraph_id}],
                )
                self.assertIn(first_dep.dependency_id, coverage.covered_dependency_ids)
                self.assertEqual(coverage.retrieved_paragraph_ids, (first_dep.supporting_paragraph_id,))

                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    corpus_path = tmp_path / "corpus.jsonl"
                    evidence_path = tmp_path / "evidence.jsonl"
                    trace_path = tmp_path / "trace.jsonl"
                    write_jsonl(corpus_path, [row.to_json() for row in corpus])
                    write_jsonl(evidence_path, [evidence.to_json()])
                    run_benchmark_search(
                        [evidence.question],
                        config={
                            "configurable": {
                                "benchmark_corpus_path": str(corpus_path),
                                "benchmark_evidence_path": str(evidence_path),
                                "benchmark_dataset": "musique",
                                "benchmark_example_id": ex["id"],
                                "benchmark_cache_dir": str(tmp_path / "cache"),
                                "benchmark_trace_path": str(trace_path),
                                "benchmark_top_k": 1,
                                "v3_run_id": f"dev_real_schema_oracle_{hop_count}",
                                "v3_condition": "OracleComplete_Neutral",
                                "v3_tool_context": {
                                    "supervisor_delegation_index": 0,
                                    "researcher_subgraph_id": f"dev_real_schema_researcher_{hop_count}",
                                    "tool_call_id": f"dev_real_schema_tool_{hop_count}",
                                    "tool_name": "tavily_search",
                                },
                            }
                        },
                    )
                    event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
                    self.assertTrue(event["oracle_complete_available"])
                    self.assertEqual(event["coverage_status_after_observation"], EVIDENCE_COMPLETE)
                    self.assertEqual(event["oracle_required_dependency_count"], hop_count)
                    self.assertEqual(event["oracle_supplied_dependency_count"], hop_count)


def _real_musique_example(hop_count: int) -> dict:
    try:
        from datasets import DownloadConfig, load_dataset
    except Exception as exc:
        raise unittest.SkipTest(f"datasets package unavailable: {exc}") from exc

    ds = load_dataset(
        "bdsaglam/musique",
        "answerable",
        split="validation",
        download_config=DownloadConfig(local_files_only=True),
    )
    for ex in ds:
        if len(ex.get("question_decomposition") or []) != hop_count:
            continue
        validation = validate_musique_example(ex)
        if not validation["valid"]:
            continue
        row = dict(ex)
        source_id_hash = hashlib.sha256(str(ex["id"]).encode("utf-8")).hexdigest()[:12]
        row["id"] = f"dev_real_musique_{hop_count}_{source_id_hash}"
        return row
    raise unittest.SkipTest(f"no cached valid {hop_count}-hop MuSiQue validation example found")


if __name__ == "__main__":
    unittest.main()
