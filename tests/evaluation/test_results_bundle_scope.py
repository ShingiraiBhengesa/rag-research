"""Regression tests: the canonical Qwen/Llama results bundle (and the
evidence-hop-coverage package that shares its scope) must not silently pick
up a third model if `prematurestop.evaluation.config.INCLUDED_MODELS` changes for a reason
unrelated to this bundle (e.g. a model temporarily re-entering production).

This locks the fix made after a reproducibility audit found
`prematurestop.analysis.fixed_cue_matrix.results_bundle` implicitly inheriting its model scope
from that repo-wide, mutable setting -- see
`src/prematurestop/analysis/fixed_cue_matrix/results_bundle/scope.py`'s own docstring for the full
history.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest import mock

from prematurestop.evaluation import config
from prematurestop.analysis.fixed_cue_matrix.results_bundle import build_bundle, gates
from prematurestop.analysis.fixed_cue_matrix.results_bundle.scope import CANONICAL_BUNDLE_MODELS
from prematurestop.analysis.fixed_cue_matrix.evidence_hop_coverage import build_evidence_hop_coverage as ehc


class _StopEarly(Exception):
    """Raised by a stub to abort a multi-step pipeline right after the call
    site under test, so the (expensive, real-data-dependent) remainder of
    that pipeline never has to run in a unit test."""


THIRD_MODEL_INCLUDED = ("qwen14b", "llama8b", "gemma27b")


class CanonicalBundleModelsConstantTests(unittest.TestCase):
    def test_locked_to_exactly_qwen_and_llama(self):
        self.assertEqual(CANONICAL_BUNDLE_MODELS, ("qwen14b", "llama8b"))


class Gate1IgnoresConfigIncludedModelsTests(unittest.TestCase):
    """Gate 1's expected matrix must come from CANONICAL_BUNDLE_MODELS, not
    config.INCLUDED_MODELS -- even when the latter grows a third model."""

    def _write_run_inventory_fixture(self, data_dir: Path) -> None:
        # Exactly the 96 canonical cells (2 models x 4 datasets x 12
        # conditions), nothing more, nothing from a third model.
        rows = [
            {"model": m, "dataset": d, "condition": c}
            for m, d, c in product(CANONICAL_BUNDLE_MODELS, config.DATASETS, config.CONDITIONS)
        ]
        path = data_dir / "run_inventory.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model", "dataset", "condition"])
            writer.writeheader()
            writer.writerows(rows)

    def test_gate_1_passes_96_of_96_even_with_a_third_model_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self._write_run_inventory_fixture(data_dir)

            with mock.patch.object(config, "INCLUDED_MODELS", THIRD_MODEL_INCLUDED):
                result = gates.gate_1_full_matrix(data_dir)

        self.assertTrue(result.passed, result.detail)
        self.assertIn("96/96", result.detail)

    def test_gate_1_would_fail_if_it_actually_expected_the_third_model(self):
        # Sanity check on the test itself: confirm the fixture really does
        # NOT satisfy a 144-cell (3-model) expectation, so the PASS above is
        # not a vacuous fixture-construction bug.
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self._write_run_inventory_fixture(data_dir)
            with open(data_dir / "run_inventory.csv", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            present = {(r["model"], r["dataset"], r["condition"]) for r in rows}
            three_model_expected = set(product(THIRD_MODEL_INCLUDED, config.DATASETS, config.CONDITIONS))
            self.assertLess(len(present), len(three_model_expected))


class Gate14IgnoresConfigIncludedModelsTests(unittest.TestCase):
    """src/prematurestop/analysis/fixed_cue_matrix/evidence_hop_coverage's gate 14 (manual-sample coverage)
    must check against CANONICAL_BUNDLE_MODELS too, not config.INCLUDED_MODELS."""

    def _fixture_rows(self) -> list[dict]:
        rows = []
        for dataset in config.DATASETS:
            for model in CANONICAL_BUNDLE_MODELS:
                for i in range(3):
                    rows.append({"dataset": dataset, "model": model, "terminal_route": "RESEARCH_COMPLETE" if i else "NO_TOOL_DIRECT_EXIT"})
        return rows

    def test_gate_14_passes_with_only_qwen_and_llama_even_if_config_grows(self):
        with mock.patch.object(config, "INCLUDED_MODELS", THIRD_MODEL_INCLUDED):
            result = ehc.gate_14_manual_sample_coverage(self._fixture_rows())
        self.assertTrue(result.passed, result.detail)


class BuildInventoryCalledWithExplicitScopeTests(unittest.TestCase):
    """Behavioral check: the actual call sites pass models=CANONICAL_BUNDLE_MODELS
    explicitly, so completion_inventory.build_inventory's default
    (config.INCLUDED_MODELS) is never silently relied upon."""

    def test_pcr_bundle_run_stage_data_passes_explicit_scope(self):
        captured = {}

        def fake_build_inventory(production_root, **kwargs):
            captured["models"] = kwargs.get("models")
            raise _StopEarly()

        with mock.patch.object(config, "INCLUDED_MODELS", THIRD_MODEL_INCLUDED):
            with mock.patch.object(build_bundle, "build_inventory", side_effect=fake_build_inventory):
                with self.assertRaises(_StopEarly):
                    build_bundle.run_stage_data(Path("/tmp/unused-bundle-root"), Path("/tmp/unused-production-root"), 1)

        self.assertEqual(captured["models"], CANONICAL_BUNDLE_MODELS)

    def test_evidence_hop_coverage_main_passes_explicit_scope(self):
        captured = {}

        def fake_build_inventory(production_root, **kwargs):
            captured["models"] = kwargs.get("models")
            raise _StopEarly()

        argv = [
            "build_evidence_hop_coverage.py",
            "--output-dir", "/tmp/unused-output-dir",
            "--production-root", "/tmp/unused-production-root",
            "--pcr-bundle-dir", "/tmp/unused-pcr-bundle-dir",
        ]
        with mock.patch.object(config, "INCLUDED_MODELS", THIRD_MODEL_INCLUDED):
            with mock.patch.object(ehc, "build_inventory", side_effect=fake_build_inventory):
                with mock.patch("sys.argv", argv):
                    with self.assertRaises(_StopEarly):
                        ehc.main()

        self.assertEqual(captured["models"], CANONICAL_BUNDLE_MODELS)


if __name__ == "__main__":
    unittest.main()
