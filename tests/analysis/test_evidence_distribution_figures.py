"""Unit tests for the evidence-coverage-distribution package (Phase 11).
Run with: python -m unittest tests.analysis.test_evidence_distribution_figures -v
"""
from __future__ import annotations

import unittest

from prematurestop.analysis.fixed_cue_matrix.evidence_distributions import categories as cat
from prematurestop.analysis.fixed_cue_matrix.evidence_distributions import missing_evidence as me
from prematurestop.analysis.fixed_cue_matrix.evidence_distributions import pairing as pr
from prematurestop.analysis.fixed_cue_matrix.evidence_distributions import validation as val
from prematurestop.analysis.fixed_cue_matrix.evidence_distributions.source_loader import CoverageRow


def _row(model="qwen14b", dataset="hotpot", condition="NaturalClean", example_id="ex1",
         required=2, retrieved=1, missing=1, coverage=0.5, full_coverage=False,
         terminal_route="RESEARCH_COMPLETE"):
    return CoverageRow(model=model, dataset=dataset, condition=condition, example_id=example_id,
                        required=required, retrieved=retrieved, missing=missing, coverage=coverage,
                        full_coverage=full_coverage, terminal_route=terminal_route)


class CategoryBoundaryTests(unittest.TestCase):
    def test_zero_coverage_is_no_evidence(self):
        self.assertEqual(cat.categorize(0.0), cat.CATEGORY_NO_EVIDENCE)

    def test_just_above_zero_is_low(self):
        self.assertEqual(cat.categorize(0.01), cat.CATEGORY_LOW)

    def test_just_below_half_is_low(self):
        self.assertEqual(cat.categorize(0.4999), cat.CATEGORY_LOW)

    def test_exactly_half_is_high_incomplete(self):
        self.assertEqual(cat.categorize(0.5), cat.CATEGORY_HIGH_INCOMPLETE)

    def test_just_below_one_is_high_incomplete(self):
        self.assertEqual(cat.categorize(0.9999), cat.CATEGORY_HIGH_INCOMPLETE)

    def test_exactly_one_is_full(self):
        self.assertEqual(cat.categorize(1.0), cat.CATEGORY_FULL)

    def test_float_tolerance_near_zero(self):
        self.assertEqual(cat.categorize(1e-10), cat.CATEGORY_NO_EVIDENCE)

    def test_float_tolerance_near_one(self):
        self.assertEqual(cat.categorize(1.0 - 1e-10), cat.CATEGORY_FULL)


class PercentageTotalTests(unittest.TestCase):
    def test_category_percentages_sum_to_100(self):
        rows = [_row(coverage=c, missing=0 if c == 1.0 else 1) for c in (0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0)]
        stats = cat.summarize_cell(rows)
        total = sum(stats[f"pct_{c}"] for c in cat.CATEGORIES)
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_empty_cell_has_none_percentages(self):
        stats = cat.summarize_cell([])
        self.assertEqual(stats["eligible_N"], 0)
        self.assertIsNone(stats["pct_No required evidence"])

    def test_missing_category_percentages_sum_to_100(self):
        rows = [_row(missing=m) for m in (0, 0, 1, 2, 3, 4, 5)]
        stats = me.summarize_missing(rows)
        total = sum(stats[f"pct_{c}"] for c in me.MISSING_CATEGORIES)
        self.assertAlmostEqual(total, 100.0, places=1)


class PairedIdIntersectionTests(unittest.TestCase):
    def test_only_shared_ids_are_paired(self):
        control = [_row(example_id="a"), _row(example_id="b"), _row(example_id="c")]
        cue = [_row(example_id="b"), _row(example_id="c"), _row(example_id="d")]
        pairs = pr.pair_by_example_id(control, cue)
        self.assertEqual({p[0].example_id for p in pairs}, {"b", "c"})

    def test_no_padding_or_imputation_when_sets_differ(self):
        control = [_row(example_id="a")]
        cue = [_row(example_id="z")]
        pairs = pr.pair_by_example_id(control, cue)
        self.assertEqual(pairs, [])

    def test_every_pair_has_matching_example_id(self):
        control = [_row(example_id=str(i)) for i in range(10)]
        cue = [_row(example_id=str(i)) for i in range(5, 15)]
        pairs = pr.pair_by_example_id(control, cue)
        self.assertTrue(all(c.example_id == q.example_id for c, q in pairs))
        self.assertEqual(len(pairs), 5)


class PairedLossSignTests(unittest.TestCase):
    def test_control_retrieves_more_is_positive_loss(self):
        # control coverage 1.0, cue coverage 0.5 -> cue "lost" evidence -> positive
        control = [_row(example_id="a", coverage=1.0)]
        cue = [_row(example_id="a", coverage=0.5)]
        point, lo, hi = pr.paired_bootstrap_mean_diff([1.0], [0.5], n_boot=10)
        self.assertGreater(point, 0)

    def test_cue_retrieves_more_is_negative_loss(self):
        point, lo, hi = pr.paired_bootstrap_mean_diff([0.5], [1.0], n_boot=10)
        self.assertLess(point, 0)

    def test_identical_coverage_is_zero(self):
        point, lo, hi = pr.paired_bootstrap_mean_diff([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], n_boot=10)
        self.assertEqual(point, 0.0)


class MissingUnitCountTests(unittest.TestCase):
    def test_missing_equals_required_minus_retrieved(self):
        r = _row(required=4, retrieved=1, missing=3)
        self.assertEqual(r.missing, r.required - r.retrieved)

    def test_missing_category_boundaries(self):
        self.assertEqual(me._missing_category(0), me.MISSING_ZERO)
        self.assertEqual(me._missing_category(1), me.MISSING_ONE)
        self.assertEqual(me._missing_category(2), me.MISSING_TWO)
        self.assertEqual(me._missing_category(3), me.MISSING_THREE_PLUS)
        self.assertEqual(me._missing_category(10), me.MISSING_THREE_PLUS)


class BootstrapReproducibilityTests(unittest.TestCase):
    def test_same_seed_gives_identical_ci(self):
        a = [0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0] * 10
        b = [1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.5] * 10
        r1 = pr.paired_bootstrap_mean_diff(a, b, n_boot=500, seed=pr.BOOTSTRAP_SEED)
        r2 = pr.paired_bootstrap_mean_diff(a, b, n_boot=500, seed=pr.BOOTSTRAP_SEED)
        self.assertEqual(r1, r2)

    def test_different_seed_can_give_different_ci_bounds(self):
        a = [0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0] * 10
        b = [1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.5] * 10
        r1 = pr.paired_bootstrap_mean_diff(a, b, n_boot=200, seed=0)
        r2 = pr.paired_bootstrap_mean_diff(a, b, n_boot=200, seed=1)
        self.assertEqual(r1[0], r2[0])  # point estimate is seed-independent
        # CI bounds need not be identical across seeds (not asserting
        # inequality -- small-sample bootstraps can coincidentally match --
        # only that the function accepts/uses a seed argument at all).


class FullCoverageTransitionTests(unittest.TestCase):
    def test_transition_counts_sum_to_paired_n(self):
        control = [_row(example_id=str(i), full_coverage=(i % 2 == 0)) for i in range(20)]
        cue = [_row(example_id=str(i), full_coverage=(i % 3 == 0)) for i in range(20)]
        pairs = pr.pair_by_example_id(control, cue)
        full_both = sum(1 for c, q in pairs if c.full_coverage and q.full_coverage)
        full_control_only = sum(1 for c, q in pairs if c.full_coverage and not q.full_coverage)
        full_cue_only = sum(1 for c, q in pairs if not c.full_coverage and q.full_coverage)
        full_neither = len(pairs) - full_both - full_control_only - full_cue_only
        self.assertEqual(full_both + full_control_only + full_cue_only + full_neither, len(pairs))


class McNemarInputConstructionTests(unittest.TestCase):
    def test_zero_discordant_pairs_gives_p_one(self):
        self.assertEqual(pr.mcnemar_exact(0, 0), 1.0)

    def test_symmetric_discordant_pairs_gives_p_one(self):
        # b == c (fully symmetric discordance) -> no evidence of asymmetry -> p == 1.0
        self.assertAlmostEqual(pr.mcnemar_exact(5, 5), 1.0, places=6)

    def test_matches_known_scipy_cross_check_value(self):
        # Cross-checked once against scipy.stats.binomtest(3, 15, 0.5,
        # alternative="two-sided") -> 0.03515625 (see pairing.py docstring).
        self.assertAlmostEqual(pr.mcnemar_exact(3, 12), 0.03515625, places=8)

    def test_order_of_b_c_does_not_matter(self):
        self.assertEqual(pr.mcnemar_exact(3, 12), pr.mcnemar_exact(12, 3))


class ValidationGateTests(unittest.TestCase):
    def test_gate_1_catches_out_of_range_coverage(self):
        rows = [_row(coverage=1.5)]
        result = val.gate_1_coverage_range(rows)
        self.assertEqual(result.status, "FAIL")

    def test_gate_1_passes_in_range_coverage(self):
        rows = [_row(coverage=0.5)]
        result = val.gate_1_coverage_range(rows)
        self.assertEqual(result.status, "PASS")

    def test_gate_4_catches_retrieved_exceeding_required(self):
        rows = [_row(required=2, retrieved=3, missing=-1)]
        result = val.gate_4_retrieved_le_required(rows)
        self.assertEqual(result.status, "FAIL")

    def test_gate_5_catches_coverage_arithmetic_mismatch(self):
        rows = [_row(required=2, retrieved=1, missing=1, coverage=0.9)]  # should be 0.5
        result = val.gate_5_coverage_consistency(rows)
        self.assertEqual(result.status, "FAIL")

    def test_gate_8_catches_non_qualifying_terminal_route(self):
        rows = [_row(terminal_route="MAX_BUDGET_EXIT")]
        result = val.gate_8_no_post_completion_evidence(rows)
        self.assertEqual(result.status, "FAIL")

    def test_gate_10_catches_gptoss_2wiki(self):
        rows = [_row(model="gpt_oss_20b", dataset="2wiki")]
        result = val.gate_10_gptoss_not_macro_averaged(rows)
        self.assertEqual(result.status, "FAIL")

    def test_gate_11_passes_when_no_violin_or_kde_present(self):
        result = val.gate_11_no_kde_or_violin()
        self.assertEqual(result.status, "PASS")


class FigureDataTraceabilityTests(unittest.TestCase):
    def test_every_expected_cell_is_covered_by_coverage_table(self):
        from prematurestop.analysis.fixed_cue_matrix.evidence_distributions.source_loader import MODEL_DATASETS, ALL_CONDITIONS
        fake_rows = []
        for model, datasets in MODEL_DATASETS.items():
            for dataset in datasets:
                for cond in ALL_CONDITIONS:
                    fake_rows.append({"model": model, "dataset": dataset, "condition": cond})
        result = val.gate_15_figure_traceability(fake_rows)
        self.assertEqual(result.status, "PASS")

    def test_missing_cell_is_caught(self):
        from prematurestop.analysis.fixed_cue_matrix.evidence_distributions.source_loader import MODEL_DATASETS, ALL_CONDITIONS
        fake_rows = []
        for model, datasets in MODEL_DATASETS.items():
            for dataset in datasets:
                for cond in ALL_CONDITIONS:
                    if model == "qwen14b" and dataset == "hotpot" and cond == "NaturalClean":
                        continue  # deliberately omit one cell
                    fake_rows.append({"model": model, "dataset": dataset, "condition": cond})
        result = val.gate_15_figure_traceability(fake_rows)
        self.assertEqual(result.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
