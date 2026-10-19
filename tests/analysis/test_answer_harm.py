"""Unit tests for the answer-harm v2 analysis (synthetic data, no files)."""

from __future__ import annotations

import unittest

from prematurestop.analysis.fixed_cue_matrix import answer_harm as AH
from prematurestop.analysis.fixed_cue_matrix.results_bundle.comparisons import COMPARISONS


def run(correct, final_full=True, local_full=True, anchor_full=False, hop="2", flagged=False):
    return AH.Run(correct, flagged, final_full, local_full, anchor_full, hop)


class StatisticsTests(unittest.TestCase):
    def test_net_harm_counts_discordant_pairs_only(self):
        pairs = [(True, False)] * 3 + [(False, True)] * 1 + [(True, True)] * 4 + [(False, False)] * 2
        n, b, c, nh = AH.net_harm(pairs)
        self.assertEqual((n, b, c), (10, 3, 1))
        self.assertAlmostEqual(nh, 0.2)

    def test_mcnemar_exact_known_values(self):
        self.assertAlmostEqual(AH.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(AH.mcnemar_exact(5, 5), 1.0)
        # b=10, c=0: two-sided exact p = 2 * 0.5**10
        self.assertAlmostEqual(AH.mcnemar_exact(10, 0), 2 * 0.5 ** 10)
        self.assertAlmostEqual(AH.mcnemar_exact(0, 10), AH.mcnemar_exact(10, 0))

    def test_mcnemar_large_n_does_not_overflow(self):
        p = AH.mcnemar_exact(1486, 1190)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1e-4)

    def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic(self):
        strata = [[(True, False)] * 30 + [(False, True)] * 10 + [(True, True)] * 60]
        lo, hi = AH.bootstrap_ci(strata)
        self.assertLess(lo, 0.2)
        self.assertGreater(hi, 0.2)
        self.assertEqual((lo, hi), AH.bootstrap_ci(strata))

    def test_bootstrap_ci_zero_when_no_discordant_pairs(self):
        self.assertEqual(AH.bootstrap_ci([[(True, True)] * 20]), (0.0, 0.0))

    def test_bh_qvalues_monotone_and_bounded(self):
        q = AH.bh_qvalues([0.01, 0.04, 0.03, 0.5])
        self.assertAlmostEqual(q[0], 0.04)
        self.assertTrue(all(0 <= x <= 1 for x in q))
        self.assertLessEqual(q[0], q[2])


class PairingAndTransitionTests(unittest.TestCase):
    def test_transition_classes(self):
        self.assertEqual(AH.transition(True, False), "evidence_lost")
        self.assertEqual(AH.transition(False, True), "evidence_gained")
        self.assertEqual(AH.transition(True, True), "both_complete")
        self.assertEqual(AH.transition(False, False), "both_incomplete")
        self.assertIsNone(AH.transition(None, True))

    def test_pairs_only_questions_present_in_both_conditions(self):
        comp = next(c for c in COMPARISONS if c.cue_condition == "Natural_PreRegisteredSufficiency"
                    and c.control_condition == "NaturalNeutral")
        runs = {
            ("qwen14b", "hotpot", "NaturalNeutral", "q1"): run(True),
            ("qwen14b", "hotpot", "NaturalNeutral", "q2"): run(True),
            ("qwen14b", "hotpot", comp.cue_condition, "q1"): run(False),
            ("qwen14b", "hotpot", comp.cue_condition, "q3"): run(True),
        }
        pairs = AH.make_pairs(AH.index_runs(runs), comp, "qwen14b", "hotpot")
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0][0].correct, pairs[0][1].correct), (True, False))

    def test_family_assignment(self):
        fams = {c.comparison_name: AH.family(c) for c in COMPARISONS}
        primary = [n for n, f in fams.items() if f == "primary_attack"]
        self.assertEqual(len(primary), 2)
        self.assertEqual(fams["Natural Neutral vs Natural Clean"], "control")
        self.assertEqual(fams["Continue Search vs Natural Neutral"], "control")


class EndToEndTests(unittest.TestCase):
    def test_harm_lands_in_evidence_lost_class(self):
        comp = next(c for c in COMPARISONS if c.cue_condition == "Natural_PreRegisteredSufficiencySaturation"
                    and c.control_condition == "NaturalNeutral")
        runs = {}
        # 10 questions: control complete+correct; cue lost evidence and is wrong
        for i in range(10):
            runs[("qwen14b", "hotpot", "NaturalNeutral", f"l{i}")] = run(True, final_full=True)
            runs[("qwen14b", "hotpot", comp.cue_condition, f"l{i}")] = run(False, final_full=False)
        # 10 questions: both complete, no change
        for i in range(10):
            runs[("qwen14b", "hotpot", "NaturalNeutral", f"c{i}")] = run(True)
            runs[("qwen14b", "hotpot", comp.cue_condition, f"c{i}")] = run(True)
        tables = AH.analyse(runs)
        mech = {(r["comparison"], r["boundary"], r["model"], r["transition"]): r for r in tables["mechanism"]}
        lost = mech[(comp.comparison_name, "final_report_input", "qwen14b", "evidence_lost")]
        both = mech[(comp.comparison_name, "final_report_input", "qwen14b", "both_complete")]
        self.assertEqual((lost["n_pairs"], lost["ctl_correct_cue_wrong"]), (10, 10))
        self.assertAlmostEqual(lost["net_harm_pp"], 100.0)
        self.assertEqual(both["net_harm_pp"], 0.0)
        cell = next(r for r in tables["cells"] if r["comparison"] == comp.comparison_name
                    and r["model"] == "qwen14b" and r["dataset"] == "hotpot")
        self.assertAlmostEqual(cell["net_harm_pp"], 50.0)


if __name__ == "__main__":
    unittest.main()
