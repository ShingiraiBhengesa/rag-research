"""Tests for pcr.py's CDF math using synthetic PCRRecord objects (no production
data dependency) -- these check the mathematical *properties* Section 22's
validation gates require, independent of any specific machine's raw traces.
"""

from __future__ import annotations

import unittest

from analysis import config
from analysis.pcr import (
    PCRRecord,
    completion_cdf_for_cell,
    unconditional_incidence_for_cell,
)


def _rec(route, t_i, branch_reached=True, no_branch_reason=None):
    return PCRRecord(
        model="qwen14b", dataset="hotpot", condition="NaturalClean", example_id="x",
        trace_path="", parse_ok=True, parse_errors="",
        branch_reached=branch_reached, no_branch_reason=no_branch_reason,
        terminal_post_branch_index=t_i, terminal_route=route, overall_terminal_route=route,
        post_injection_search_count=0, coverage_fraction_at_injection=None,
        required_dependency_count=0, supervisor_redelegation_count=0,
    )


class CompletionCdfTests(unittest.TestCase):
    def test_reaches_exactly_100_when_populated(self):
        records = [
            _rec(config.ROUTE_RESEARCH_COMPLETE, 1),
            _rec(config.ROUTE_NO_TOOL_EXIT, 2),
            _rec(config.ROUTE_RESEARCH_COMPLETE, 5),
        ]
        cdf = completion_cdf_for_cell(records, k_max=5)
        self.assertEqual(cdf["per_step"][-1]["combined_pct"], 100.0)

    def test_route_sum_identity_at_every_step(self):
        records = [
            _rec(config.ROUTE_RESEARCH_COMPLETE, 1),
            _rec(config.ROUTE_NO_TOOL_EXIT, 2),
            _rec(config.ROUTE_RESEARCH_COMPLETE, 3),
            _rec(config.ROUTE_NO_TOOL_EXIT, 4),
        ]
        cdf = completion_cdf_for_cell(records, k_max=4)
        for step in cdf["per_step"]:
            self.assertAlmostEqual(step["rc_pct"] + step["no_tool_pct"], step["combined_pct"], places=9)
            self.assertEqual(step["rc_cum_n"] + step["no_tool_cum_n"], step["combined_cum_n"])

    def test_no_arbitrary_cutoff_k_max_38_supported(self):
        records = [_rec(config.ROUTE_RESEARCH_COMPLETE, 38)]
        cdf = completion_cdf_for_cell(records, k_max=38)
        self.assertEqual(len(cdf["per_step"]), 38)
        self.assertEqual(cdf["per_step"][-1]["combined_pct"], 100.0)
        self.assertEqual(cdf["per_step"][-2]["combined_pct"], 0.0)

    def test_censored_and_no_search_rows_excluded_from_denominator(self):
        records = [
            _rec(config.ROUTE_RESEARCH_COMPLETE, 1),
            _rec(None, None),  # censored: branch reached, no terminal observed
            _rec(None, None, branch_reached=False, no_branch_reason="no_search_attempted"),
        ]
        cdf = completion_cdf_for_cell(records, k_max=3)
        self.assertEqual(cdf["n_eligible"], 1)
        self.assertEqual(cdf["per_step"][0]["combined_pct"], 100.0)


class UnconditionalIncidenceTests(unittest.TestCase):
    def test_plateaus_below_100_when_some_rows_never_terminate(self):
        records = [
            _rec(config.ROUTE_RESEARCH_COMPLETE, 1),
            _rec(None, None),  # never observed a terminal decision
            _rec(None, None),
        ]
        result = unconditional_incidence_for_cell(records, k_max=5)
        self.assertEqual(result["n_total"], 3)
        last = result["per_step"][-1]
        self.assertLess(last["cum_pct"], 100.0)
        self.assertAlmostEqual(last["cum_pct"], 100.0 / 3.0, places=5)

    def test_denominator_is_every_row_not_just_eligible_ones(self):
        records = [_rec(config.ROUTE_RESEARCH_COMPLETE, 1), _rec(None, None, branch_reached=False)]
        result = unconditional_incidence_for_cell(records, k_max=1)
        self.assertEqual(result["n_total"], 2)


if __name__ == "__main__":
    unittest.main()
