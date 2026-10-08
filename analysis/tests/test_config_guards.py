"""Tests for the excluded-model guard -- must never silently pass a path
belonging to an excluded model family.

Updated 2026-09-28: gemma27b's fresh, correctly-configured re-run completed
across all 4 datasets (see analysis/CHANGELOG.md and
analysis/config.py's MODEL_SCOPE_STATEMENT) and gemma27b is re-included in
INCLUDED_MODELS. gpt_oss_20b is also re-included, on the basis of its 3
completed datasets (hotpot, musique, multihoprag); its 2wiki dataset
remains at 0/108 shards (a separate, unrelated vLLM harmony-parser crash,
fix built but not yet deployed) -- there is no path-level exclusion for
this partial-completeness state, since completion_inventory.build_inventory
already tolerates a model missing an entire dataset. EXCLUDED_MODEL_FAMILIES
and _EXCLUDED_PATH_TOKENS are both empty as of this update; the guard
mechanism itself (assert_included_model / assert_path_not_excluded) is
kept, not removed, for the next time a model needs quarantining."""

from __future__ import annotations

import unittest

from analysis import config


class ExclusionGuardTests(unittest.TestCase):
    def test_included_models_are_all_four(self):
        self.assertEqual(set(config.INCLUDED_MODELS), {"qwen14b", "llama8b", "gemma27b", "gpt_oss_20b"})

    def test_no_models_currently_excluded(self):
        self.assertEqual(config.EXCLUDED_MODEL_FAMILIES, ())
        self.assertEqual(config._EXCLUDED_PATH_TOKENS, ())

    def test_no_path_is_excluded_while_the_token_list_is_empty(self):
        for path in (
            "results/runs/gpt_oss_20b/hotpot/NaturalClean/shard00of09.jsonl",
            "results/runs/gemma27b/hotpot/NaturalClean/shard00of09.jsonl",
            "results/traces/qwen14b/branches/hotpot/x/NaturalClean/trace.json",
            "results/traces/llama8b/branches/hotpot/x/NaturalClean/trace.json",
        ):
            config.assert_path_not_excluded(path)  # must not raise

    def test_assert_path_not_excluded_still_enforces_a_nonempty_token_list(self):
        # Regression guard for the mechanism itself, independent of what's
        # currently in _EXCLUDED_PATH_TOKENS: if something is ever added
        # back to that list, this function must still catch it.
        from unittest import mock

        with mock.patch.object(config, "_EXCLUDED_PATH_TOKENS", ("some_future_excluded_model",)):
            with self.assertRaises(config.ExcludedModelPathError):
                config.assert_path_not_excluded("results/runs/some_future_excluded_model/x")

    def test_assert_included_model_accepts_all_four(self):
        for model in ("qwen14b", "llama8b", "gemma27b", "gpt_oss_20b"):
            config.assert_included_model(model)  # must not raise

    def test_assert_included_model_still_rejects_a_truly_unknown_model(self):
        with self.assertRaises(config.ExcludedModelPathError):
            config.assert_included_model("not_a_real_model")

    def test_all_twelve_conditions_present_and_grouped_exactly_once(self):
        self.assertEqual(len(config.CONDITIONS), 12)
        grouped = [c for group in config.CONDITION_GROUPS.values() for c in group]
        self.assertEqual(sorted(grouped), sorted(config.CONDITIONS))
        self.assertEqual(len(grouped), len(set(grouped)))


if __name__ == "__main__":
    unittest.main()
