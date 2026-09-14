"""Tests for the GPT-OSS exclusion guard -- must never silently pass a path
belonging to an excluded model family.

Updated 2026-09-13: gemma27b completed its production matrix (432/432
shards) and moved from EXCLUDED_MODEL_FAMILIES into INCLUDED_MODELS, per the
plan already recorded in PUBLISHING.md's "Deferred Cleanup After All Jobs
Finish" section. gpt_oss_20b remains excluded (still blocked, see
PUBLISHING.md)."""

from __future__ import annotations

import unittest

from analysis import config


class ExclusionGuardTests(unittest.TestCase):
    def test_included_models_are_qwen_llama_and_gemma(self):
        self.assertEqual(set(config.INCLUDED_MODELS), {"qwen14b", "llama8b", "gemma27b"})

    def test_excluded_models_are_gpt_oss_only(self):
        self.assertEqual(set(config.EXCLUDED_MODEL_FAMILIES), {"gpt_oss_20b"})

    def test_gpt_oss_path_raises(self):
        with self.assertRaises(config.ExcludedModelPathError):
            config.assert_path_not_excluded(
                "results/runs/gpt_oss_20b/hotpot/NaturalClean/shard00of09.jsonl"
            )

    def test_qwen_llama_and_gemma_paths_are_not_excluded(self):
        config.assert_path_not_excluded("results/traces/qwen14b/branches/hotpot/x/NaturalClean/trace.json")
        config.assert_path_not_excluded("results/traces/llama8b/branches/hotpot/x/NaturalClean/trace.json")
        config.assert_path_not_excluded("results/traces/gemma27b/branches/hotpot/x/NaturalClean/trace.json")

    def test_assert_included_model_rejects_excluded_family(self):
        with self.assertRaises(config.ExcludedModelPathError):
            config.assert_included_model("gpt_oss_20b")

    def test_assert_included_model_accepts_gemma(self):
        config.assert_included_model("gemma27b")  # must not raise

    def test_all_twelve_conditions_present_and_grouped_exactly_once(self):
        self.assertEqual(len(config.CONDITIONS), 12)
        grouped = [c for group in config.CONDITION_GROUPS.values() for c in group]
        self.assertEqual(sorted(grouped), sorted(config.CONDITIONS))
        self.assertEqual(len(grouped), len(set(grouped)))


if __name__ == "__main__":
    unittest.main()
