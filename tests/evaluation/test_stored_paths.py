"""Stored trace paths resolve to the production data root, not the repo root.

Run records written before the 2026-10-16 restructure store paths such as
`results/traces/<model>/...`, relative to the old repository root. After the
restructure that data lives under `$PREMATURESTOP_LOCAL_DATA`; resolving
against the new repo root silently marks every trace as missing.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from prematurestop.evaluation import config


class ResolveStoredPathTests(unittest.TestCase):
    def test_results_prefixed_path_resolves_under_production_root(self):
        with mock.patch.object(config, "DEFAULT_PRODUCTION_ROOT", Path("/data/prod")):
            got = config.resolve_stored_path("results/traces/qwen14b/hotpot/x/trace.json")
        self.assertEqual(got, Path("/data/prod/traces/qwen14b/hotpot/x/trace.json"))

    def test_absolute_path_passes_through(self):
        self.assertEqual(config.resolve_stored_path("/abs/trace.json"), Path("/abs/trace.json"))

    def test_other_relative_path_resolves_under_repo_root(self):
        self.assertEqual(config.resolve_stored_path("docs/x.json"), config.REPO_ROOT / "docs/x.json")

    def test_never_resolves_results_prefix_against_repo_root(self):
        with mock.patch.object(config, "DEFAULT_PRODUCTION_ROOT", Path("/data/prod")):
            got = config.resolve_stored_path("results/runs/a.jsonl")
        self.assertNotEqual(got, config.REPO_ROOT / "results/runs/a.jsonl")


if __name__ == "__main__":
    unittest.main()
