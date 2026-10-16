"""Regression tests for the two real evidence-boundary bugs found and fixed
during the corrected paper-evaluation pass: (1) episode-terminal coverage
must be a true union across all searches in the episode, never trust a
single stored snapshot; (2) paragraph-ID matching must be normalized to the
per-example index, because MuSiQue's `required_paragraph_ids` are bare
indices while citation text always embeds the full compound ID.
"""

from __future__ import annotations

import unittest

from prematurestop.evaluation.evidence import (
    _cited_paragraph_ids,
    _cumulative_union_coverage,
    _normalized_required_paragraph_ids,
    _paragraph_index,
)
from prematurestop.evaluation.trace_schema import ResearcherEpisode, SearchEvent


def _search_event(step, required_ids, required_paragraphs, retrieved_paragraphs):
    return SearchEvent(
        step=step, subgraph_id="ep0", supervisor_delegation_index=0, attacked_observation=False,
        coverage_fraction_after_observation=None,
        dependencies_seen_after_observation=(), missing_dependency_ids_after_observation=(),
        required_dependency_ids=tuple(required_ids), required_paragraph_ids=tuple(required_paragraphs),
        retrieved_paragraph_ids=tuple(retrieved_paragraphs), normalized_queries=(),
    )


class CumulativeUnionCoverageTests(unittest.TestCase):
    def test_union_recovers_full_coverage_even_when_last_search_alone_would_not(self):
        """Regression for the exact violation pattern found in real Llama traces:
        search 0 finds dep A, search 1 finds dep A+B, search 2's OWN yield only
        shows dep A again (as if B were "forgotten"). The true union across the
        whole episode must still show both A and B found.
        """
        episode = ResearcherEpisode(
            subgraph_id="ep0", episode_order=0, supervisor_iter=1, decisions=[],
            search_events=[
                _search_event(1, ["A", "B"], ["pA", "pB"], ["pA"]),
                _search_event(2, ["A", "B"], ["pA", "pB"], ["pA", "pB"]),
                _search_event(3, ["A", "B"], ["pA", "pB"], ["pA"]),  # "forgets" pB in isolation
            ],
        )
        coverage, missing = _cumulative_union_coverage(episode)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(missing, 0)

    def test_union_never_exceeds_true_retrieved_set(self):
        episode = ResearcherEpisode(
            subgraph_id="ep0", episode_order=0, supervisor_iter=1, decisions=[],
            search_events=[_search_event(1, ["A", "B"], ["pA", "pB"], ["pA"])],
        )
        coverage, missing = _cumulative_union_coverage(episode)
        self.assertEqual(coverage, 0.5)
        self.assertEqual(missing, 1)

    def test_no_search_events_returns_none(self):
        episode = ResearcherEpisode(subgraph_id="ep0", episode_order=0, supervisor_iter=1, decisions=[], search_events=[])
        self.assertEqual(_cumulative_union_coverage(episode), (None, None))


class ParagraphIndexNormalizationTests(unittest.TestCase):
    def test_full_compound_id_extracts_index(self):
        self.assertEqual(_paragraph_index("hotpot:5a70f3:9:some_slug:4305b4cd9acb"), "9")

    def test_bare_index_passes_through_unchanged(self):
        """MuSiQue's required_paragraph_ids are bare indices in the raw trace data."""
        self.assertEqual(_paragraph_index("18"), "18")

    def test_musique_style_mismatch_is_resolved_by_normalization(self):
        """Regression for the real bug: bare required index '18' must match a
        cited full compound ID whose index segment is also '18', even though
        the two strings are not equal."""
        required = _normalized_required_paragraph_ids(("18", "13"))
        cited = _cited_paragraph_ids(
            "SOURCE: benchmark://musique/2hop__1/musique:2hop__1:18:zhejiang:8aa6fdc49d29\n"
            "SOURCE: benchmark://musique/2hop__1/musique:2hop__1:13:hokkien:b2ecfabebb35"
        )
        self.assertEqual(required, {"18", "13"})
        self.assertEqual(cited, {"18", "13"})
        self.assertEqual(required.intersection(cited), {"18", "13"})

    def test_full_format_dataset_citation_matching_still_works(self):
        required = _normalized_required_paragraph_ids(("hotpot:ex:9:slug_a:hash1", "hotpot:ex:0:slug_b:hash2"))
        cited = _cited_paragraph_ids("benchmark://hotpot/ex/hotpot:ex:9:slug_a:hash1")
        self.assertEqual(len(required.intersection(cited)), 1)


if __name__ == "__main__":
    unittest.main()
