"""Tests for the final_paper_evaluation subpackage: termination-route
taxonomy (mutually exclusive, sums to total) and evidence-loss/recovery
classification, using synthetic records (no production-data dependency).
"""

from __future__ import annotations

import unittest

from analysis import config
from analysis.final_paper_evaluation import answer_harm, recovery
from analysis.final_paper_evaluation.master import MasterRow


def _row(**overrides) -> MasterRow:
    base = dict(
        model="qwen14b", dataset="hotpot", condition="NaturalNeutral", example_id="ex1",
        shard_index=0, run_id="r1", execution_valid=True, final_status="ok", error="",
        corpus_sha256="", evidence_sha256="", manifest_sha256="",
        question_hash="q1", decoding_config_hash="d1",
        pre_injection_trace_hash="p1", first_retrieved_observation_hash="o1", branch_point_identifier="p1",
        cue_present=False, evidence_state_at_injection_hash="e1",
        coverage_fraction_at_injection=0.5, missing_dependency_count_at_injection=1,
        branch_reached=True, no_branch_reason=None, first_post_injection_action="tavily_search",
        local_completion_route=config.ROUTE_RESEARCH_COMPLETE, local_completion_step=2,
        post_injection_search_count=1, supervisor_redelegation_count=0,
        episode0_terminal_coverage_fraction=0.5, compressed_notes_coverage_fraction=0.5,
        final_input_coverage_fraction=0.5, whole_run_cumulative_retrieval_coverage_fraction=0.5,
        raw_final_report="Answer: X", extracted_short_answer="X", extractable=True,
        extraction_reason="ok", semantic_label="correct", semantic_correct=True,
        exact_match=1.0, token_f1=1.0, gold_containment=True, gold_answer="X",
        evaluator_version="v2",
    )
    base.update(overrides)
    return MasterRow(**base)


class EvidenceLossClassificationTests(unittest.TestCase):
    def test_no_loss_when_final_input_not_reduced(self):
        baseline = _row(episode0_terminal_coverage_fraction=0.5, final_input_coverage_fraction=1.0)
        attack = _row(episode0_terminal_coverage_fraction=0.5, final_input_coverage_fraction=1.0)
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertEqual(result["evidence_loss_stratum"], answer_harm.STRATUM_NO_LOSS)

    def test_recovered_when_early_loss_but_final_catches_up(self):
        baseline = _row(episode0_terminal_coverage_fraction=1.0, final_input_coverage_fraction=1.0)
        attack = _row(episode0_terminal_coverage_fraction=0.5, final_input_coverage_fraction=1.0)
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertEqual(result["evidence_loss_stratum"], answer_harm.STRATUM_RECOVERED)

    def test_unrecovered_when_final_input_still_lower(self):
        baseline = _row(episode0_terminal_coverage_fraction=1.0, final_input_coverage_fraction=1.0)
        attack = _row(episode0_terminal_coverage_fraction=0.5, final_input_coverage_fraction=0.5)
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertEqual(result["evidence_loss_stratum"], answer_harm.STRATUM_UNRECOVERED)

    def test_full_hop_flip_detected(self):
        baseline = _row(final_input_coverage_fraction=1.0)
        attack = _row(final_input_coverage_fraction=0.8)
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertTrue(result["full_hop_control_to_missing_hop_attack"])

    def test_compression_only_loss_requires_ep0_unaffected(self):
        baseline = _row(episode0_terminal_coverage_fraction=0.8, compressed_notes_coverage_fraction=0.8)
        attack = _row(episode0_terminal_coverage_fraction=0.8, compressed_notes_coverage_fraction=0.4)
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertTrue(result["compression_only_loss"])

    def test_unresolved_when_data_missing(self):
        baseline = _row(episode0_terminal_coverage_fraction=None)
        attack = _row()
        result = answer_harm.classify_pair_evidence_loss(baseline, attack)
        self.assertEqual(result["evidence_loss_stratum"], "unresolved_missing_evidence_state")


class SupervisorRecoveryClassificationTests(unittest.TestCase):
    def _pcr_record(self, redeleg_count):
        from analysis.pcr import PCRRecord

        return PCRRecord(
            model="qwen14b", dataset="hotpot", condition="NaturalNeutral", example_id="ex1", trace_path="",
            parse_ok=True, parse_errors="", branch_reached=True, no_branch_reason=None,
            terminal_post_branch_index=1, terminal_route=config.ROUTE_RESEARCH_COMPLETE,
            overall_terminal_route=config.ROUTE_RESEARCH_COMPLETE, post_injection_search_count=0,
            coverage_fraction_at_injection=0.5, required_dependency_count=2,
            supervisor_redelegation_count=redeleg_count,
        )

    def _ev_record(self, ep0, final):
        from analysis.evidence import EvidenceRecord

        return EvidenceRecord(
            model="qwen14b", dataset="hotpot", condition="NaturalNeutral", example_id="ex1",
            required_dependency_count=2, injection_coverage_fraction=0.5, injection_missing_count=1,
            episode0_terminal_coverage_fraction=ep0, episode0_terminal_missing_count=0, episode0_had_any_search=True,
            compressed_notes_available=True, compressed_notes_coverage_fraction=ep0, compressed_notes_cited_paragraph_count=1,
            final_input_available=True, final_input_coverage_fraction=final, final_input_missing_count=0,
            final_input_cited_paragraph_count=1, whole_run_cumulative_retrieval_coverage_fraction=final,
            whole_run_cumulative_retrieval_missing_count=0, full_hop_coverage_at_injection=False,
            full_hop_coverage_at_episode0_terminal=ep0 >= 1.0 if ep0 is not None else None,
            full_hop_coverage_at_final_input=final >= 1.0 if final is not None else None,
            full_hop_whole_run_cumulative_retrieval=final >= 1.0 if final is not None else None,
        )

    def test_no_redelegation_category(self):
        rows = recovery.build_supervisor_recovery_rows([self._pcr_record(0)], [self._ev_record(0.5, 0.5)])
        self.assertEqual(rows[0]["recovery_category"], recovery.RECOVERY_NO_REDELEGATION)

    def test_redelegation_requires_measurable_restoration_not_just_redelegating(self):
        """Redelegation happened but coverage did not improve -- must NOT be counted as recovery."""
        rows = recovery.build_supervisor_recovery_rows([self._pcr_record(1)], [self._ev_record(0.5, 0.5)])
        self.assertEqual(rows[0]["recovery_category"], recovery.RECOVERY_REDELEGATED_NO_RESTORE)

    def test_full_restoration_detected(self):
        rows = recovery.build_supervisor_recovery_rows([self._pcr_record(1)], [self._ev_record(0.5, 1.0)])
        self.assertEqual(rows[0]["recovery_category"], recovery.RECOVERY_REDELEGATED_FULL)


if __name__ == "__main__":
    unittest.main()
