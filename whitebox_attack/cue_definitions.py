# cue_definitions.py — canonical cue suffix strings for all cue-condition scripts.
# No imports. No side effects. Safe to import from any script.
#
# wrong_gap_assessment.py defines an identical dict locally (PERTURBATION_FAMILIES)
# and is not imported here to avoid touching it while its jobs are active.
# The smoke-test validation script asserts exact string equality between the two.

PERTURBATION_FAMILIES = {
    "epistemic_confirmation":        "This directly confirms the point in question.",
    "search_match_cue":              "This matches exactly what was being investigated.",
    "source_coverage":               "Multiple independent sources mention this same detail.",
    "epistemic_confirmation_source": "This directly confirms the point in question. "
                                     "Multiple independent sources mention this same detail.",
    "search_match_source":           "This matches exactly what was being investigated. "
                                     "Multiple independent sources mention this same detail.",
}
