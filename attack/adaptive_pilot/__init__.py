"""Adaptive cue pilot: contextual, feedback-driven adaptation of the existing
premature-completion cue attacks against Open Deep Research.

This package is additive. It never edits the canonical runtime
(`attack/runtime/*`, `open_deep_research/*`) or canonical analysis
(`analysis/*`); it imports them read-only. See README.md and
INSPECTION_REPORT.md in this directory.
"""

PILOT_VERSION = "adaptive_cue_pilot_v1"
