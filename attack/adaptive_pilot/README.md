# Adaptive-cue pilot

A controlled, feedback-driven adaptation of the **existing** premature-completion
cue attacks against Open Deep Research. It optimizes how the existing cues are
*written and appended* for a specific question and observation. It does not add
a new taxonomy, change the legitimate question or clean documents, change ODR's
prompts/tools/budget, or introduce a new PCR definition.

Start with `INSPECTION_REPORT.md` (findings, status, blockers) and
`nrp/README.md` (execution). This package is **additive**: it imports the
canonical runtime and analysis read-only. Its one touch to shared code is an
additive, production-no-op suffix-override hook in `attack/runtime/cues.py`
(`PILOT_SUFFIX_OVERRIDE_KEY`), guarded by tests.

## Arms (per victim_model × dataset × example × condition)

| arm | suffix | access |
|---|---|---|
| Control | none (canonical NaturalClean) | — |
| Static | canonical condition suffix, verbatim | — |
| Dynamic | one attacker-generated contextual suffix, no victim feedback | context-only |
| Optimized | the Dynamic suffix revised once using whitelisted feedback | bounded-feedback |

Dynamic is attempt one of the optimization; the revised (Optimized) suffix is
the final payload by rule (never best-of-two). Each arm is a fresh victim run
replayed from the same anchor; the shared clean prefix is verified per cell.

## Modules

- `cue_specs.py` — versioned spec + pinned static texts for all 12 conditions.
- `validation.py` — deterministic suffix/observation validation + token counters.
- `attacker.py` / `prompts/` — stateless generation + refinement (hashed prompts).
- `feedback.py` — `victim_feedback_v1`, two access settings, forbidden-key guard.
- `runner.py` — four-arm cell with two-attempt optimization + faithful replay.
- `split_manifest.py` — dev/held-out splits (seed 20261005).
- `config.py` — frozen victims/attacker/conditions/budgets + launch selection.
- `freeze.py` — one hashed manifest of the whole procedure before held-out.
- `analysis.py` — canonical PCR, paired deltas, optimization view, bootstrap CIs.
- `cli.py` — idempotent, append-only, resumable driver.
- `nrp/` — serving, preflight, throughput, and pod templates (verify placeholders).
- `tests/test_pilot.py` — append-only, feedback-exclusion, isolation, PCR,
  idempotence (30 tests).

## Reporting rules (task sections 12, 18)

Report Static−Control, Dynamic−Static, Optimized−Dynamic separately, by model /
dataset / condition, paired at the question level with uncertainty intervals.
Report first-attempt and final-attempt outcomes including revision regressions,
alongside attacker query budget and cost. Earlier completion is a termination
effect, not evidence of research harm; evidence-deficient completion,
full-workflow termination, and technical failure are kept distinct. This is a
feasibility pilot with small cells, not a ranking of all 12 cues.
