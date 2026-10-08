# rag-research

PhD research on premature-completion attacks against agentic RAG (Open Deep
Research). The canonical 12-condition study lives in `attack/` and `analysis/`.

## Active work: dynamic bridge-cue pilot (branch `pilot/dynamic-bridge-two-cues`)

Operating it on NRP? Read `experiments/dynamic_bridge_cue_pilot/HANDOVER_NRP.md`
first (rules + runbook), then `experiments/dynamic_bridge_cue_pilot/README.md`
(design).

Hard rules:
- **Science is frozen; only infrastructure changes.** Science means the pilot code and prompts, the splits, λ, the freeze manifest, the canonical runtime files, and the victim serving flags.
- **Ask before every `kubectl apply`/`delete` and every git push.**
- **No quantisation, no V100 fallback.** L40 only.
- **No tokens in chat.** Credentials stay in Kubernetes Secrets.
- **Never subset or drop questions,** and never hide failures.

Tests: `python -m unittest attack.dynamic_bridge_pilot.tests.test_pilot`. They need numpy and matplotlib; the NRP setup Job runs them in-pod.
