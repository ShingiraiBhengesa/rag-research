# adaptive-rag/attack — Repository Map

Last updated: April 30, 2026

This directory contains all adversarial attack experiments against the deployed
Adaptive-RAG system. Multiple iterations exist — see "Status" annotations to
identify what is current vs deprecated.

---

## Attack Scripts

### Active

| File | Purpose | Threat model | Status |
|------|---------|--------------|--------|
| attack_blackbox_v3.py | Single-word synonym swap on deployed system | Decision-only BB + side-channel | Active — produced 48.6% strict success |
| attack_whitebox_v1.py | TextFooler-style on T5 router (importance ranking + WordNet) | Score-based (uses @torch.no_grad()) | Active — to be run by SLURM job 110191 |
| attack_complexity_together_v2.py | Llama-3.3-70B rephrasing + 5 strict filters, classifier verification | Score-based on T5 router | Active (latest) — produced 16.82% strict-clean flip rate |
| semantic_filter.py | 5 strict filters: entity, NLI, qform, qualifier, categorical-descriptor | Library | Active — used by complexity attack |
| route_detector.py | Decision tree predicting route from latency + num_sources | Side-channel reconstruction | Active — 99.8% CV accuracy, used to recover mechanism breakdown |
| analyze_phase1.py | Aggregate analysis across phase 1/2/3 BB results | Analysis | Active |

### Superseded (kept for traceability, do not run)

| File | Why superseded |
|------|----------------|
| attack_blackbox.py | Replaced by attack_blackbox_v3.py |
| attack_complexity_groq.py | Replaced by _groq_v2.py, then by _together_v2.py |
| attack_complexity_groq_v2.py | Replaced by Together version when Groq rate-limited |
| attack_complexity_whitebox.py | Early prototype, superseded by _together_v2.py |
| run_textattack.py | Early TextAttack-based prototype |
| run_textattack_pilot.py | Smoke test of above |

### To-Be-Built (planned)

| File | Purpose | Threat model |
|------|---------|--------------|
| attack_singleword_gradient.py | True white-box (HotFlip-style) on T5 router | Gradient-based WB |
| attack_complexity_gradient.py | True white-box complexity attack | Gradient-based WB |
| attack_complexity_blackbox.py | Complexity attack against deployed system | Decision-only BB |

---

## Diagnostic & Utility Scripts

| File | Purpose |
|------|---------|
| diagnose_router.py | Router diagnostics: logit gaps, importance distributions, route confidence |
| reevaluate_with_strict_filter.py | Apply current strict filters retroactively to old results |
| inspect_complexity_flips.py | Manual flip-quality tagger (used briefly for v1.4 paper) |
| auto_tag_flips.py | LLM-as-judge auto-tagger (Groq) — blocked by daily TPD, deprecated |
| run_pilot_baseline.py | Baseline measurement on subset before attacking |

---

## Shell Scripts (SLURM/launcher)

| File | What it runs |
|------|--------------|
| run_phase3_after_phase2.sh | Auto-launches phase 3 BB attack on hardest victims after phase 2 finishes |
| run_textattack.sh, run_textattack_pilot.sh | Early TextAttack prototype runs (deprecated) |
| setup_env.sh | Conda activation, env vars |
| test_wrapper.sh | One-off test |

---

## Results Directory Structure

results/
  final/
    baseline_musique_route_A.jsonl     # baseline routing predictions
    baseline_musique_route_B.jsonl
    baseline_musique_route_C.jsonl
    chains_archive/                    # earlier attack-chain experiments
    route_detector_report.json         # decision tree training results (99.8% CV)

  attacks/
    whitebox/
      victims_107.jsonl                # 107 baseline-correct MuSiQue victims (the test set)

    phase1/                            # Single-word BB phase 1 (initial pass)
      phase1_A.jsonl                   # Route A victims
      phase1_B.jsonl                   # Route B victims
      phase1_C.jsonl                   # Route C victims
      phase1_analysis.json

    phase2/                            # Single-word BB phase 2 (extended budget for failures)
      phase2_C.jsonl

    phase3/                            # Single-word BB phase 3 (hardest failures, budget=200)
      phase3.jsonl
      phase3_targets.jsonl
      phase3.log

    whitebox_complexity_together_v5_full/   # *** Latest complexity WB result (16.82%) ***
      attack_v2_n107.jsonl
      summary_n107.json

    whitebox_complexity_together_v2..v5/    # Tuning iterations (deprecated)
    whitebox_complexity_groq/               # Pre-Together iterations (deprecated)
    whitebox_complexity, whitebox_complexity_v2/    # Earliest prototypes (deprecated)

    smoke_*.jsonl                      # Various smoke tests (deprecated)

  diagnostics/
    router_diagnostics_summary.json    # Logit gap stats, importance distributions

---

## Key Result Files (cite in paper)

| Result | Lives in |
|--------|----------|
| Single-word BB strict success: 52/107 = 48.6% | results/attacks/phase1/*.jsonl + phase2/phase2_C.jsonl (deduped) |
| Single-word BB mechanism: 5 route-flip / 47 in-route | Above + route_detector.py post-hoc reconstruction |
| Complexity WB strict-clean flip rate: 18/107 = 16.82% | results/attacks/whitebox_complexity_together_v5_full/summary_n107.json |
| Router diagnostics (logit gaps, importance) | results/diagnostics/router_diagnostics_summary.json |
| Route detector accuracy: 99.8% CV | results/final/route_detector_report.json |

---

## Cleanup TODO (after gradient experiments + before paper writing)

1. Delete deprecated attack scripts (move to archive/ if uncomfortable deleting)
2. Consolidate complexity result dirs — keep only _v5_full, archive others
3. Delete smoke result files (smoke_*.jsonl)
4. Move shell-script prototypes to archive/
5. Add a requirements.txt documenting environment (currently captured by setup_env.sh)
6. Add a top-level Makefile or run_all.sh showing canonical commands for each experiment

Rule for cleanup: anything not cited in the paper Methods section is a deletion candidate.

---

## Naming Convention (going forward)

For new experiments, use:

  results/wb_router/<attack>_<threat-model>_n<N>.{jsonl,json}
  results/bb_system/<attack>_n<N>.{jsonl,json}

Examples:
  results/wb_router/single_word_gradient_n107.jsonl
  results/wb_router/complexity_score_based_n107.jsonl
  results/bb_system/complexity_n107.jsonl

This avoids the v2/v3/v4 proliferation we accumulated during iteration.
