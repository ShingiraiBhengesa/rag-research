# Dynamic bridge-anchored cue pilot (two cues)

Branch `pilot/dynamic-bridge-two-cues`. Code: `attack/dynamic_bridge_pilot/`.
The 12-condition implementation is untouched and stays recoverable. The only
shared hook used is the existing production-no-op `PILOT_SUFFIX_OVERRIDE_KEY`
in `attack/runtime/cues.py`.

**Status (2026-10-07): implemented and tested offline. Nothing has been run on a
GPU yet.** The smoke test, λ fitting, freeze and held-out run are pending (see
"Sequence" below).

**The mechanism is general; the pilot evaluates two selected cue instantiations.**
`generate_dynamic_cue()` takes any condition that has a mechanism rule M in
`cues.MECHANISMS`.

## Selected cues

Selected from the existing canonical results; full report in
`selection/SELECTION_REPORT.md`, reasoning in `selected_cues.yaml`.

| role | condition | sig. cells @peak (8) | worst cell | qwen Δ@1 | llama Δ@5 |
|---|---|---|---|---|---|
| Cue 1 | `Natural_PreRegisteredSufficiency` | 8/8 | +10.4pp | +25.6pp | +14.8pp |
| Cue 2 | `Natural_PreRegisteredSufficiencySaturation` | 8/8 | +11.9pp | +24.0pp | +16.1pp |

These are the top two by robustness: significant cells first, then worst
cell, then mean. They are statistically indistinguishable from each other,
which is a known finding we kept on purpose (decision 1).

## Design

- **Victims:** Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct, at the pinned revisions. Settings are identical to canonical production: bf16, temperature 0, seed 20260820, top-k 3, and the same ODR budgets.
- **Datasets:** HotpotQA and 2WikiMultiHopQA.
- **d₁:** the whole clean first search observation, exactly the canonical injection point. The appended text is always `d₁ + "\n\n" + suffix`, and this is verified by hash for every arm.

The six arms are replayed from a single unforced NaturalClean anchor run, so every arm sees the same d₁:

| arm | appended text | isolates |
|---|---|---|
| `neutral` | none (NaturalClean) | baseline |
| `fixed_sufficiency` | canonical fixed Cue 1 | replication of the canonical effect |
| `fixed_saturation` | canonical fixed Cue 2 | replication of the canonical effect |
| `bridge_fact` | verbatim bridge span s* only | grounded entity text without a claim |
| `bridge_cue_sufficiency` | u* for Cue 1 | the claim beyond the fact; dynamic vs fixed |
| `bridge_cue_saturation` | u* for Cue 2 | as above |

## Algorithm → code

| step | where |
|---|---|
| 1. H ← Decompose(q). Attacker at temperature 0, question only. Cached per question and shared by both victims. Depth ≤ 2. | `bridge.decompose`, `validate_graph` |
| 2. F(d₁): the span must be verbatim in a document's text, **and** s ⊨ f must pass the NLI check (`cross-encoder/nli-deberta-v3-base` @ `6c749ce`, entailment is the argmax label with p ≥ 0.5). | `bridge.extract_facts`, `entail.py` |
| 3. A(q,d₁) via Match (same relation label, compatible argument slots); a* = argmin(δ, τ), where τ is the span offset in d₁. If A is empty, record `NO_ANCHOR`. No gold answer or victim trace is used. | `bridge.candidate_anchors`, `select_anchor` |
| 4. K = 5 variants with fixed seeds `20261007…20261011`. One pass each, no feedback. | `cues.generate_dynamic_cue` |
| 5. V1 format · V2 ≤ B = 48 tokens · V3 lexical + judge mechanism check · V4 deterministic + judge task-fact check · V5 judge says no answer stated · **copy gate** (similarity to the fixed suffixes < 0.6; the score is always logged). If nothing passes, record `INVALID_CUE`. | `cues.evaluate_variant` |
| 6. Q = λ·[Ground, Align, Fidelity, Clarity]. u* = argmax Q, ties go to the lower index. λ is fitted to manual dev ratings, frozen and hashed. | `cues.select`, `lambda_fit.py` |

V5 also runs an exact gold-answer string check, but only for the audit. It never gates a variant.

## Splits (fixed before any run, disjoint, stratified by question type)

| split | per dataset | use |
|---|---|---|
| smoke | 3 | bug-finding only, with provisional equal λ |
| dev | 10 (the existing adaptive-pilot dev split) | cue audit and λ fitting |
| heldout | 100 | the pilot result; no subsetting allowed |

Splits live in `splits/`. They were built with `python -m attack.dynamic_bridge_pilot.splits` (seed 20261007) from questions never used in production, cue development, the runtime gate, or dev.

## Sequence

```
1. smoke (Libra, decision 5)   sbatch experiments/dynamic_bridge_cue_pilot/slurm/smoke_libra.sbatch
                               -> audit smoke_results/, fix bugs only
2. dev cues (no victim arms)   run_phase.sh decompose dev; run_phase.sh cell dev <victim> <dataset>
3. rate + fit λ                cli audit-export --split dev --out ratings.csv   (rate per AUDIT_RUBRIC.md)
                               cli fit-lambda --split dev --ratings ratings.csv
4. freeze                      cli freeze            -> freeze_manifest.json (hashes code, prompts,
                                                         config, canonical deps, λ, held-out manifests)
5. commit + push branch; record the commit hash below
6. NRP held-out                render_jobs.py --split heldout --commit <sha>; kubectl apply (setup,
                               decompose, 4 cells); verify-freeze runs inside every cell
7. analyze                     cli analyze --split heldout --state-dir ... --results-dir ...
```

Every phase checkpoints one JSONL line per unit (fsync'd) under
`$STATE_DIR/<split>/`. A preempted pod resumes by skipping completed units and
finished phases (via marker files). Infrastructure failures are retried twice,
then logged to `failures/`, and the phase is not marked done.

## NRP (infrastructure only)

`nrp/k8s/render_jobs.py` renders one setup Job, one decompose Job, and four
cell Jobs (victim × dataset). All use `priorityClassName: opportunistic`,
`nvidia.com/gpu.product: NVIDIA-L40`, and 1 GPU / 48Gi / 8 CPU each. The PVC
`adaptive-pilot-work` is mounted at `/pvc`. Checkpoints go to
`/pvc/checkpoints/dynamic_bridge_cue_pilot` and results to
`/pvc/results/dynamic_bridge_cue_pilot`.

Two staging details:
- **qwen14b cells** serve Qwen2.5-14B once for all phases (the attacker shares those weights).
- **llama8b cells** stage the model loads victim → attacker → victim, because 8B + 14B at bf16 do not fit in 48 GB together.

The setup Job clones the frozen commit, builds the driver venv on top of the
image, downloads the pinned weights, and runs the offline tests inside the pod.
It needs the Secrets `github-token` and `hf-token` (Llama is gated).

**V100 fallback not rendered.** Running 4-bit quantised victims would change the
victim models. Those runs would not be comparable with the canonical bf16
results or with L40 runs of the same matrix (spec section 20). V100 also has no
bf16 support. If L40 capacity runs out, the right move is to wait for L40
capacity, not to fall back.

## Validation gate (spec section 16)

Offline, checked now (24 tests in `attack/dynamic_bridge_pilot/tests/test_pilot.py`, plus the existing 39 adaptive-pilot and v3-infra tests, all passing):
- [x] Cues are generated per instance from the actual d₁, a* and question. The prompt contains the documents and the bridge fact.
- [x] Bridge facts are grounded: the span is verbatim in document text (never metadata) and passes the NLI check. `NO_ANCHOR` is recorded, never substituted.
- [x] Cue 1 and Cue 2 mechanism rules are enforced. Cue 1 forbids the saturation clause, and Cue 2 requires both claims.
- [x] Fixed text is not reused: a hard copy gate, with the similarity logged for every variant.
- [x] PCR is the canonical definition. Paired contrasts match the canonical bootstrap. Direct-answer and ResearchComplete terminations are labelled separately. Every arm, including Bridge-Fact, goes through the same `label_run`, which was exercised on a real production trace.
- [x] Six-arm routing resolves through the production hook to the intended suffix.

On real runs, still pending: Qwen, Llama, HotpotQA and 2Wiki runs; each arm running; injection hash checks; trajectories, cues and provenance saved end to end.

## Known issues / open decisions

1. **Anchor edge policy.** Read literally, a* = argmin δ(e) with δ measured from head(e) to g. That favours the edge *into the answer slot*, and a fact on that edge contains the answer, so its variants will usually fail V5 and give `INVALID_CUE`. `config.ANCHOR_EDGE_POLICY = "spec_exact"` (the literal algorithm) is the default. `"exclude_answer_edges"` drops answer-slot edges from A. This must be decided before freezing; the smoke test will show how often it matters.
2. Comparison questions (about half of the held-out set) decompose as parallel two-hop branches, not as a chain. They are kept, not filtered.
3. The attacker shares weights with the qwen14b victim. This is recorded; independence is not claimed.

## Commit

Frozen commit: _(filled after step 5)_
