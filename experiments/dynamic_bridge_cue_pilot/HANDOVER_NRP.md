# Handover: running the dynamic bridge-cue pilot on NRP

For the Claude Code agent operating NRP from the user's Mac. Read this whole
file, then `README.md` in this folder (the design), before running anything.
Repo: `ShingiraiBhengesa/rag-research`, branch `pilot/dynamic-bridge-two-cues`.

## 0. Rules (non-negotiable)

1. **The science is fixed; only infrastructure may change.** Science means:
   - everything in `attack/dynamic_bridge_pilot/` (code and prompts);
   - `splits/`, `lambda/`, `freeze_manifest.json`;
   - the canonical runtime files listed in `freeze.py` (`CANONICAL_DEPENDENCIES`);
   - the victim serving flags in `nrp/run_phase.sh`: bf16, `--max-model-len 32768`, the tool parsers, `--generation-config vllm`, `--enforce-eager`.

   Infrastructure means Job YAML, resources, shard count, retries, paths, and the image, but only for another vLLM 0.23.0 build.
2. **Ask the user before every `kubectl apply`, `kubectl delete`, and git push.** Read-only `kubectl get/describe/logs/exec` is fine.
3. **Never quantise and never use the V100 fallback.** It changes the victim models and breaks comparability. L40 nodes only (`nvidia.com/gpu.product: NVIDIA-L40`).
4. **Never ask the user to paste a token into chat.** Credentials live only in Kubernetes Secrets.
5. **Never subset, re-pick, or drop questions,** and never delete failed results to make numbers look better. Failures are recorded and reported.
6. **If you find a bug in the science after the freeze, stop and tell the user.** The fix needs a new freeze and a rerun of the affected splits, and gets documented under "Known issues" in `README.md`.
7. **Cue ratings are the user's.** Do not fill in `rating` yourself unless the user explicitly asks, and never show the user victim outcomes for dev before they finish rating.

## 1. State at handover (2026-10-07)

- **Code:** branch head `da087ac3` or later. Implemented and tested offline: 26 pilot tests plus 39 existing tests, and an end-to-end analysis on real canonical traces.
- **Nothing has run on a GPU yet.**
- **Design:** 2 victims (qwen14b, llama8b) × 2 datasets (hotpot, 2wiki) × 6 arms. The arms are `neutral`, `fixed_sufficiency`, `fixed_saturation`, `bridge_fact`, `bridge_cue_sufficiency` and `bridge_cue_saturation`; details in `README.md`.
- **Sequence:**
  1. **dev** (10 questions per dataset): cue generation.
  2. The user rates cues; fit λ.
  3. Decide the anchor-edge policy.
  4. **Freeze**, commit and push.
  5. **dev** six arms at the frozen commit; review.
  6. **full** (the canonical 450 questions per dataset, sharded).
- **Open decision:** `config.ANCHOR_EDGE_POLICY`, either `spec_exact` (the current default) or `exclude_answer_edges`. It must be settled before the freeze (step 4).
- **Security:** the user has been told to rotate a GitHub token exposed on Libra. Make sure the `github-token` Secret holds the new one.

## 2. Environment facts

| item | value |
|---|---|
| namespace | `trace-ai` (verify; it must be where the PVC lives) |
| PVC | `adaptive-pilot-work`, RWX, mounted at `/pvc` |
| priority class | `opportunistic`; pods can be preempted, and every phase checkpoints per question and resumes |
| GPU | 1× NVIDIA-L40 (48 GB) per pod; 48Gi RAM, 8 CPU |
| image | `vllm/vllm-openai:v0.23.0` (untested on NRP; the setup Job runs the offline tests in-pod) |
| Secrets | `github-token` (key `token`) and `hf-token` (key `token`, with access to Llama-3.1) |
| code in pods | `/pvc/code/<commit>` (cloned by the setup Job); driver venv `/pvc/envs/odr-<commit>` |
| model cache | `/pvc/hf` |
| checkpoints | `/pvc/checkpoints/dynamic_bridge_cue_pilot/<split>/…` (phase JSONLs, traces, markers, `failures/`, vLLM logs in `logs/`) |
| results | `/pvc/results/dynamic_bridge_cue_pilot/<split>/…` |

`nrp/k8s/render_jobs.py --split S --commit <40-char sha> [--shards N]` writes these files into `nrp/k8s/rendered/S/`:
- `00-setup.yaml`: once per commit. It clones, builds the venv, downloads weights and runs the tests, and must log `SETUP_OK`.
- `01-decompose.yaml`: once per split.
- `02-cell-<victim>-<dataset>.yaml`: four Jobs, each an Indexed Job of N pods when sharded.
- `03-toolbox.yaml`: a GPU-less pod for `exec`/`cp`, `audit-export` and `analyze`.

Job names end in the commit's first 8 characters. Re-applying an existing Job fails, so delete it first; checkpoints on the PVC keep the progress.

## 3. Runbook

### Step 0: preflight (read-only)

```bash
kubectl config current-context
kubectl -n trace-ai get pvc adaptive-pilot-work
kubectl get priorityclass opportunistic
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-L40 -o wide | head
kubectl -n trace-ai get secret github-token hf-token     # if missing: tell the user to create them (below)
kubectl -n trace-ai get resourcequota                    # GPU quota limits the shard count
git rev-parse HEAD && git status --short                 # must be clean and pushed
```

Secrets are created by the user in their own terminal; never in chat:
`kubectl -n trace-ai create secret generic github-token --from-literal=token=…`
(and the same for `hf-token`).

### Step 1: dev cue generation (pre-freeze commit)

```bash
SHA=$(git rev-parse HEAD)
python3 experiments/dynamic_bridge_cue_pilot/nrp/k8s/render_jobs.py --split dev --commit $SHA
R=experiments/dynamic_bridge_cue_pilot/nrp/k8s/rendered/dev
kubectl apply -f $R/00-setup.yaml      # ask first; wait for SETUP_OK in its log
kubectl apply -f $R/01-decompose.yaml  # ask first
kubectl apply -f $R/02-cell-qwen14b-hotpot.yaml -f $R/02-cell-qwen14b-2wiki.yaml \
              -f $R/02-cell-llama8b-hotpot.yaml -f $R/02-cell-llama8b-2wiki.yaml   # ask first
```

There is no `freeze_manifest.json` in this commit yet, so the cells run the anchor runs and cue generation, then stop. Each logs `[done] dev <cell> cues only (lambda not frozen yet)`.

### Step 2: export cues for rating, and anchor-edge statistics

```bash
kubectl apply -f $R/03-toolbox.yaml          # ask first
T=dbcp-toolbox-${SHA:0:8}
kubectl -n trace-ai exec $T -- bash -lc 'cd $REPO && $PY_ODR -B -m attack.dynamic_bridge_pilot.cli audit-export \
   --split dev --state-dir $STATE_DIR --out /pvc/results/dynamic_bridge_cue_pilot/dev_ratings_blank.csv'
kubectl -n trace-ai cp $T:/pvc/results/dynamic_bridge_cue_pilot/dev_ratings_blank.csv ./dev_ratings.csv
```

Give `dev_ratings.csv` and `experiments/dynamic_bridge_cue_pilot/AUDIT_RUBRIC.md` to the user to rate (1–5 per row, plus initials in the `rater` column).

For the anchor-edge decision, summarise the following for the user. These are statistics only, no victim outcomes:
- per cell, the bridge status counts;
- how often the chosen anchor has `edge_tail_is_answer_slot = true`;
- the `INVALID_CUE` rate, split by that flag;
- the most common failed checks (V1–V5, COPY).

```bash
kubectl -n trace-ai exec $T -- bash -lc 'cd $STATE_DIR/dev && python3 - <<EOF
import json,glob,collections
for f in sorted(glob.glob("bridge_selection/*.jsonl")):
    rs=[json.loads(l) for l in open(f) if l.strip()]
    st=collections.Counter(r["status"] for r in rs)
    ans=sum(1 for r in rs if (r.get("anchor") or {}).get("edge_tail_is_answer_slot"))
    print(f, dict(st), "anchor_on_answer_edge:", ans)
for f in sorted(glob.glob("generated_cues/*.jsonl")):
    rs=[json.loads(l) for l in open(f) if l.strip()]
    print(f, collections.Counter((r["condition"][-10:], r["status"]) for r in rs))
    fails=collections.Counter(k for r in rs for v in r.get("variants",[]) for k,ok in v["checks"].items() if not ok)
    print("   failed checks:", dict(fails))
EOF'
```

If the user switches the policy to `exclude_answer_edges`:
1. Edit `attack/dynamic_bridge_pilot/config.py`, then commit and push (with the user's approval).
2. Run setup for the new commit.
3. On the PVC, delete only `dev/bridge_selection/`, `dev/generated_cues/`, `dev/markers/cues_*`, and the `dev/failures/{bridge,cues}_*` files. Anchors and decompositions stay valid, because the policy affects only anchor selection.
4. Re-run the dev cells, then return to step 2.

### Step 3: fit λ and freeze (on the Mac, plain python3 ≥ 3.10, from the repo root)

```bash
mkdir -p experiments/dynamic_bridge_cue_pilot/state
kubectl -n trace-ai cp $T:/pvc/checkpoints/dynamic_bridge_cue_pilot/dev/generated_cues \
   experiments/dynamic_bridge_cue_pilot/state/dev/generated_cues
mkdir -p experiments/dynamic_bridge_cue_pilot/lambda
cp dev_ratings.csv experiments/dynamic_bridge_cue_pilot/lambda/dev_ratings.csv        # the user's rated file
PYTHONPATH=. python3 -m attack.dynamic_bridge_pilot.cli fit-lambda --split dev \
   --ratings experiments/dynamic_bridge_cue_pilot/lambda/dev_ratings.csv
PYTHONPATH=. python3 -m attack.dynamic_bridge_pilot.cli freeze
PYTHONPATH=. python3 -m attack.dynamic_bridge_pilot.cli verify-freeze      # must print ok: true
git add experiments/dynamic_bridge_cue_pilot/lambda experiments/dynamic_bridge_cue_pilot/freeze_manifest.json
git commit -m "Freeze dynamic bridge pilot: lambda from dev ratings"     # ask first; push after approval
```

Tell the user the fitted λ, its pairwise concordance, and the equal-weights baseline (all in `lambda/lambda_frozen.json`). The `state/` folder is gitignored, so don't commit it.

From here on `FROZEN=$(git rev-parse HEAD)`, and nothing in rule 1's "science" list may change.

### Step 4: dev arms at the frozen commit

```bash
python3 experiments/dynamic_bridge_cue_pilot/nrp/k8s/render_jobs.py --split dev --commit $FROZEN
R=experiments/dynamic_bridge_cue_pilot/nrp/k8s/rendered/dev
kubectl apply -f $R/00-setup.yaml      # new commit -> new setup; wait for SETUP_OK
kubectl apply -f $R/02-cell-*.yaml     # no decompose needed (marker exists); cells skip anchor/cues and run select + arms
```

Then analyze in a toolbox pod at `$FROZEN`, and copy the results out:

```bash
kubectl -n trace-ai exec dbcp-toolbox-${FROZEN:0:8} -- bash -lc 'cd $REPO && $PY_ODR -B -m attack.dynamic_bridge_pilot.cli analyze \
   --split dev --state-dir $STATE_DIR --results-dir $RESULTS_DIR'
kubectl -n trace-ai cp dbcp-toolbox-${FROZEN:0:8}:/pvc/results/dynamic_bridge_cue_pilot/dev ./results_dev
```

Go/no-go review with the user before full:
- `diagnostics/bridge_and_cue_coverage.csv`: `complete=True` for all 4 cells and `injection_failures=0`.
- In the anchors, `clean_observation_matches_anchor` should be true; report every exception.
- Bridge `NO_ANCHOR`/`INVALID_CUE` rates, and `generated_cues/cue_audit.csv` spot-read for whether the cues do what they're meant to.
- `summaries/PILOT_SUMMARY.md` (PCR, accuracy, evidence) as a sanity check only; n is 10 per cell.
- Real L40 runtime per arm run (`runtime_seconds` in `$STATE_DIR/dev/arms/*.jsonl`). Use it to choose `--shards`.
- `failures/` should be empty, or every entry explained.

### Step 5: full run (canonical n450, sharded)

```bash
python3 experiments/dynamic_bridge_cue_pilot/nrp/k8s/render_jobs.py --split full --commit $FROZEN --shards 8
R=experiments/dynamic_bridge_cue_pilot/nrp/k8s/rendered/full
kubectl apply -f $R/01-decompose.yaml   # setup for $FROZEN is already done
kubectl apply -f $R/02-cell-*.yaml      # 4 Indexed Jobs x 8 pods; they wait for the decompose marker
```

Every full-split phase runs `verify-freeze` first and refuses to start on any mismatch. Pick the shard count from the dev timings and the GPU quota: 8 gives about 9–10 h for qwen, 12 gives about 6–7 h. Results are identical whatever the shard count.

Then analyze `--split full` in the toolbox pod as in step 4, and copy the results out. The analysis also writes `diagnostics/replication_vs_canonical.csv`. Report:
- the completeness table;
- the PCR contrasts at @1/@2/@5/@30, on both `all_paired` and `all_six_arms_ran`;
- answer accuracy and paired flips;
- evidence loss;
- the replication agreement;
- every failure.

## 4. Monitoring and troubleshooting

```bash
kubectl -n trace-ai get jobs,pods -l app=dynamic-bridge-cue-pilot -o wide
kubectl -n trace-ai logs <pod> --tail=50            # phase progress is JSON lines: {"phase": "...", "key": ...}
kubectl -n trace-ai describe pod <pod>              # Pending -> scheduling/quota; OOMKilled -> memory
kubectl -n trace-ai exec <toolbox> -- bash -lc 'wc -l $STATE_DIR/full/arms/*.jsonl | tail -1'      # progress
kubectl -n trace-ai exec <toolbox> -- bash -lc 'ls $STATE_DIR/full/markers; cat $STATE_DIR/full/failures/*.jsonl 2>/dev/null | tail'
```

- **Preemption or eviction:** the Job recreates the pod, which skips finished phases (markers) and finished units (JSONL keys). Nothing to do.
- **`[phase] … had infrastructure failures; not marking done`:** the pod exits non-zero and restarts, and the restart retries only the failed units. If the same unit keeps failing, read `failures/*.jsonl` and the vLLM log in `$STATE_DIR/logs/`.
- **vLLM won't start:** check its log. If it's a KV-cache or memory error, raising `GPU_MEM_UTIL` (default 0.90) is allowed. Lowering `--max-model-len` or `--dtype` is not.
- **Setup Job fails at the tests or pip:** this is an environment problem. Fix the infrastructure (image, pip pins in the setup script); never edit the science to make tests pass.
- **A Job hits `backoffLimit`:** find the cause, fix the infrastructure, delete the Job, and re-apply it (with approval). Progress is kept.
- **Pods Pending:** usually the GPU quota or no free L40s. Lower `--shards` (re-render), or wait.

## 5. Final report to the user

Include:
- what ran;
- the commit hashes (pre-freeze, frozen);
- the shard count;
- wall time;
- the completeness table;
- every failure or exclusion with its reason;
- the headline tables from `PILOT_SUMMARY.md`;
- where the copied results live on the Mac.

Record the frozen commit hash in the "Commit" section of `README.md`, and commit it with the user's approval.
