# prematurestop

PhD research on premature-completion attacks against agentic RAG (Open Deep
Research). See `README.md` for the layout. In short:
* code is in `src/prematurestop/`;
* the victim agent is vendored in `third_party/open_deep_research/` (patches
  listed in `PATCHES.md` there);
* experiments are in `experiments/whitebox/{e1_fixed_cue_matrix,e2_bridge_mechanism}`;
* published results are in `results/whitebox/`.

Set `PYTHONPATH="$PWD/src:$PWD/third_party/open_deep_research/src"`, or
`pip install -e .`.

## Active work: E2 bridge mechanism

Operating it? Read `experiments/whitebox/e2_bridge_mechanism/HANDOVER_NRP.md`
first (rules + runbook), then `experiments/whitebox/e2_bridge_mechanism/README.md`
(design). Runtime state and logs go to `$PREMATURESTOP_LOCAL_DATA` (default `local_data/`, gitignored).

Hard rules:
- **Science is frozen; only infrastructure changes.** Science means the pilot code and prompts, the splits, λ, the freeze manifest, the canonical runtime files, and the victim serving flags.
- **Ask before every `kubectl apply`/`delete`, every `sbatch`, and every git push.**
- **No quantisation, no V100 fallback.**
- **No tokens in chat.** Credentials stay in Kubernetes Secrets / env files.
- **Never subset or drop questions,** and never hide failures.
- **All commits are authored by the repo owner;** no co-author trailers.

Tests: `python -m unittest discover -s tests -t .` (E2 only: `python -m unittest tests.pipelines.test_bridge_mechanism`). They need numpy and matplotlib.
