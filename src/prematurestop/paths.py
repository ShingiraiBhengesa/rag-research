"""Every filesystem location the project uses, defined once.

Two kinds of location:

* **Tracked** (in the repository): experiment configs and published results.
* **Local data** (gitignored, too large or not ours to redistribute): raw victim
  traces, per-run records, dataset manifests/corpora/evidence, vLLM logs. It
  lives under ``LOCAL_DATA``, which defaults to ``<repo>/local_data`` and can be
  pointed elsewhere with the ``PREMATURESTOP_LOCAL_DATA`` environment variable
  (on the Libra cluster it is the original ``rag-research/results`` folder).

Threat models are separated at the experiment/results level
(``whitebox/`` now, ``blackbox/`` later); the code itself is shared.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- third-party victim agent (vendored, with documented patches) ----------------------------
THIRD_PARTY_ODR = REPO_ROOT / "third_party" / "open_deep_research"
ODR_SRC = THIRD_PARTY_ODR / "src"

# --- experiments (configs, freeze manifests, launch scripts) -----------------------------------
EXPERIMENTS = REPO_ROOT / "experiments"
EXPERIMENT_E1 = EXPERIMENTS / "whitebox" / "e1_fixed_cue_matrix"
EXPERIMENT_E2 = EXPERIMENTS / "whitebox" / "e2_bridge_mechanism"

# --- published results (small, tracked) ---------------------------------------------------------
RESULTS = REPO_ROOT / "results"
RESULTS_E1 = RESULTS / "whitebox" / "e1_fixed_cue_matrix"
RESULTS_E1_QWEN_LLAMA = RESULTS_E1 / "qwen_llama"
RESULTS_E1_GEMMA_GPTOSS = RESULTS_E1 / "gemma_gptoss"
RESULTS_E1_CANONICAL = RESULTS_E1 / "canonical"
RESULTS_E2 = RESULTS / "whitebox" / "e2_bridge_mechanism"

# --- local, gitignored raw data -----------------------------------------------------------------
LOCAL_DATA = Path(os.environ.get("PREMATURESTOP_LOCAL_DATA", REPO_ROOT / "local_data")).resolve()
LOCAL_MANIFESTS = LOCAL_DATA / "manifests"
LOCAL_CORPUS = LOCAL_DATA / "corpus"
LOCAL_EVIDENCE = LOCAL_DATA / "evidence"
LOCAL_DEV_BUNDLE = LOCAL_DATA / "nonproduction_validation" / "dev_bundle"
LOCAL_ADAPTIVE_SPLITS = LOCAL_DATA / "adaptive_pilot" / "splits"
LOCAL_CACHE = LOCAL_DATA / "cache"
