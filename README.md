 Overview

Retrieval-Augmented Generation (RAG) systems enhance language models by selectively retrieving external documents to ground their outputs in factual evidence. However, the mechanisms that govern *when* and *how* these systems retrieve information remain poorly understood, particularly under adversarial conditions.

This project conducts a systematic adversarial analysis of retrieval decision-making in two architecturally distinct RAG systems:

- **Self-RAG** (Asai et al., ICLR 2024 Oral) — an *internalized* approach where a single LLaMA 2 model generates special reflection tokens to decide when to retrieve and how to evaluate retrieved content
- **Adaptive-RAG** (Jeong et al., NAACL 2024) — a *modular* approach where an external FLAN-T5 classifier predicts query complexity upfront, routing queries to no-retrieval, single-step retrieval, or multi-step iterative retrieval strategies

These two systems represent opposite ends of the RAG design spectrum: fully integrated vs. fully decoupled retrieval decision-making. By establishing baselines and then probing each system with adversarial attacks, we aim to expose fundamental vulnerabilities in how RAG systems decide when and how much to retrieve.

---

## Research Questions

1. **How effective are retrieval decisions in current self-reflective RAG systems?** Do Self-RAG and Adaptive-RAG make genuinely adaptive retrieval decisions, or do they default to fixed behaviors?
2. **What are the adversarial attack surfaces of integrated vs. modular retrieval decisions?** Can we manipulate Self-RAG's internal reflection tokens? Can we fool Adaptive-RAG's external classifier?
3. **Which architectural paradigm is more robust?** Does internalizing the retrieval decision (Self-RAG) make it harder to attack, or does the tight coupling create new vulnerabilities?

---

## Key Findings (Baselines)

### Self-RAG: The retrieval decision is barely adaptive

| Dataset | Adaptive | Always Retrieve | No Retrieval | Retrieval % |
|---------|----------|-----------------|--------------|-------------|
| PopQA | 54.9% | 54.9% | 28.7% | 100.0% |
| TriviaQA | 66.2% | 66.2% | 50.3% | 100.0% |
| PubHealth | 72.1% | 71.7% | 70.1% | 94.4% |
| ARC-Challenge | 67.2% | 67.2% | 67.4% | 100.0% |

- Self-RAG retrieves on **100% of queries** for 3 out of 4 datasets. The `[Retrieve]` token almost never says "no."
- Only PubHealth shows genuine adaptive behavior (94.4% retrieval), and on this dataset adaptive mode slightly outperforms always-retrieve, proving the mechanism works when engaged.
- On ARC-Challenge, retrieval actually **hurts** accuracy by 0.3 percentage points, yet the model retrieves every single time — a clear failure in the retrieval decision.

### Adaptive-RAG: The classifier under-routes to the correct strategy

| Dataset | Adaptive | No Retrieval (A) | Single-Step (B) | Multi-Step (C) |
|---------|----------|-------------------|-----------------|----------------|
| MuSiQue | 21.4% | 4.0% | 15.8% | 19.2% |
| HotpotQA | 45.6% | 18.4% | 39.6% | 49.4% |
| 2WikiMultiHopQA | 51.2% | 25.6% | 46.2% | 62.6% |

- The classifier achieves only **50.0% validation accuracy** at best (epoch 30), with particularly poor performance on route A (no-retrieval) at 28.3%.
- On MuSiQue, adaptive routing (21.4%) **beats** all individual strategies, demonstrating the classifier can add value when it routes correctly.
- On HotpotQA and 2WikiMultiHopQA, adaptive routing **underperforms** the always-multi-step strategy by 3.8pp and 11.4pp respectively, indicating systematic mis-routing of complex queries.

### The core insight

Both systems fail at retrieval decisions, but in **opposite directions**:
- **Self-RAG over-retrieves**: wastes computation and sometimes hurts accuracy by retrieving when it shouldn't.
- **Adaptive-RAG under-routes**: the classifier isn't accurate enough to route queries to the optimal strategy, particularly failing to identify when complex multi-step retrieval is needed.

---

## Architecture Comparison

### Self-RAG (Internalized Reflection)

```
Query → LLaMA 2 (7B) → [Retrieve] token → {Yes: Retrieve + Generate + [IsRel] + [IsSup] + [IsUse]}
                                            {No:  Generate directly}
```

- **Model**: `selfrag/selfrag_llama2_7b` (13GB, fine-tuned LLaMA 2-7B)
- **Decision mechanism**: Special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) generated as part of the output vocabulary
- **Granularity**: Segment-level (can retrieve multiple times during generation)
- **Training**: Supervised fine-tuning with reflection tokens inserted by a GPT-4-trained critic model

### Adaptive-RAG (External Classifier)

```
Query → T5-Large Classifier → {A: No retrieval (direct LLM)}
                                {B: Single-step retrieval}
                                {C: Multi-step iterative retrieval (IRCoT)}
```

- **Classifier**: T5-Large (~770M params), trained on silver labels from FLAN-T5-XXL predictions
- **Generator**: FLAN-T5-XXL (used for pre-computed predictions)
- **Decision mechanism**: Upfront query complexity classification before any generation
- **Granularity**: Query-level (one routing decision per query, no fallback)

---

## Repository Structure

```
rag-research/
├── README.md
├── self-rag/                          # Self-RAG codebase (from AkariAsai/self-rag)
│   └── retrieval_lm/
│       ├── run_short_form.py          # Main inference script
│       └── eval_data/                 # Evaluation datasets (PopQA, TriviaQA, PubHealth, ARC-C)
├── adaptive-rag/                      # Adaptive-RAG codebase (from starsuzi/Adaptive-RAG)
│   └── repos/Adaptive-RAG/
│       ├── classifier/                # T5-Large complexity classifier
│       │   ├── run_classifier.py      # Training script
│       │   ├── data/                  # Silver training labels
│       │   └── outputs/               # Trained models (epochs 15-35)
│       ├── predictions/               # Pre-computed LLM predictions (nor_qa, oner_qa, ircot_qa)
│       ├── processed_data/            # Evaluation datasets (MuSiQue, HotpotQA, 2WikiMHop)
│       └── evaluate_final_acc.py      # Official evaluation script
├── results/
│   ├── selfrag/
│   │   ├── popqa_adaptive.json        # Raw results per dataset/mode
│   │   ├── triviaqa_adaptive.json
│   │   ├── pubhealth_adaptive.json
│   │   ├── arc_adaptive.json
│   │   ├── (+ always_retrieve and no_retrieval variants)
│   │   └── figures/                   # 5 matplotlib figures + 3 CSV tables
│   ├── adaptiverag/
│   │   ├── adaptive_rag_results.json  # Summary results
│   │   └── figures/                   # 5 matplotlib figures
│   └── combined/
│       └── figures/                   # 2 cross-system comparison figures
└── scripts/                           # Slurm job scripts
```

---

## Infrastructure & Setup

### SLU Libra HPC Cluster

| Resource | Specification |
|----------|--------------|
| GPU Nodes | 3x Dell R760xa, 2x NVIDIA L40s (40GB VRAM each) |
| ML Nodes | 2x Dell R760xa, 4x NVIDIA H100 NVL (80GB VRAM each) |
| Scheduler | Slurm |
| OS | RedHat Enterprise Linux 9 |
| Storage | 244TB shared home filesystem |

### Environment: Self-RAG

```bash
module load anaconda/3
conda create -n selfrag python=3.10
conda activate selfrag
pip install vllm==0.4.2 torch==2.3.0 transformers==4.40.0
export LD_LIBRARY_PATH=$HOME/.conda/envs/selfrag/lib:$LD_LIBRARY_PATH
```

**Known fixes applied:**
- `LD_LIBRARY_PATH` override for libstdc++ compatibility
- vllm 0.4.2 limits logprobs to 5 (changed from 32016 in original code)
- Logprob objects require `.logprob` attribute access
- Removed `max_depth` kwarg from `call_model_rerank_w_scores_batch`

### Environment: Adaptive-RAG

```bash
module load anaconda/3
conda create -n adaptiverag python=3.8
conda activate adaptiverag
pip install torch==1.13.1+cu117 transformers==4.26.0.dev0 accelerate==0.20.3
pip install datasets evaluate scikit-learn sentencepiece nltk jsonlines rapidfuzz ftfy
```

**Known fixes applied:**
- PyTorch 1.13 does not support H100 (sm_90); must use `--partition=gpu` (L40s nodes with sm_89)
- accelerate must be pinned to 0.20.3 (newer versions removed `use_fp16` attribute)
- jsonnet-binary used instead of jsonnet (GCC compilation issues on cluster)

---

## Reproducing Results

### Self-RAG Baselines

```bash
# Download model
huggingface-cli download selfrag/selfrag_llama2_7b --local-dir ~/selfrag/models/selfrag_llama2_7b/

# Run experiments (example: PopQA, adaptive mode)
python run_short_form.py \
    --model_name ~/selfrag/models/selfrag_llama2_7b/ \
    --input_file eval_data/popqa_longtail_w_gs.jsonl \
    --mode adaptive_retrieval \
    --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file ~/selfrag/results/popqa_adaptive.json \
    --metric match \
    --ndocs 5 \
    --use_groundness --use_utility --use_seqscore \
    --dtype half \
    --task qa

# Repeat with --mode always_retrieve and --mode no_retrieval
# Repeat for triviaqa, pubhealth, arc datasets
```

### Adaptive-RAG Baselines

```bash
# Step 1: Train classifier (submits to L40s GPU nodes)
cd ~/adaptive-rag/repos/Adaptive-RAG/classifier
sbatch ~/adaptive-rag/train_classifier.sh  # Trains epochs 15, 20, 25, 30, 35

# Step 2: Route predictions using best classifier (epoch 30)
cd ~/adaptive-rag/repos/Adaptive-RAG
python classifier/postprocess/predict_complexity_on_classification_results.py flan_t5_xxl

# Step 3: Evaluate
python evaluate_final_acc.py
```

### Generate Figures

```bash
# Self-RAG figures
python ~/selfrag/analysis/generate_plots.py

# Adaptive-RAG + combined figures
python ~/adaptive-rag/analysis/generate_plots.py
```

---

## Figures & Visualizations

### Self-RAG (5 figures)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| fig1_accuracy_comparison | Grouped bar chart: accuracy across 3 retrieval modes | Retrieval critical for PopQA/TriviaQA, neutral for PubHealth, harmful for ARC-C |
| fig2_retrieval_frequency | Horizontal stacked bar: how often Self-RAG retrieves | 100% retrieval on 3/4 datasets; barely adaptive |
| fig3_retrieval_gain | Positive/negative bar: impact of retrieval per dataset | +26pp on PopQA, -0.3pp on ARC-C |
| fig4_decision_effectiveness | Quadrant scatter: retrieval frequency vs benefit | All datasets cluster in "retrieves often" region |
| fig5_per_dataset_breakdown | 2x2 subplot grid with sample sizes | Complete per-dataset view with n and retrieval % |

### Adaptive-RAG (5 figures)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| fig1_accuracy_comparison | Grouped bar: accuracy across 4 strategies (A/B/C/adaptive) | Adaptive beats individual strategies only on MuSiQue |
| fig2_route_distribution | Stacked bar: classifier routing distribution | Heavy bias toward route C (multi-step) on all datasets |
| fig3_adaptive_vs_best | Gap chart: adaptive vs best single strategy | -11.4pp gap on 2WikiMHop, +2.2pp gain on MuSiQue |
| fig4_per_route_accuracy | Per-route accuracy breakdown (3 subplots) | Route A accuracy is low; classifier struggles with "no retrieval" |
| fig5_training_curve | Validation accuracy vs training epochs | Best performance at epoch 30 (50.0%) |

### Combined Comparison (2 figures)

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| combined_fig1_decision_comparison | Side-by-side: Self-RAG retrieval gain vs Adaptive-RAG routing gap | Self-RAG over-retrieves, Adaptive-RAG under-routes |
| combined_fig2_retrieval_spectrum | Scatter plot: both systems on retrieval frequency vs gain axes | Systems occupy opposite regions of the retrieval behavior space |

---

## Planned Attack Experiments

### Attack 1: Retrieval Suppression on Self-RAG

**Goal:** Craft adversarial inputs that flip the `[Retrieve]` token from Yes to No on retrieval-dependent queries (PopQA, TriviaQA).

**Hypothesis:** If retrieval is suppressed on PopQA (where retrieval provides +26pp accuracy), we expect accuracy to collapse to near the no-retrieval baseline (~28.7%).

**Approach:** Gradient-based perturbation of input tokens to shift `[Retrieve]` logit probabilities. Measure attack success rate and accuracy degradation.

### Attack 2: Complexity Downgrade on Adaptive-RAG

**Goal:** Add adversarial perturbations to multi-hop questions that cause the T5-Large classifier to predict route A (no retrieval) instead of route C (multi-step).

**Hypothesis:** The classifier already achieves only 28% accuracy on route A, suggesting the decision boundary is fragile. Small perturbations should be sufficient to cause mis-routing.

**Approach:** TextAttack-based word substitution attacks targeting the classifier's input. Measure how often complex queries get downgraded and the resulting accuracy impact.

### Attack 3: Retrieval Forcing on Self-RAG (ARC-Challenge)

**Goal:** Force Self-RAG to retrieve and trust irrelevant passages on ARC-Challenge, where retrieval already hurts performance.

**Hypothesis:** Amplifying retrieval on a dataset where it is harmful will quantify the maximum damage from poor retrieval decisions.

**Approach:** Manipulate `[IsRel]` token probabilities to force acceptance of low-relevance passages.

### Tools & Frameworks

| Tool | Purpose | Repository |
|------|---------|------------|
| PoisonedRAG | Corpus-level knowledge base poisoning | github.com/sleeepeer/PoisonedRAG |
| TextAttack | Input-level text perturbation | github.com/QData/TextAttack |
| Joint-GCG | Gradient-based adversarial suffix generation | — |
| AgentPoison | Agent-targeted RAG poisoning | — |

---

## References

1. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection. *ICLR 2024* (Oral, Top 1%).
2. Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. *NAACL 2024*.
3. Yan, S., Gu, J., Zhu, Y., & Ling, Z. (2024). Corrective Retrieval Augmented Generation (CRAG). *arXiv preprint*.
4. Zou, J., et al. (2024). PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models. *arXiv preprint*.

---

## License

This is an academic research project. Code modifications and analysis scripts are provided for reproducibility. The underlying Self-RAG and Adaptive-RAG codebases retain their original licenses.

---



