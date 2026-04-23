"""
Run TextAttack (TextFooler) on the Adaptive-RAG classifier.
Target: MuSiQue queries correctly classified as Route C.
Goal: Find word substitutions that flip C -> A or C -> B.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# TextAttack imports
import textattack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

# Our wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier_wrapper import AdaptiveRAGClassifierWrapper


# ============================================================
# Config
# ============================================================
MODEL_PATH = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
              "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
              "flan_t5_xxl/epoch/30")

PREDICT_FILE = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                "data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json")

ROUTING_FILE = ("/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
                "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
                "flan_t5_xxl/epoch/30/predict/dict_id_pred_results.json")

OUTPUT_DIR = "/home/sbhengesa/adaptive-rag/attack/results"
NUM_QUERIES_TO_ATTACK = 25   # Start small to validate. Scale up later.

# Label mapping: A=0, B=1, C=2
LABEL_MAP = {"A": 0, "B": 1, "C": 2}


# ============================================================
# Step 1: Load and filter data
# ============================================================
print("=" * 60)
print("Loading MuSiQue data...")
print("=" * 60)

with open(PREDICT_FILE) as f:
    all_queries = json.load(f)
print(f"Total queries in predict.json: {len(all_queries)}")

# Filter to MuSiQue only
musique_queries = [q for q in all_queries if q["dataset_name"] == "musique"]
print(f"MuSiQue queries: {len(musique_queries)}")

with open(ROUTING_FILE) as f:
    routing = json.load(f)
print(f"Routing predictions: {len(routing)}")

# Build list of (question, predicted_route) for MuSiQue queries
# Attack only those the classifier predicts as Route C
route_c_queries = []
for q in musique_queries:
    qid = q["id"]
    if qid not in routing:
        continue
    pred = routing[qid]["prediction"]
    if pred == "C":
        route_c_queries.append({
            "id": qid,
            "question": q["question"],
            "predicted_route": pred,
        })

print(f"MuSiQue queries predicted as Route C: {len(route_c_queries)}")

# Take the first N for attack
attack_queries = route_c_queries[:NUM_QUERIES_TO_ATTACK]
print(f"Will attack {len(attack_queries)} queries")


# ============================================================
# Step 2: Load model wrapper
# ============================================================
print("\n" + "=" * 60)
print("Loading classifier wrapper...")
print("=" * 60)
model_wrapper = AdaptiveRAGClassifierWrapper(MODEL_PATH, device="cuda:0")


# ============================================================
# Step 3: Build TextAttack dataset
# Label = 2 means "ground truth is C". Attack succeeds if model predicts A(0) or B(1).
# ============================================================
ta_data = [(q["question"], LABEL_MAP["C"]) for q in attack_queries]
ta_dataset = Dataset(ta_data)
print(f"TextAttack dataset size: {len(ta_dataset)}")


# ============================================================
# Step 4: Build attack (TextFooler recipe)
# ============================================================
print("\n" + "=" * 60)
print("Building TextFooler attack...")
print("=" * 60)
attack = TextFoolerJin2019.build(model_wrapper)


# ============================================================
# Step 5: Run attack
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"textfooler_attack_{timestamp}.txt")
csv_file = os.path.join(OUTPUT_DIR, f"textfooler_attack_{timestamp}.csv")

attack_args = AttackArgs(
    num_examples=len(ta_dataset),
    log_to_txt=log_file,
    log_to_csv=csv_file,
    checkpoint_interval=25,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
    disable_stdout=False,
)

print(f"\nStarting attack on {len(ta_dataset)} Route C queries...")
print(f"Results will be written to: {log_file}")
print(f"CSV: {csv_file}")
print("=" * 60)

attacker = Attacker(attack, ta_dataset, attack_args)
results = attacker.attack_dataset()

print("\n" + "=" * 60)
print("Attack complete.")
print("=" * 60)
print(f"Total queries attacked: {len(results)}")
