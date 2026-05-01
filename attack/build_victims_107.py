"""Combine the 107 black-box-attacked victims into one input file for white-box."""
import json
from pathlib import Path

PHASE_FILES = [
    "results/attacks/phase1/phase1_A.jsonl",
    "results/attacks/phase1/phase1_B.jsonl",
    "results/attacks/phase1/phase1_C.jsonl",
    "results/attacks/phase2/phase2_C.jsonl",
]
OUT = "results/attacks/whitebox/victims_107.jsonl"

n_total = 0
n_baseline_correct = 0
seen_qids = set()
with open(OUT, "w") as fo:
    for path in PHASE_FILES:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_total += 1
                qid = rec["qid"]
                if qid in seen_qids:
                    print("DUPLICATE qid skipped: " + qid)
                    continue
                if not rec.get("baseline_em", False):
                    continue
                n_baseline_correct += 1
                seen_qids.add(qid)
                # Rename baseline_answer -> system_answer to match white-box script
                out_rec = {
                    "qid": rec["qid"],
                    "question": rec["question"],
                    "gold_answer_spans": rec.get("gold_answer_spans", []),
                    "actual_route": rec.get("baseline_actual_route"),
                    "baseline_actual_route": rec.get("baseline_actual_route"),
                    "system_answer": rec.get("baseline_answer"),
                    "exact_match": rec.get("baseline_em", False),
                }
                fo.write(json.dumps(out_rec) + "\n")

print("Read " + str(n_total) + " records across " + str(len(PHASE_FILES)) + " phase files")
print("Wrote " + str(n_baseline_correct) + " baseline-correct victims to " + OUT)
