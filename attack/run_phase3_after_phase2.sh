#!/bin/bash
# Wait for Phase 2 (PID 1911327) to complete, then auto-launch Phase 3.
# Phase 3 attacks the hardest failures from Phase 1+2 with extended budget (200).

cd /home/sbhengesa/adaptive-rag/attack
source /home/sbhengesa/adaptive-rag/attack/setup_env.sh

echo "$(date): Watching Phase 2 (PID 1911327)..."
while ps -p 1911327 > /dev/null 2>&1; do
    sleep 60
done

echo "$(date): Phase 2 done. Selecting Phase 3 victims..."
mkdir -p results/attacks/phase3

python << 'PYEOF'
import json
from pathlib import Path

all_attacks = []
for path in [
    'results/attacks/phase1/phase1_A.jsonl',
    'results/attacks/phase1/phase1_B.jsonl',
    'results/attacks/phase1/phase1_C.jsonl',
    'results/attacks/phase2/phase2_C.jsonl',
]:
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                all_attacks.append(json.loads(line))

failed = [a for a in all_attacks
          if not (a.get('attack_success_strict') or a.get('attack_success'))]
print(f"Total failed attacks: {len(failed)}")

by_route = {'A': [], 'B': [], 'C': []}
for a in failed:
    r = a.get('baseline_actual_route')
    if r in by_route:
        by_route[r].append(a)
print(f"  By route: A={len(by_route['A'])}, B={len(by_route['B'])}, C={len(by_route['C'])}")

selected = (by_route['A'][:5] + by_route['B'][:15] + by_route['C'][:30])
print(f"Selected {len(selected)} for Phase 3")

out = []
for a in selected:
    out.append({
        'qid': a['qid'],
        'question': a['question'],
        'system_answer': a['baseline_answer'],
        'gold_answer_spans': a.get('gold_answer_spans'),
        'actual_route': a.get('baseline_actual_route'),
        'predicted_route': a.get('baseline_actual_route'),
        'exact_match': a.get('baseline_em', True),
        'loose_match': True,
    })

with open('results/attacks/phase3/phase3_targets.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
print(f"Wrote {len(out)} targets to results/attacks/phase3/phase3_targets.jsonl")
PYEOF

echo "$(date): Launching Phase 3 with budget=200..."
nohup python -u attack_blackbox_v3.py \
    --input-jsonl results/attacks/phase3/phase3_targets.jsonl \
    --output-jsonl results/attacks/phase3/phase3.jsonl \
    --only-correct \
    --max-oracle-queries 200 \
    > results/attacks/phase3/phase3.log 2>&1 &

P3_PID=$!
echo "$(date): Phase 3 launched, PID=$P3_PID"
echo $P3_PID > results/attacks/phase3/phase3.pid
