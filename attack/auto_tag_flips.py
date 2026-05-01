"""Auto-tag flips using Groq-Llama as an NLI judge. Quick & dirty."""
import json, os, time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
SLEEP = 2.5

JUDGE_PROMPT = """You are evaluating whether a rephrasing of a question is faithful to the original.

Original question: {original}
Rephrased question: {rephrase}

Reply with EXACTLY ONE of these labels and a brief reason:
- CLEAN: Same question, different surface form. May be verbose or stilted but introduces no new facts/entities.
- HALLUCINATED: Introduces a specific entity, name, place, date, or fact NOT in the original.
- CHANGED: Asks something genuinely different. The referent or the asked-about property shifted.
- UNCERTAIN: Borderline, hard to call.

Format: LABEL: reason
"""

flips = []
with open("results/attacks/whitebox_complexity_groq/attack_complexity_groq_n107.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("first_success"):
            flips.append({
                "qid": r["qid"], "original": r["question"],
                "rephrase": r["first_success"]["rephrase"],
                "style": r["first_success"]["style"]
            })

print(f"Auto-tagging {len(flips)} flips with Llama-as-judge...")
out_path = "results/attacks/whitebox_complexity_groq/flip_tags_auto.jsonl"
counts = {"CLEAN": 0, "HALLUCINATED": 0, "CHANGED": 0, "UNCERTAIN": 0, "ERROR": 0}

with open(out_path, "w") as fo:
    for i, fl in enumerate(flips, 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(original=fl["original"],
                                                          rephrase=fl["rephrase"])}],
                temperature=0.1, max_tokens=120,
            )
            txt = resp.choices[0].message.content.strip()
            label = "ERROR"
            for l in ["CLEAN", "HALLUCINATED", "CHANGED", "UNCERTAIN"]:
                if txt.upper().startswith(l):
                    label = l; break
            counts[label] += 1
            fo.write(json.dumps({"qid": fl["qid"], "label": label,
                                 "judge_response": txt, "style": fl["style"]}) + "\n")
            print(f"  [{i}/{len(flips)}] {fl['qid']:35s}  {label:13s}  ({fl['style']})")
            time.sleep(SLEEP)
        except Exception as e:
            print(f"  [{i}/{len(flips)}] ERROR: {e}"); counts["ERROR"] += 1; time.sleep(SLEEP)

total = sum(v for k, v in counts.items() if k != "ERROR")
print()
print("=" * 50)
print("AUTO-TAG SUMMARY")
print("=" * 50)
for label, n in counts.items():
    pct = 100 * n / max(1, total) if label != "ERROR" else None
    if pct is not None:
        print(f"  {label:14s}  {n:3d}  ({pct:.1f}%)")
    else:
        print(f"  {label:14s}  {n:3d}")
