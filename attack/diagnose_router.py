"""Router diagnostics: architecture + importance + logit confidence."""
import argparse, json, time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer


def inspect_architecture(model_path: str, out_path: Path) -> Dict[str, Any]:
    with open(Path(model_path) / "config.json") as f:
        config = json.load(f)
    print("Loading model on CPU for parameter count...", flush=True)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    summary = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "task_framing": "seq2seq generation, single-token target in {A,B,C}",
        "d_model": config.get("d_model"),
        "d_ff": config.get("d_ff"),
        "num_encoder_layers": config.get("num_layers"),
        "num_decoder_layers": config.get("num_decoder_layers", config.get("num_layers")),
        "num_heads": config.get("num_heads"),
        "vocab_size": config.get("vocab_size"),
        "feed_forward_proj": config.get("feed_forward_proj"),
        "total_parameters": total,
        "encoder_parameters": enc,
        "decoder_parameters": dec,
    }
    print("\n=== ARCHITECTURE SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return summary


class ClassifierLogits:
    def __init__(self, model_path: str):
        print("Loading classifier (CPU)...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.labels = ["A", "B", "C"]
        self.label_token_ids = {
            lab: self.tokenizer.encode(lab, add_special_tokens=False)[0]
            for lab in self.labels
        }
        print(f"Label token IDs: {self.label_token_ids}", flush=True)

    @torch.no_grad()
    def logits_abc(self, text: str) -> np.ndarray:
        inp = self.tokenizer(text, return_tensors="pt", max_length=384,
                             truncation=True, padding=True)
        out = self.model.generate(**inp, max_new_tokens=1,
                                  output_scores=True, return_dict_in_generate=True)
        L = out.scores[0][0]
        return np.array([float(L[self.label_token_ids[l]]) for l in self.labels])


POS_KEEP = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")


def is_content(tok) -> bool:
    return (not tok.is_stop and not tok.is_punct and not tok.is_space
            and tok.pos_ in POS_KEEP and bool(tok.text.strip()))


def importance_per_word(cls, nlp, q: str, route_label: str, unk: str = "<unk>"):
    doc = nlp(q)
    tokens = [t.text for t in doc]
    base = cls.logits_abc(q)
    r_idx = cls.labels.index(route_label) if route_label in cls.labels else int(np.argmax(base))
    out = []
    for i, tok in enumerate(doc):
        if not is_content(tok):
            continue
        masked = list(tokens)
        masked[i] = unk
        mlogits = cls.logits_abc(" ".join(masked))
        out.append({
            "token_idx": i,
            "word": tok.text,
            "pos": tok.pos_,
            "importance": float(base[r_idx] - mlogits[r_idx]),
        })
    return out


def run_importance(cls, nlp, victims, n, out_path):
    sample = victims[:n]
    print(f"\n=== IMPORTANCE DIAGNOSTIC (n={len(sample)}) ===", flush=True)
    maxes, pos_top, thr_counts = [], {}, {t: 0 for t in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]}
    with open(out_path, "w") as fo:
        for i, rec in enumerate(sample, 1):
            qid, q = rec["qid"], rec["question"]
            r = rec.get("actual_route") or rec.get("baseline_actual_route")
            t0 = time.time()
            try:
                imps = importance_per_word(cls, nlp, q, r)
            except Exception as e:
                print(f"  [{i}/{len(sample)}] qid={qid} ERROR: {e}", flush=True)
                continue
            wall = time.time() - t0
            if not imps:
                print(f"  [{i}/{len(sample)}] qid={qid} no content words", flush=True)
                continue
            imps.sort(key=lambda x: x["importance"], reverse=True)
            mx, top_pos = imps[0]["importance"], imps[0]["pos"]
            maxes.append(mx)
            pos_top[top_pos] = pos_top.get(top_pos, 0) + 1
            for th in thr_counts:
                if any(w["importance"] > th for w in imps):
                    thr_counts[th] += 1
            fo.write(json.dumps({
                "qid": qid, "question": q, "actual_route": r,
                "max_importance_word": imps[0]["word"], "max_importance_pos": top_pos,
                "max_importance": mx, "n_content_words": len(imps),
                "all_words": imps, "wall_s": round(wall, 2),
            }) + "\n")
            fo.flush()
            print(f"  [{i}/{len(sample)}] qid={qid}  top={imps[0]['word']!r} ({top_pos}, {mx:.3f})"
                  f"  n_words={len(imps)}  wall={wall:.1f}s", flush=True)
    summary = {
        "n_sampled": len(sample),
        "n_with_results": len(maxes),
        "max_importance_p50": float(np.percentile(maxes, 50)) if maxes else None,
        "max_importance_p90": float(np.percentile(maxes, 90)) if maxes else None,
        "max_importance_max": float(max(maxes)) if maxes else None,
        "top_pos_distribution": pos_top,
        "fraction_with_word_above": {str(t): round(c / max(1, len(maxes)), 3)
                                     for t, c in thr_counts.items()},
    }
    print("\n=== IMPORTANCE SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_confidence(cls, victims, out_path):
    print(f"\n=== LOGIT CONFIDENCE DIAGNOSTIC (n={len(victims)}) ===", flush=True)
    gaps, probs, match, by_route = [], [], [], {"A": [], "B": [], "C": []}
    with open(out_path, "w") as fo:
        for i, rec in enumerate(victims, 1):
            qid, q = rec["qid"], rec["question"]
            r_actual = rec.get("actual_route") or rec.get("baseline_actual_route")
            try:
                logits = cls.logits_abc(q)
            except Exception as e:
                print(f"  [{i}/{len(victims)}] ERROR: {e}", flush=True)
                continue
            sidx = np.argsort(logits)[::-1]
            gap = float(logits[sidx[0]] - logits[sidx[1]])
            ps = F.softmax(torch.tensor(logits), dim=0).numpy()
            top1 = cls.labels[int(sidx[0])]
            fo.write(json.dumps({
                "qid": qid, "actual_route": r_actual, "predicted_route": top1,
                "logits": logits.tolist(), "probs": ps.tolist(),
                "top1": top1, "top2": cls.labels[int(sidx[1])],
                "logit_gap_top1_top2": gap, "prob_top1": float(ps[int(sidx[0])]),
            }) + "\n")
            fo.flush()
            gaps.append(gap)
            probs.append(float(ps[int(sidx[0])]))
            match.append(top1 == r_actual)
            if r_actual in by_route:
                by_route[r_actual].append(gap)
            if i % 20 == 0 or i == len(victims):
                print(f"  [{i}/{len(victims)}]  mean gap so far: {np.mean(gaps):.2f}", flush=True)

    def pct(a, p): return float(np.percentile(a, p)) if a else None
    summary = {
        "n_victims": len(victims),
        "logit_gap": {"p10": pct(gaps, 10), "p50": pct(gaps, 50),
                      "p90": pct(gaps, 90), "mean": float(np.mean(gaps)) if gaps else None},
        "prob_top1": {"p10": pct(probs, 10), "p50": pct(probs, 50),
                      "p90": pct(probs, 90), "mean": float(np.mean(probs)) if probs else None},
        "fraction_correctly_routed": float(np.mean(match)) if match else None,
        "logit_gap_by_actual_route": {
            r: {"n": len(g), "p50": pct(g, 50),
                "mean": float(np.mean(g)) if g else None}
            for r, g in by_route.items()
        },
    }
    print("\n=== CONFIDENCE SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=(
        "/home/sbhengesa/adaptive-rag/repos/Adaptive-RAG/classifier/"
        "outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/"
        "flan_t5_xxl/epoch/30"))
    ap.add_argument("--victims", default="results/attacks/whitebox/victims_107.jsonl")
    ap.add_argument("--out-dir", default="results/diagnostics")
    ap.add_argument("--n-importance-sample", type=int, default=30)
    ap.add_argument("--skip-architecture", action="store_true")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--skip-confidence", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.skip_architecture:
        overall["architecture"] = inspect_architecture(
            args.model_path, out / "router_architecture.json")

    cls = None
    if not args.skip_importance or not args.skip_confidence:
        cls = ClassifierLogits(args.model_path)

    with open(args.victims) as f:
        victims = [json.loads(l) for l in f]
    print(f"\nLoaded {len(victims)} victims from {args.victims}", flush=True)

    if not args.skip_importance:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        overall["importance"] = run_importance(
            cls, nlp, victims, args.n_importance_sample, out / "router_importance.jsonl")

    if not args.skip_confidence:
        overall["logit_confidence"] = run_confidence(
            cls, victims, out / "router_logit_confidence.jsonl")

    overall["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out / "router_diagnostics_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {out / 'router_diagnostics_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
