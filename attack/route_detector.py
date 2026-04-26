"""
Route detector — diagnostic tool for the black-box attack.

Maps observable signals (latency_ms, num_sources, answer text features) to
the route the deployed Adaptive-RAG took (A=nor_qa, B=oner_qa, C=ircot_qa).

In our pure black-box attack threat model, the ATTACKER does not use this.
We use it post-hoc to attribute attack outcomes to mechanisms:
  - Did the attack flip the route?
  - Or did same-route LLM behavior change?

Trained on the 500 baseline samples in results/final/baseline_musique_route_*.jsonl.
Designed to be interpretable (decision tree) so we can cite the discovered
thresholds as evidence in the paper.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


FEATURE_NAMES = [
    "latency_ms",
    "log_latency_ms",
    "num_sources",
    "answer_length_chars",
    "answer_length_words",
]


def extract_features(record: Dict[str, Any]) -> List[float]:
    """Compute the 5-dimensional feature vector from a pilot record."""
    answer = record.get("system_answer") or ""
    latency = float(record.get("latency_ms") or 0.0)
    num_sources = int(record.get("num_sources") or 0)
    return [
        latency,
        float(np.log1p(latency)),
        float(num_sources),
        float(len(answer)),
        float(len(answer.split())),
    ]


def load_baseline_dataset(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all 500 baseline pilot records as (X, y, qids)."""
    X, y, qids = [], [], []
    for route in ("A", "B", "C"):
        path = results_dir / f"baseline_musique_route_{route}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline file: {path}")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("actual_route") not in ("A", "B", "C"):
                    continue
                X.append(extract_features(r))
                y.append(r["actual_route"])
                qids.append(r["qid"])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    return X, y, qids


def cross_validate(X: np.ndarray, y: np.ndarray, max_depth: int = 4) -> Dict[str, float]:
    """5-fold stratified CV on the labeled baseline data."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = DecisionTreeClassifier(
            max_depth=max_depth, class_weight="balanced", random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        acc = float((pred == y[test_idx]).mean())
        accs.append(acc)
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "fold_accuracies": [float(a) for a in accs],
    }


def train_final(X: np.ndarray, y: np.ndarray, max_depth: int = 4) -> DecisionTreeClassifier:
    """Train the final detector on ALL data."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=42
    )
    clf.fit(X, y)
    return clf


def explain(clf: DecisionTreeClassifier, feature_names: List[str]) -> str:
    """Human-readable description of the trained tree."""
    return export_text(clf, feature_names=feature_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/final"))
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument(
        "--save-model",
        type=Path,
        default=Path("results/final/route_detector.joblib"),
    )
    ap.add_argument(
        "--save-report",
        type=Path,
        default=Path("results/final/route_detector_report.json"),
    )
    args = ap.parse_args()

    print(f"[1/4] Loading baseline data from {args.results_dir}...")
    X, y, qids = load_baseline_dataset(args.results_dir)
    print(f"  Loaded {len(X)} samples")
    print(f"  Class distribution: A={int((y=='A').sum())}, "
          f"B={int((y=='B').sum())}, C={int((y=='C').sum())}")

    print(f"\n[2/4] 5-fold cross-validation (max_depth={args.max_depth})...")
    cv = cross_validate(X, y, max_depth=args.max_depth)
    print(f"  Mean accuracy: {cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}")
    print(f"  Per-fold: {[f'{a:.3f}' for a in cv['fold_accuracies']]}")

    print(f"\n[3/4] Training final model on all 500 samples...")
    clf = train_final(X, y, max_depth=args.max_depth)
    train_pred = clf.predict(X)
    train_acc = float((train_pred == y).mean())
    print(f"  Training accuracy: {train_acc:.3f}")
    print("\n  Confusion matrix (rows=true, cols=pred, order=[A,B,C]):")
    cm = confusion_matrix(y, train_pred, labels=["A", "B", "C"])
    for row, label in zip(cm, ["A", "B", "C"]):
        print(f"    {label}: {row.tolist()}")

    print(f"\n[4/4] Saving model and report...")
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(
            {"model": clf, "feature_names": FEATURE_NAMES, "classes": list(clf.classes_)},
            args.save_model,
        )
        print(f"  Model saved to {args.save_model}")
    except ImportError:
        print(f"  joblib not available; skipping model save")

    report = {
        "n_samples": int(len(X)),
        "class_distribution": {
            "A": int((y == "A").sum()),
            "B": int((y == "B").sum()),
            "C": int((y == "C").sum()),
        },
        "cross_validation": cv,
        "training_accuracy": train_acc,
        "confusion_matrix": cm.tolist(),
        "feature_names": FEATURE_NAMES,
        "max_depth": args.max_depth,
        "decision_tree_text": explain(clf, FEATURE_NAMES),
    }
    with open(args.save_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {args.save_report}")

    print("\n" + "=" * 70)
    print("DECISION TREE")
    print("=" * 70)
    print(report["decision_tree_text"])


if __name__ == "__main__":
    main()
