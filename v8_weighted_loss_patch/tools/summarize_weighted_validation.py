#!/usr/bin/env python3
"""Audit and compare validation-only outputs; never reads test predictions."""
import argparse
import csv
import json
from pathlib import Path

CLASSES = ["FORWARD", "LEFT", "RIGHT"]

def prediction_metrics(path):
    cm = [[0] * 3 for _ in range(3)]
    labels = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["true", "pred"]:
            raise ValueError(f"Unexpected prediction columns: {path}")
        for row in reader:
            t, p = int(row["true"]), int(row["pred"])
            if t not in range(3) or p not in range(3):
                raise ValueError(f"Invalid labels: {path}")
            cm[t][p] += 1
            labels.append(t)
    if not labels:
        raise ValueError(f"Empty predictions: {path}")
    recalls, precisions, f1s = [], [], []
    per_class = {}
    for i, label in enumerate(CLASSES):
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        recall = cm[i][i] / max(support, 1)
        precision = cm[i][i] / max(predicted, 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-9)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        per_class[label] = dict(recall=round(recall, 4), precision=round(precision, 4),
                               f1=round(f1, 4), support=support)
    return dict(accuracy=round(100 * sum(cm[i][i] for i in range(3)) / len(labels), 3),
                balanced_accuracy=round(100 * sum(recalls) / 3, 3),
                macro_f1=round(sum(f1s) / 3, 4),
                per_class=per_class, confusion=cm), labels

def collect(root, arm):
    rows = []
    label_reference = None
    for model, graph in [("wireflynet", "bio"), ("cnn_matched", "none")]:
        for seed in [42, 43, 44]:
            name = f"{model}_{graph}_seed{seed}"
            folder = root / name
            summary = folder / "validation_result.json"
            if not summary.is_file():
                summary = folder / "result.json"  # Original baseline's val section only.
            history = folder / "history.csv"
            predictions = folder / "val_predictions.csv"
            for path in [summary, history, predictions]:
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete {arm} run: {path}")
            r = json.loads(summary.read_text())
            if r["run"] != name or r["seed"] != seed or r["model"] != model or r["graph"] != graph:
                raise ValueError(f"Run identity mismatch: {summary}")
            with history.open(newline="") as f:
                h = list(csv.DictReader(f))
            if not h or [int(x["epoch"]) for x in h] != list(range(1, len(h) + 1)):
                raise ValueError(f"Invalid epoch sequence: {history}")
            best = max(h, key=lambda x: float(x["val_f1"]))  # First maximum, matching trainer.
            if int(best["epoch"]) != r["best_epoch"] or float(best["val_f1"]) != r["val"]["macro_f1"]:
                raise ValueError(f"Checkpoint/history mismatch: {folder}")
            computed, labels = prediction_metrics(predictions)
            if computed != r["val"]:
                raise ValueError(f"Saved validation metrics do not match predictions: {folder}")
            if label_reference is None:
                label_reference = labels
            elif labels != label_reference:
                raise ValueError(f"Validation label order differs: {folder}")
            if arm == "weighted_loss":
                metadata = json.loads((folder / "run_metadata.json").read_text())
                if metadata["train_sampler"] != "shuffle" or metadata["class_weighting"] != "inverse_frequency" or metadata["eval_test"]:
                    raise ValueError(f"Unexpected follow-up protocol: {folder}")
                if (folder / "test_predictions.csv").exists() or (folder / "result.json").exists():
                    raise ValueError(f"Follow-up folder already contains test outputs: {folder}")
            row = dict(arm=arm, run=name, model=model, graph=graph, seed=seed,
                       params=r["params"], epochs_completed=len(h), best_epoch=r["best_epoch"],
                       val_accuracy=computed["accuracy"],
                       val_balanced_accuracy=computed["balanced_accuracy"], val_macro_f1=computed["macro_f1"])
            for c in CLASSES:
                pc = computed["per_class"][c]
                for metric in ["recall", "precision", "f1", "support"]:
                    row[f"val_{c}_{metric}"] = pc[metric]
            rows.append(row)
    return rows, label_reference

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=Path("experiments"))
    ap.add_argument("--weighted", type=Path, default=Path("experiments_weighted_loss"))
    ap.add_argument("--out", type=Path, default=Path("experiments_weighted_loss/validation_comparison.csv"))
    args = ap.parse_args()
    if args.baseline.resolve() == args.weighted.resolve():
        raise ValueError("Baseline and weighted output roots must be different")
    baseline, blabels = collect(args.baseline, "baseline")
    weighted, wlabels = collect(args.weighted, "weighted_loss")
    if blabels != wlabels:
        raise ValueError("Baseline/follow-up validation labels or row order differ")
    rows = baseline + weighted
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("VALIDATED: 6 baseline + 6 follow-up runs, same validation labels; no test metrics compared.")
    print("arm | model | seed | selected epoch | val macro-F1 | val balanced accuracy | LEFT precision | LEFT recall")
    for r in rows:
        print(f"{r['arm']} | {r['model']} | {r['seed']} | {r['best_epoch']} | "
              f"{r['val_macro_f1']:.4f} | {r['val_balanced_accuracy']:.3f}% | "
              f"{100*r['val_LEFT_precision']:.2f}% | {100*r['val_LEFT_recall']:.2f}%")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()