"""
Evaluate all prompt strategies on test_2000.csv.

Ground truth: binary `label` column (0=not hate, 1=hate).
Prediction  : P(yes) >= 0.5  →  predicted hate.

Saves: results/classification_comparison.txt and .json
"""

import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

import config
from data_loader import load_dataset

HATE_THRESHOLD = 0.5


def latest_match(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def metrics_block(y_true_bin, y_score, decision_threshold=0.5):
    y_pred_bin = (np.asarray(y_score) >= decision_threshold).astype(int)
    block = {
        "n":          int(len(y_true_bin)),
        "n_pos_gt":   int(np.sum(y_true_bin)),
        "n_pos_pred": int(np.sum(y_pred_bin)),
        "accuracy":   float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1":         float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision":  float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":     float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
    }
    try:
        block["auroc"] = (
            float(roc_auc_score(y_true_bin, y_score))
            if len(np.unique(y_true_bin)) == 2 else None
        )
    except ValueError:
        block["auroc"] = None
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    block["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return block


def evaluate_strategy(strategy):
    pattern = os.path.join(config.RESULTS_FOLDER, strategy,
                           f"results_{strategy}_*.csv")
    pred_path = latest_match(pattern)
    if pred_path is None:
        print(f"[skip:{strategy}] No results CSV.")
        return None
    print(f"[eval:{strategy}] {os.path.basename(pred_path)}")

    pred_df = pd.read_csv(pred_path, low_memory=False)
    pred_df["p_yes"] = pd.to_numeric(pred_df["p_yes"], errors="coerce")
    pred_df["label"] = pd.to_numeric(pred_df["label"], errors="coerce")
    pred_df = pred_df.dropna(subset=["p_yes", "label"])

    y_true = pred_df["label"].astype(int).values
    y_score = pred_df["p_yes"].values

    return {
        "pred_file": pred_path,
        "results":   metrics_block(y_true, y_score),
    }


def fmt_row(label, m):
    if m is None:
        return f"  {label:<32}  (no data)"
    auroc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "  n/a"
    return (f"  {label:<32}"
            f"{m['accuracy']:>9.4f}"
            f"{m['f1']:>9.4f}"
            f"{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}"
            f"{auroc:>9}"
            f"{m['n_pos_pred']:>10}")


def write_report(all_results):
    txt_path  = os.path.join(config.RESULTS_FOLDER, "classification_comparison.txt")
    json_path = os.path.join(config.RESULTS_FOLDER, "classification_comparison.json")

    sample  = next((r for r in all_results.values() if r), None)
    n_total = sample["results"]["n"] if sample else 0
    n_pos   = sample["results"]["n_pos_gt"] if sample else 0

    lines = [
        "=" * 92,
        "  HATE-SPEECH CLASSIFICATION — baseline2000 (test_2000.csv)",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 92,
        "",
        "TASK",
        f"  Dataset        : test_2000.csv",
        f"  Binary target  : label column (0=not hate, 1=hate)",
        f"  Decision rule  : P(yes) >= 0.5  →  predicted hate",
        f"  Total comments : {n_total}",
        f"  Positive count : {n_pos}  ({100*n_pos/n_total:.1f}%)" if n_total else "",
        "",
        "RESULTS",
        "  " + "-" * 90,
        f"  {'strategy':<32}{'accuracy':>9}{'F1':>9}{'precision':>11}{'recall':>9}{'AUROC':>9}{'#pos_pred':>10}",
        "  " + "-" * 90,
    ]
    for s in config.PROMPT_STRATEGIES:
        r = all_results.get(s)
        lines.append(fmt_row(s, r["results"] if r else None))
    lines.append("  " + "-" * 90)

    lines += [
        "",
        "INTERPRETATION",
        "  - zero_shot                   : bare yes/no, no extra context",
        "  - few_shot                    : zero-shot + 4 labelled examples",
        "                                   (from Kennedy et al., hate_speech_score > 0.5)",
        "  - definition                  : formal hate-speech definition added",
        "  - attribute_aware_no_values   : lists 10 dimensions, model assesses each",
        "  - attribute_aware_with_values : uses Llama-3.1-70B attribute annotations",
        "                                   from testRdige/results/ (low/moderate/high)",
        "  Compare to:",
        "    Ridge pre-trained (thresh=0.5)   F1=0.573  AUROC=0.843",
        "    Ridge pre-trained (thresh=-1.5)  F1=0.802  AUROC=0.843",
        "    Ridge trained on test_2000 (OOF) F1=0.816  AUROC=0.892",
        "=" * 92,
    ]

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")


def main():
    all_results = {}
    for strategy in config.PROMPT_STRATEGIES:
        try:
            all_results[strategy] = evaluate_strategy(strategy)
        except Exception as e:
            print(f"[error:{strategy}] {e}")
            import traceback; traceback.print_exc()

    write_report(all_results)

    print("\n" + "=" * 70)
    print(f"  SUMMARY")
    print(f"  {'strategy':<32}{'F1':>9}{'AUROC':>9}{'acc':>9}")
    print("  " + "-" * 60)
    for s in config.PROMPT_STRATEGIES:
        r = all_results.get(s)
        if not r:
            print(f"  {s:<32}  (no data)")
            continue
        m = r["results"]
        au = f"{m['auroc']:.4f}" if m["auroc"] is not None else "  n/a"
        print(f"  {s:<32}{m['f1']:>9.4f}{au:>9}{m['accuracy']:>9.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
