"""
Classification metrics for the Ridge model trained on test_2000.

Loads OOF predictions from results/oof_predictions.csv and evaluates
at threshold 0.5, comparing F1/AUROC against the pre-trained model results.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
HATE_THRESHOLD = 0.5


def metrics_at(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auroc = None
    return {
        "threshold":   threshold,
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc":       auroc,
        "n_pos_pred":  int(np.sum(y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    oof_df = pd.read_csv(RESULTS_DIR / "oof_predictions.csv")
    y_true  = oof_df["y_true"].values.astype(int)
    y_score = oof_df["y_pred"].values
    n_pos_gt = int(np.sum(y_true))

    m = metrics_at(y_true, y_score, HATE_THRESHOLD)
    auroc_str = f"{m['auroc']:.4f}" if m["auroc"] is not None else "n/a"

    lines = [
        "=" * 72,
        "  RIDGE CLASSIFICATION METRICS — trainRidge2000",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 72,
        "",
        "TASK",
        "  Annotation model   : Llama-3.1-70B AWQ-INT4 (vanilla)",
        "  Ridge target       : binary label (0=not hate, 1=hate)",
        "  OOF predictions    : 5-fold cross-validation",
        f"  Decision threshold : >= {HATE_THRESHOLD}",
        f"  Total samples      : {len(y_true)}",
        f"  Positive (hate)    : {n_pos_gt} / {len(y_true)}  ({100*n_pos_gt/len(y_true):.1f}%)",
        "",
        "RESULTS  (OOF — out-of-fold, unbiased estimate)",
        f"  {'metric':<20} {'value':>10}",
        "  " + "-" * 32,
        f"  {'accuracy':<20} {m['accuracy']:>10.4f}",
        f"  {'F1':<20} {m['f1']:>10.4f}",
        f"  {'precision':<20} {m['precision']:>10.4f}",
        f"  {'recall':<20} {m['recall']:>10.4f}",
        f"  {'AUROC':<20} {auroc_str:>10}",
        f"  {'#pos_predicted':<20} {m['n_pos_pred']:>10}",
        "  " + "-" * 32,
        "",
        "CONFUSION MATRIX  (rows = true, cols = predicted; labels = {0, 1})",
        "            pred=0   pred=1",
        f"  true=0   {m['tn']:>6}   {m['fp']:>6}",
        f"  true=1   {m['fn']:>6}   {m['tp']:>6}",
        "",
        "COMPARISON",
        "  Pre-trained Llama-3.1-70B vanilla (threshold=0.5, Kennedy et al.)",
        "    F1=0.6863  AUROC=0.8988  acc=0.8402",
        "  Pre-trained Llama-3.1-70B vanilla (threshold=-1.5, test_2000)",
        "    F1=0.8023  AUROC=0.8433  acc=0.7595",
        "  This model trained directly on test_2000 (OOF, threshold=0.5)",
        f"    F1={m['f1']:.4f}  AUROC={auroc_str}  acc={m['accuracy']:.4f}",
        "=" * 72,
    ]

    report = "\n".join(lines)
    print(report)

    out_txt  = RESULTS_DIR / "classification_comparison.txt"
    out_json = RESULTS_DIR / "classification_comparison.json"

    out_txt.write_text(report)
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n": len(y_true),
            "n_pos_gt": n_pos_gt,
            **{k: m[k] for k in ("accuracy","f1","precision","recall","auroc",
                                  "n_pos_pred","tn","fp","fn","tp")},
        }, f, indent=2)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
