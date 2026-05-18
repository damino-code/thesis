"""
Apply the saved Llama-3.1-70B vanilla Ridge weights to test annotations,
threshold at 0.5 to produce binary predictions, and evaluate against the
ground-truth `label` column (0 = not hate, 1 = hate).

Usage:
    python src/evaluate_ridge.py                  # uses latest merged CSV
    python src/evaluate_ridge.py path/to/merged.csv
"""

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

import config

HATE_THRESHOLD = 0.5


def latest_merged():
    files = glob.glob(str(config.RESULTS_DIR / "merged_annotations_*.csv"))
    if not files:
        raise FileNotFoundError("No merged_annotations_*.csv found. Run merge_results.py first.")
    return max(files, key=os.path.getmtime)


def load_weights():
    with open(config.RIDGE_WEIGHTS_PATH) as f:
        w = json.load(f)
    ridge = w["weights"]["ridge"]
    return ridge["intercept"], ridge["coefficients"]


def compute_features(df):
    feats = {}
    for attr in config.ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        a = pd.to_numeric(df.get(attr, 0), errors="coerce").fillna(0).values
        c = pd.to_numeric(df.get(conf_col, 0), errors="coerce").fillna(0).values
        feats[attr] = a * c
    return pd.DataFrame(feats, index=df.index)


def main():
    # Load merged annotations
    path = sys.argv[1] if len(sys.argv) > 1 else latest_merged()
    print(f"Loading annotations: {Path(path).name}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    y_true = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int)
    df = df.loc[y_true.index]

    # Apply Ridge
    intercept, coef = load_weights()
    X = compute_features(df)
    y_score = intercept + X.values @ np.array([coef[a] for a in config.ATTRIBUTES])
    y_pred  = (y_score >= HATE_THRESHOLD).astype(int)
    y_true  = y_true.values

    # Metrics
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auroc = None

    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_pos_gt   = int(np.sum(y_true))
    n_pos_pred = int(np.sum(y_pred))

    lines = [
        "=" * 72,
        "  RIDGE TEST EVALUATION — Llama-3.1-70B vanilla",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 72,
        "",
        "TASK",
        f"  Annotation model   : Llama-3.1-70B AWQ-INT4 (vanilla)",
        f"  Ridge weights from : HateSpeechScoreFull/Llama-3.1-70B/standard",
        f"  Decision threshold : hate_speech_score >= {HATE_THRESHOLD}",
        f"  Ground truth       : binary label column (0=not hate, 1=hate)",
        f"  Total samples      : {len(y_true)}",
        f"  Positive (hate)    : {n_pos_gt} / {len(y_true)}  ({100*n_pos_gt/len(y_true):.1f}%)",
        "",
        "RESULTS",
        f"  {'metric':<20} {'value':>10}",
        "  " + "-" * 32,
        f"  {'accuracy':<20} {acc:>10.4f}",
        f"  {'F1':<20} {f1:>10.4f}",
        f"  {'precision':<20} {precision:>10.4f}",
        f"  {'recall':<20} {recall:>10.4f}",
        f"  {'AUROC':<20} {auroc:>10.4f}" if auroc is not None else f"  {'AUROC':<20} {'n/a':>10}",
        f"  {'#pos_predicted':<20} {n_pos_pred:>10}",
        "  " + "-" * 32,
        "",
        "CONFUSION MATRIX  (rows = true, cols = predicted; labels = {0, 1})",
        f"            pred=0   pred=1",
        f"  true=0   {tn:>6}   {fp:>6}",
        f"  true=1   {fn:>6}   {tp:>6}",
        "",
        "INTERPRETATION",
        "  F1 / accuracy depend on the 0.5 threshold.",
        "  AUROC is threshold-free.",
        "  Training-set reference (same model, Kennedy et al. dataset):",
        "    Ridge-Vanilla  F1=0.6863  AUROC=0.8988  acc=0.8402",
        "=" * 72,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_txt  = config.RESULTS_DIR / "test_evaluation.txt"
    out_json = config.RESULTS_DIR / "test_evaluation.json"

    out_txt.write_text(report)
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n": len(y_true),
            "n_pos_gt": n_pos_gt,
            "n_pos_pred": n_pos_pred,
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auroc": auroc,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }, f, indent=2)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
