"""
Apply the saved Llama-3.1-70B vanilla Ridge weights to test annotations.

Two evaluations are produced:
  1. test_evaluation.txt          — fixed threshold 0.5 (original)
  2. test_evaluation_calibrated.txt — best threshold found on dev split,
                                      applied to test split, with full
                                      threshold sweep table (-1 to +1)

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

FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = np.round(np.arange(-2.0, 2.05, 0.05), 2)


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


def metrics_at(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold":  threshold,
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "n_pos_pred": int(np.sum(y_pred)),
    }


def full_metrics(y_true, y_score, threshold):
    m = metrics_at(y_true, y_score, threshold)
    try:
        m["auroc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        m["auroc"] = None
    cm = confusion_matrix(y_true, (y_score >= threshold).astype(int), labels=[0, 1])
    m["tn"], m["fp"], m["fn"], m["tp"] = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return m


def fmt_report(title, m, threshold, n_total, n_pos_gt, threshold_source=""):
    auroc_str = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "  n/a"
    return "\n".join([
        "=" * 72,
        f"  {title}",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 72,
        "",
        "TASK",
        f"  Annotation model   : Llama-3.1-70B AWQ-INT4 (vanilla)",
        f"  Ridge weights from : HateSpeechScoreFull/Llama-3.1-70B/standard",
        f"  Decision threshold : {threshold}" + (f"  ({threshold_source})" if threshold_source else ""),
        f"  Ground truth       : binary label column (0=not hate, 1=hate)",
        f"  Total samples      : {n_total}",
        f"  Positive (hate)    : {n_pos_gt} / {n_total}  ({100*n_pos_gt/n_total:.1f}%)",
        "",
        "RESULTS",
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
        f"            pred=0   pred=1",
        f"  true=0   {m['tn']:>6}   {m['fp']:>6}",
        f"  true=1   {m['fn']:>6}   {m['tp']:>6}",
        "",
        "INTERPRETATION",
        "  F1 / accuracy depend on the threshold.",
        "  AUROC is threshold-free.",
        "  Training-set reference (same model, Kennedy et al. dataset):",
        "    Ridge-Vanilla  F1=0.6863  AUROC=0.8988  acc=0.8402",
        "=" * 72,
    ])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_merged()
    print(f"Loading annotations: {Path(path).name}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")

    # Re-attach split column from original test CSV if present
    orig = pd.read_csv(config.TEST_CSV, on_bad_lines="skip", engine="python").reset_index(drop=True)
    orig.index.name = "index"
    orig = orig.reset_index()
    has_split = "split" in orig.columns
    if has_split:
        df = df.merge(orig[["index", "split"]], on="index", how="left")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Ridge scores for all rows
    intercept, coef = load_weights()
    X = compute_features(df)
    df["ridge_score"] = intercept + X.values @ np.array([coef[a] for a in config.ATTRIBUTES])

    # ----------------------------------------------------------------
    # 1. Fixed threshold (0.5) on all rows — save to test_evaluation.txt
    # ----------------------------------------------------------------
    m_fixed = full_metrics(df["label"].values, df["ridge_score"].values, FIXED_THRESHOLD)
    report_fixed = fmt_report(
        f"RIDGE TEST EVALUATION — Llama-3.1-70B vanilla  [{config.DATASET}]",
        m_fixed, FIXED_THRESHOLD, len(df), int(df["label"].sum()),
    )
    out_fixed = config.RESULTS_DIR / "test_evaluation.txt"
    out_fixed.write_text(report_fixed)
    print(report_fixed)

    # ----------------------------------------------------------------
    # 2. Calibrated threshold — sweep on dev, evaluate on test split
    #    (skipped if dataset has no 'split' column)
    # ----------------------------------------------------------------
    if not has_split:
        # No dev/test split — sweep on all rows
        print("\n[info] No 'split' column — sweeping threshold on full dataset.")
        sweep = [metrics_at(df["label"].values, df["ridge_score"].values, t)
                 for t in SWEEP_THRESHOLDS]
        best = max(sweep, key=lambda x: x["f1"])
        best_threshold = best["threshold"]
        print(f"Best threshold (full set): {best_threshold}  (F1={best['f1']:.4f})")

        m_cal = full_metrics(df["label"].values, df["ridge_score"].values, best_threshold)

        sweep_lines = [
            "",
            "THRESHOLD SWEEP ON FULL DATASET",
            f"  {'threshold':>10}  {'F1':>8}  {'precision':>10}  {'recall':>8}  {'#pos_pred':>10}",
            "  " + "-" * 54,
        ]
        for s in sweep:
            marker = " ◄ best" if s["threshold"] == best_threshold else ""
            sweep_lines.append(
                f"  {s['threshold']:>10.2f}  {s['f1']:>8.4f}  {s['precision']:>10.4f}"
                f"  {s['recall']:>8.4f}  {s['n_pos_pred']:>10}{marker}"
            )
        sweep_lines.append("  " + "-" * 54)

        report_cal = fmt_report(
            f"RIDGE TEST EVALUATION (CALIBRATED) — Llama-3.1-70B vanilla  [{config.DATASET}]",
            m_cal, best_threshold, len(df), int(df["label"].sum()),
            threshold_source="best F1 on full dataset"
        )
        report_cal += "\n" + "\n".join(sweep_lines)

        out_cal = config.RESULTS_DIR / "test_evaluation_calibrated.txt"
        out_cal.write_text(report_cal)
        print("\n" + report_cal)
        print(f"\nSaved: {out_fixed}")
        print(f"Saved: {out_cal}")
        return

    dev_df  = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    print(f"\nDev split : {len(dev_df)} rows  |  Test split : {len(test_df)} rows")

    # Sweep on dev
    sweep = [metrics_at(dev_df["label"].values, dev_df["ridge_score"].values, t)
             for t in SWEEP_THRESHOLDS]

    best = max(sweep, key=lambda x: x["f1"])
    best_threshold = best["threshold"]
    print(f"Best threshold on dev: {best_threshold}  (F1={best['f1']:.4f})")

    # Evaluate best threshold on test split
    m_cal = full_metrics(test_df["label"].values, test_df["ridge_score"].values, best_threshold)

    # Threshold sweep table
    sweep_lines = [
        "",
        "THRESHOLD SWEEP ON DEV SPLIT",
        f"  {'threshold':>10}  {'F1':>8}  {'precision':>10}  {'recall':>8}  {'#pos_pred':>10}",
        "  " + "-" * 54,
    ]
    for s in sweep:
        marker = " ◄ best" if s["threshold"] == best_threshold else ""
        sweep_lines.append(
            f"  {s['threshold']:>10.2f}  {s['f1']:>8.4f}  {s['precision']:>10.4f}"
            f"  {s['recall']:>8.4f}  {s['n_pos_pred']:>10}{marker}"
        )
    sweep_lines.append("  " + "-" * 54)

    report_cal = fmt_report(
        f"RIDGE TEST EVALUATION (CALIBRATED) — Llama-3.1-70B vanilla  [{config.DATASET}]",
        m_cal, best_threshold, len(test_df), int(test_df["label"].sum()),
        threshold_source="best F1 on dev split"
    )
    report_cal += "\n" + "\n".join(sweep_lines)

    out_cal = config.RESULTS_DIR / "test_evaluation_calibrated.txt"
    out_cal.write_text(report_cal)
    print("\n" + report_cal)

    print(f"\nSaved: {out_fixed}")
    print(f"Saved: {out_cal}")


if __name__ == "__main__":
    main()
