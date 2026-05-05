"""
Evaluation for direct hate-speech-score prompting.

Runs both regression and classification metrics against the IRT-calibrated
hate_speech_score from processed_dataset.csv.

  Per-row level (predictions × annotators):
    - Pearson, Spearman correlation
    - R², MAE, RMSE
  Unique-comment level (mean across annotators):
    - same regression metrics
    - F1 / accuracy / AUROC at threshold = 0.5

Run for the latest CSV in results/<mode>/.

Usage:
  echo "1" | python src/evaluation.py    # vanilla
  echo "2" | python src/evaluation.py    # persona
"""

import json
import os
import glob
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

import config
from data_loader import load_dataset

JOIN_KEYS = ["comment_id", "annotator_id"]
HATE_THRESHOLD = 0.5


def latest_match(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return max(files, key=os.path.getmtime)


def regression_block(y_true, y_pred):
    pr, pr_p = pearsonr(y_true, y_pred)
    sr, sr_p = spearmanr(y_true, y_pred)
    return {
        "n":         int(len(y_true)),
        "pearson_r": float(pr), "pearson_p": float(pr_p),
        "spearman_r": float(sr), "spearman_p": float(sr_p),
        "r2":        float(r2_score(y_true, y_pred)),
        "mae":       float(mean_absolute_error(y_true, y_pred)),
        "rmse":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def classification_block(y_true_score, y_pred_score, threshold=HATE_THRESHOLD):
    y_true_bin = (np.asarray(y_true_score) >= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred_score) >= threshold).astype(int)

    block = {
        "threshold":   float(threshold),
        "n":           int(len(y_true_bin)),
        "n_pos_gt":    int(y_true_bin.sum()),
        "n_pos_pred":  int(y_pred_bin.sum()),
        "accuracy":    float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1":          float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision":   float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":      float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
    }
    try:
        block["auroc"] = (
            float(roc_auc_score(y_true_bin, y_pred_score))
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


def evaluate_mode(mode):
    print(f"\n=== Mode: {mode.upper()} ===")
    pattern = os.path.join(config.RESULTS_FOLDER, mode, "results_hate_speech_score_*.csv")
    pred_path = latest_match(pattern)
    print(f"Predictions: {os.path.basename(pred_path)}")

    pred_df = pd.read_csv(pred_path, low_memory=False)
    pred_col = f"{config.ATTRIBUTE}_pred"
    if pred_col not in pred_df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in {pred_path}")

    pred_df[pred_col] = pd.to_numeric(pred_df[pred_col], errors="coerce")

    human = load_dataset()[JOIN_KEYS + ["hate_speech_score"]].dropna()

    # Per-row evaluation requires annotator_id in predictions (persona mode).
    per_row = {}
    if "annotator_id" in pred_df.columns:
        merged = pred_df.merge(human, on=JOIN_KEYS, how="inner").dropna(subset=[pred_col])
        if len(merged) > 0:
            per_row = {
                "n_rows":     int(len(merged)),
                "regression": regression_block(merged["hate_speech_score"].values,
                                               merged[pred_col].values),
                "classification": classification_block(merged["hate_speech_score"].values,
                                                       merged[pred_col].values),
            }

    # Unique-comment evaluation: average predictions per comment, average GT per comment.
    pred_by_comment = pred_df.groupby("comment_id", as_index=False)[pred_col].mean()
    gt_by_comment = (human.groupby("comment_id", as_index=False)["hate_speech_score"]
                          .mean())
    merged_uc = pred_by_comment.merge(gt_by_comment, on="comment_id", how="inner") \
                               .dropna(subset=[pred_col])
    print(f"Per-comment merged rows: {len(merged_uc)}")

    per_comment = {
        "n_comments":   int(len(merged_uc)),
        "regression":   regression_block(merged_uc["hate_speech_score"].values,
                                         merged_uc[pred_col].values),
        "classification": classification_block(merged_uc["hate_speech_score"].values,
                                                merged_uc[pred_col].values),
    }

    return {
        "pred_file":   pred_path,
        "per_row":     per_row,
        "per_comment": per_comment,
    }


def write_report(mode, results):
    out_dir = os.path.join(config.RESULTS_FOLDER, mode)
    os.makedirs(out_dir, exist_ok=True)
    txt_path  = os.path.join(out_dir, "evaluation_report.txt")
    json_path = os.path.join(out_dir, "evaluation_metrics.json")

    pc = results["per_comment"]
    pr = results.get("per_row") or {}
    pc_reg = pc["regression"]; pc_cls = pc["classification"]

    lines = [
        "=" * 78,
        f"  HATE-SPEECH-SCORE PROMPT — {mode.upper()}",
        f"  Generated   : {datetime.now().isoformat()}",
        f"  Predictions : {os.path.basename(results['pred_file'])}",
        "=" * 78,
        "",
        "PER-COMMENT (mean across annotators)",
        f"  n           : {pc['n_comments']}",
        f"  Pearson r   : {pc_reg['pearson_r']:+.4f}   (p={pc_reg['pearson_p']:.2g})",
        f"  Spearman ρ  : {pc_reg['spearman_r']:+.4f}  (p={pc_reg['spearman_p']:.2g})",
        f"  R²          : {pc_reg['r2']:.4f}",
        f"  MAE / RMSE  : {pc_reg['mae']:.4f}  /  {pc_reg['rmse']:.4f}",
        "",
        f"  Threshold   : hate_speech_score >= {pc_cls['threshold']}",
        f"  Positives   : {pc_cls['n_pos_gt']} / {pc_cls['n']}",
        f"  Accuracy    : {pc_cls['accuracy']:.4f}",
        f"  F1          : {pc_cls['f1']:.4f}",
        f"  Precision   : {pc_cls['precision']:.4f}",
        f"  Recall      : {pc_cls['recall']:.4f}",
        f"  AUROC       : {pc_cls['auroc']:.4f}" if pc_cls['auroc'] is not None
            else "  AUROC       : n/a",
        "",
    ]
    if pr:
        pr_reg = pr["regression"]; pr_cls = pr["classification"]
        lines += [
            "PER-ROW (joined on comment_id × annotator_id)",
            f"  n           : {pr['n_rows']}",
            f"  Pearson r   : {pr_reg['pearson_r']:+.4f}   (p={pr_reg['pearson_p']:.2g})",
            f"  Spearman ρ  : {pr_reg['spearman_r']:+.4f}  (p={pr_reg['spearman_p']:.2g})",
            f"  R²          : {pr_reg['r2']:.4f}",
            f"  MAE / RMSE  : {pr_reg['mae']:.4f}  /  {pr_reg['rmse']:.4f}",
            "",
            f"  Accuracy    : {pr_cls['accuracy']:.4f}",
            f"  F1          : {pr_cls['f1']:.4f}",
            f"  Precision   : {pr_cls['precision']:.4f}",
            f"  Recall      : {pr_cls['recall']:.4f}",
            f"  AUROC       : {pr_cls['auroc']:.4f}" if pr_cls['auroc'] is not None
                else "  AUROC       : n/a",
            "",
        ]

    lines.append("=" * 78)
    open(txt_path, "w").write("\n".join(lines))
    open(json_path, "w").write(json.dumps(results, indent=2, default=str))
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


def main():
    print("\nSelect Mode to Evaluate:")
    print("  1. Vanilla (results/standard)")
    print("  2. Persona (results/persona)")
    choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
    mode = "persona" if choice == "2" else "standard"

    try:
        results = evaluate_mode(mode)
        write_report(mode, results)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
