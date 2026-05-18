"""
Classification metrics — Llama-3.3-70B
=======================================
Binarises the Ridge OOF predictions and compares them against the canonical
hate label from the dataset:

  binary GT    = hate_speech_score >= 0.5   (per Kennedy et al.)
  Ridge pred   = y_pred_ridge       >= 0.5

Run for both modes:
  - Ridge-Vanilla : Ridge trained on standard merged predictions
  - Ridge-Persona : Ridge trained on persona  merged predictions

Aggregation level: one row per comment_id (matches train_weights.py).

Outputs: results/classification_comparison.txt (and .json)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

BASE_DIR = Path(__file__).parent
MODEL_LABEL = "Qwen2.5-7B"
HATE_THRESHOLD = 0.5  # binarise both GT (hate_speech_score) and Ridge prediction


def load_ridge_oof(mode: str) -> pd.DataFrame:
    path = BASE_DIR / "results" / mode / "test_predictions_oof.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing OOF file for {mode}: {path}\n  Run train_weights.py first."
        )
    return pd.read_csv(path)[["comment_id", "y_true", "y_pred_ridge"]]


def metrics_block(y_true_bin: np.ndarray, y_pred_bin: np.ndarray,
                  y_score: np.ndarray) -> dict:
    out = {
        "n":           int(len(y_true_bin)),
        "n_pos_gt":    int(y_true_bin.sum()),
        "n_pos_pred":  int(y_pred_bin.sum()),
        "accuracy":    float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1":          float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision":   float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":      float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
    }
    try:
        out["auroc"] = (
            float(roc_auc_score(y_true_bin, y_score))
            if len(np.unique(y_true_bin)) == 2 else None
        )
    except ValueError:
        out["auroc"] = None
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    out["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return out


def evaluate_mode(mode: str) -> dict:
    print(f"\n--- {MODEL_LABEL} | {mode.upper()} ---")
    df = load_ridge_oof(mode)
    print(f"  unique comments: {len(df)}")

    y_true_bin   = (df["y_true"] >= HATE_THRESHOLD).astype(int).values
    ridge_scores = df["y_pred_ridge"].values
    ridge_pred   = (ridge_scores >= HATE_THRESHOLD).astype(int)

    return {
        "ridge":            metrics_block(y_true_bin, ridge_pred, ridge_scores),
        "n_total":          int(len(df)),
        "n_positive_gt":    int(y_true_bin.sum()),
        "positive_rate_gt": float(y_true_bin.mean()),
    }


def fmt_row(label: str, m: dict) -> str:
    auroc = f"{m['auroc']:.4f}" if m["auroc"] is not None else "  n/a"
    return (f"  {label:<22}"
            f"{m['accuracy']:>9.4f}"
            f"{m['f1']:>9.4f}"
            f"{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}"
            f"{auroc:>9}"
            f"{m['n_pos_pred']:>10}")


def write_report(results: dict):
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    txt_path  = out_dir / "classification_comparison.txt"
    json_path = out_dir / "classification_comparison.json"

    std = results.get("standard")
    per = results.get("persona")
    src = std or per
    n_total  = src["n_total"]          if src else 0
    n_pos    = src["n_positive_gt"]    if src else 0
    pos_rate = src["positive_rate_gt"] if src else 0.0

    lines = [
        "=" * 84,
        f"  RIDGE CLASSIFICATION METRICS — {MODEL_LABEL}",
        f"  Generated : {results['timestamp']}",
        "=" * 84,
        "",
        "TASK",
        f"  Binary target  : hate_speech_score >= {HATE_THRESHOLD}  (per Kennedy et al.)",
        f"  Aggregation    : one row per comment_id (mean across annotators)",
        f"  Total comments : {n_total}",
        f"  Positive rate  : {n_pos} / {n_total}  ({pos_rate:.1%})",
        "",
        "PIPELINES",
        f"  Ridge-Vanilla  : Ridge(OOF, vanilla feats) >= {HATE_THRESHOLD}",
        f"  Ridge-Persona  : Ridge(OOF, persona feats) >= {HATE_THRESHOLD}",
        "",
        "  " + "-" * 80,
        f"  {'pipeline':<22}{'accuracy':>9}{'F1':>9}{'precision':>11}{'recall':>9}{'AUROC':>9}{'#pos_pred':>10}",
        "  " + "-" * 80,
    ]
    if std:
        lines.append(fmt_row("Ridge-Vanilla", std["ridge"]))
    if per:
        lines.append(fmt_row("Ridge-Persona", per["ridge"]))
    lines.append("  " + "-" * 80)

    lines += ["", "CONFUSION MATRICES  (rows = true, cols = predicted; labels = {0, 1})"]
    for label, m in (("Ridge-Vanilla", std["ridge"] if std else None),
                     ("Ridge-Persona", per["ridge"] if per else None)):
        if m is None:
            continue
        cm = m["confusion_matrix"]
        lines += [
            f"  {label}",
            f"            pred=0   pred=1",
            f"    true=0  {cm['tn']:>6}  {cm['fp']:>7}",
            f"    true=1  {cm['fn']:>6}  {cm['tp']:>7}",
            "",
        ]

    lines += [
        "INTERPRETATION",
        "  - F1/accuracy depend on the threshold (0.5); AUROC is threshold-free.",
        "  - Comparing Ridge-Vanilla vs Ridge-Persona shows whether persona prompting",
        "    yields better-calibrated predictions for the IRT hate-speech score.",
        "=" * 84,
    ]

    txt_path.write_text("\n".join(lines))
    print(f"\n[report] Saved: {txt_path}")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"[report] Saved: {json_path}")


def main():
    results = {"timestamp": datetime.now().isoformat(), "model": MODEL_LABEL}
    for mode in ("standard", "persona"):
        try:
            results[mode] = evaluate_mode(mode)
        except FileNotFoundError as e:
            print(f"[skip:{mode}]  {e}")
        except Exception as e:
            print(f"[error:{mode}]  {e}")
            import traceback; traceback.print_exc()
    write_report(results)

    print("\n" + "=" * 56)
    print(f"  SUMMARY — {MODEL_LABEL}")
    print(f"  {'pipeline':<22}{'F1':>9}{'AUROC':>9}{'acc':>9}")
    print("  " + "-" * 50)
    for mode, label in (("standard", "Ridge-Vanilla"),
                        ("persona",  "Ridge-Persona")):
        if mode in results:
            m = results[mode]["ridge"]
            au = f"{m['auroc']:.4f}" if m["auroc"] is not None else " n/a"
            print(f"  {label:<22}{m['f1']:>9.4f}{au:>9}{m['accuracy']:>9.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
