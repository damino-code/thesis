"""
Evaluate all four prompt strategies side-by-side.

For each strategy, loads the latest results CSV from results/<strategy>/ and
compares P(yes) against the IRT-derived binary label
  hate_speech_score >= 0.5
at both per-row and per-comment aggregation.

Saves: results/classification_comparison.txt and .json
"""

import json
import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
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
        return None
    return max(files, key=os.path.getmtime)


def metrics_block(y_true_bin, y_score, decision_threshold=0.5):
    y_pred_bin = (np.asarray(y_score) >= decision_threshold).astype(int)
    block = {
        "n":          int(len(y_true_bin)),
        "n_pos_gt":   int(int(np.sum(y_true_bin))),
        "n_pos_pred": int(int(np.sum(y_pred_bin))),
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

    human = (load_dataset()[JOIN_KEYS + ["hate_speech_score"]]
             .dropna(subset=["hate_speech_score"]))

    # Per-row
    per_row = None
    if "annotator_id" in pred_df.columns:
        merged = pred_df.merge(human, on=JOIN_KEYS, how="inner") \
                        .dropna(subset=["p_yes"])
        if len(merged) > 0:
            y_true_bin = (merged["hate_speech_score"] >= HATE_THRESHOLD).astype(int).values
            per_row = metrics_block(y_true_bin, merged["p_yes"].values)

    # Per-comment
    pred_uc = pred_df.groupby("comment_id", as_index=False)["p_yes"].mean()
    gt_uc = human.groupby("comment_id", as_index=False)["hate_speech_score"].mean()
    merged_uc = pred_uc.merge(gt_uc, on="comment_id", how="inner") \
                       .dropna(subset=["p_yes"])
    y_true_uc = (merged_uc["hate_speech_score"] >= HATE_THRESHOLD).astype(int).values
    per_comment = metrics_block(y_true_uc, merged_uc["p_yes"].values)

    return {
        "pred_file":  pred_path,
        "per_row":    per_row,
        "per_comment": per_comment,
    }


def fmt_row(label, m):
    auroc = f"{m['auroc']:.4f}" if m and m.get("auroc") is not None else "  n/a"
    if m is None:
        return f"  {label:<32}  (no data)"
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

    sample = next((r for r in all_results.values() if r), None)
    n_total = sample["per_comment"]["n"] if sample else 0
    n_pos   = sample["per_comment"]["n_pos_gt"] if sample else 0

    lines = [
        "=" * 92,
        "  HATE-SPEECH CLASSIFICATION PROMPT — COMPARISON",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 92,
        "",
        "TASK",
        f"  Binary target  : hate_speech_score >= {HATE_THRESHOLD}  (per Kennedy et al.)",
        f"  Decision rule  : P(yes) >= 0.5  →  predicted hate",
        f"  Total comments : {n_total}",
        f"  Positive count : {n_pos}",
        "",
        "PER-COMMENT RESULTS  (predictions averaged across annotators)",
        "  " + "-" * 90,
        f"  {'strategy':<32}{'accuracy':>9}{'F1':>9}{'precision':>11}{'recall':>9}{'AUROC':>9}{'#pos_pred':>10}",
        "  " + "-" * 90,
    ]
    for s in config.PROMPT_STRATEGIES:
        r = all_results.get(s)
        lines.append(fmt_row(s, r["per_comment"] if r else None))
    lines.append("  " + "-" * 90)

    have_per_row = any(r and r.get("per_row") for r in all_results.values())
    if have_per_row:
        lines += [
            "",
            "PER-ROW RESULTS  (joined on comment_id × annotator_id)",
            "  " + "-" * 90,
            f"  {'strategy':<32}{'accuracy':>9}{'F1':>9}{'precision':>11}{'recall':>9}{'AUROC':>9}{'#pos_pred':>10}",
            "  " + "-" * 90,
        ]
        for s in config.PROMPT_STRATEGIES:
            r = all_results.get(s)
            lines.append(fmt_row(s, r["per_row"] if r and r.get("per_row") else None))
        lines.append("  " + "-" * 90)

    lines += [
        "",
        "INTERPRETATION",
        "  - zero_shot                    : bare yes/no, no extra context",
        "  - definition                   : same plus a formal hate-speech definition",
        "  - attribute_aware_no_values    : lists 10 dimensions, model still has to assess",
        "  - attribute_aware_with_values  : substitutes the model's own previous attribute",
        "                                    predictions (low/moderate/high)",
        "  Compare to:",
        "    Ridge-Vanilla F1 ~0.685-0.697 (HateSpeechScoreFull/<model>/results)",
        "    Raw `hatespeech` vote F1 ~0.64-0.68",
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
    print(f"  SUMMARY (per-comment)")
    print(f"  {'strategy':<32}{'F1':>9}{'AUROC':>9}{'acc':>9}")
    print("  " + "-" * 60)
    for s in config.PROMPT_STRATEGIES:
        r = all_results.get(s)
        if not r:
            print(f"  {s:<32}  (no data)")
            continue
        m = r["per_comment"]
        au = f"{m['auroc']:.4f}" if m["auroc"] is not None else " n/a"
        print(f"  {s:<32}{m['f1']:>9.4f}{au:>9}{m['accuracy']:>9.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
