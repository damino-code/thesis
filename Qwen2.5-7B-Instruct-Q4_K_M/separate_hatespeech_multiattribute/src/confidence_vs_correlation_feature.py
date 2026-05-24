"""
confidence_vs_correlation_feature.py

Same analysis as confidence_vs_correlation.py but for the feature-based
(dynamic) results stored in:
  results/persona_results/merged_persona_results_*.csv

For dynamic mode the dataset can have multiple rows per comment_id (one per
annotator). Human annotations are averaged per comment_id before computing
the correlation so that the comparison is at the comment level.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_dataset


def load_latest_persona_results():
    print("🔍 Searching for merged FEATURE (persona) results...")
    pattern = os.path.join(
        config.RESULTS_FOLDER, "persona_results", "merged_persona_results_*.csv"
    )
    files = glob.glob(pattern)

    if not files:
        print(f"❌ No merged persona result files found.")
        print(f"   Expected pattern: {pattern}")
        print("   Run: python src/merge_results.py dynamic")
        return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Loading predictions from: {os.path.basename(latest_file)}")
    return pd.read_csv(latest_file)


def calculate_confidence_correlation(predictions_df, human_df):
    metrics = []

    # Average human annotations per comment_id so we compare at comment level.
    # Dynamic results can have multiple rows per comment (one per annotator).
    human_agg = (
        human_df.groupby("comment_id")[config.ATTRIBUTES]
        .mean()
    )

    if "comment_id" not in predictions_df.columns:
        print("❌ 'comment_id' missing from predictions file.")
        return pd.DataFrame()

    # Also average LLM predictions per comment_id for the same reason
    pred_cols   = [a for a in config.ATTRIBUTES if a in predictions_df.columns]
    conf_cols   = {a: f"{a}_confidence" for a in pred_cols
                   if f"{a}_confidence" in predictions_df.columns}

    group_cols  = ["comment_id"] + pred_cols + list(conf_cols.values())
    available   = [c for c in group_cols if c in predictions_df.columns]
    pred_agg    = predictions_df[available].groupby("comment_id").mean()

    for attr in config.ATTRIBUTES:
        conf_col = f"{attr}_confidence"

        if attr not in pred_agg.columns:
            print(f"⚠️  Prediction column missing for {attr} — skipping.")
            continue
        if conf_col not in pred_agg.columns:
            print(f"⚠️  Confidence column missing for {attr} — skipping.")
            continue
        if attr not in human_agg.columns:
            print(f"⚠️  Human annotation column missing for {attr} — skipping.")
            continue

        merged = pred_agg[[attr, conf_col]].join(human_agg[[attr]], rsuffix="_human")
        merged = merged.dropna()
        merged.columns = ["pred", "confidence", "human"]

        if len(merged) < 10:
            print(f"⚠️  Not enough matched rows for {attr} (n={len(merged)}) — skipping.")
            continue

        correlation    = merged["pred"].corr(merged["human"])
        avg_confidence = merged["confidence"].mean()

        metrics.append({
            "attribute":      attr,
            "correlation":    correlation,
            "avg_confidence": avg_confidence,
            "n":              len(merged),
        })

    return pd.DataFrame(metrics)


def plot_all(metrics_df):
    if metrics_df.empty:
        print("❌ No metrics to plot.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    sns.scatterplot(
        data=metrics_df,
        x="avg_confidence",
        y="correlation",
        s=120,
        color="darkorange",
        alpha=0.8,
    )

    for _, row in metrics_df.iterrows():
        plt.text(
            row["avg_confidence"] + 0.002,
            row["correlation"] + 0.002,
            row["attribute"].replace("_", " "),
            fontsize=18,
        )

    plt.xlabel("Average Model Confidence", fontsize=15)
    plt.ylabel("Correlation with Human Annotations", fontsize=15)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    output_path = os.path.join(
        config.VISUALIZATIONS_FOLDER,
        f"confidence_vs_correlation_feature_{timestamp}.png",
    )
    plt.savefig(output_path)
    print(f"\n💾 Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    llm_df = load_latest_persona_results()
    if llm_df is None:
        sys.exit(1)

    try:
        human_df = load_dataset()
    except Exception as e:
        print(f"❌ Failed to load human dataset: {e}")
        sys.exit(1)

    print("\n📊 Calculating metrics...")
    metrics_df = calculate_confidence_correlation(llm_df, human_df)

    if not metrics_df.empty:
        print("\nResults:")
        print(metrics_df.sort_values("correlation", ascending=False).to_string(index=False))
        plot_all(metrics_df)
    else:
        print("❌ Could not compute metrics for any attribute.")
