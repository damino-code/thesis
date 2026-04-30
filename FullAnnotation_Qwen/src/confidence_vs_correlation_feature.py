"""
confidence_vs_correlation_feature.py

Same analysis as confidence_vs_correlation.py but for the feature-based
(persona) results in results/persona_results/merged_persona_results_*.csv.
Joins on (comment_id, annotator_id).
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_dataset

MODEL_NAME = "Qwen2.5-72B"
GLOBAL_VIZ_ROOT = "/storage/home/amine/thesis/global_visualisation"


def load_latest_persona_results():
    print("Searching for merged PERSONA results...")
    pattern = os.path.join(
        config.RESULTS_FOLDER, "persona_results", "merged_persona_results_*.csv"
    )
    files = glob.glob(pattern)
    if not files:
        print(f"No merged persona result files found.")
        print(f"Expected pattern: {pattern}")
        print("Run: python src/merge_results.py dynamic")
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading predictions from: {os.path.basename(latest_file)}")
    return pd.read_csv(latest_file)


def calculate_confidence_correlation(predictions_df, human_df):
    join_keys = ['comment_id', 'annotator_id']
    for k in join_keys:
        if k not in predictions_df.columns:
            print(f"'{k}' missing from predictions — re-run merge_results.py dynamic")
            return pd.DataFrame()

    metrics = []
    for attr in config.ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        if attr not in predictions_df.columns or conf_col not in predictions_df.columns:
            continue
        if attr not in human_df.columns:
            continue

        sub = predictions_df[join_keys + [attr, conf_col]].copy()
        sub[attr] = pd.to_numeric(sub[attr], errors='coerce')
        sub = sub.merge(
            human_df[join_keys + [attr]].rename(columns={attr: 'human'}),
            on=join_keys,
            how='inner',
        ).dropna()

        if len(sub) < 10:
            print(f"Not enough matched rows for {attr} (n={len(sub)})")
            continue

        metrics.append({
            'attribute': attr,
            'correlation': sub[attr].corr(sub['human']),
            'avg_confidence': sub[conf_col].mean(),
            'n': len(sub),
        })

    return pd.DataFrame(metrics)


def plot_all(metrics_df, mode_dir="persona"):
    if metrics_df.empty:
        print("No metrics to plot.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(
        data=metrics_df, x='avg_confidence', y='correlation',
        s=120, color='darkorange', alpha=0.8,
    )
    for _, row in metrics_df.iterrows():
        plt.text(row['avg_confidence'] + 0.002, row['correlation'] + 0.002,
                 row['attribute'], fontsize=11)

    plt.title('Feature-based (Persona) — Model Confidence vs. Correlation with Human Annotations', fontsize=13)
    plt.xlabel('Average Model Confidence', fontsize=12)
    plt.ylabel('Correlation with Human Annotations', fontsize=12)
    plt.tight_layout()

    filename = f"confidence_vs_correlation_feature_{timestamp}.png"

    local_path = os.path.join(config.VISUALIZATIONS_FOLDER, mode_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    plt.savefig(local_path)
    print(f"Plot saved to LOCAL: {local_path}")

    if os.path.isdir(GLOBAL_VIZ_ROOT):
        try:
            global_path = os.path.join(GLOBAL_VIZ_ROOT, MODEL_NAME, mode_dir, filename)
            os.makedirs(os.path.dirname(global_path), exist_ok=True)
            plt.savefig(global_path)
            print(f"Plot saved to GLOBAL: {global_path}")
        except OSError as e:
            print(f"Could not save to global path: {e}")

    plt.close()


if __name__ == "__main__":
    llm_df = load_latest_persona_results()
    if llm_df is None:
        sys.exit(1)

    try:
        human_df = load_dataset()
    except Exception as e:
        print(f"Failed to load human dataset: {e}")
        sys.exit(1)

    print("\nCalculating metrics...")
    metrics_df = calculate_confidence_correlation(llm_df, human_df)

    if not metrics_df.empty:
        print("\nResults:")
        print(metrics_df.sort_values('correlation', ascending=False).to_string(index=False))
        plot_all(metrics_df, mode_dir="persona")
    else:
        print("Could not compute metrics for any attribute.")
