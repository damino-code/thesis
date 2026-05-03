"""
confidence_vs_correlation_comparison.py

Overlays the standard (vanilla) and feature-based (persona) confidence-vs-
correlation results on a single scatter plot so the two modes can be compared
side-by-side. Output goes to visualizations/comparison/ (and the global
visualisation tree under <model>/comparison/).
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


def load_latest(pattern, label):
    files = glob.glob(pattern)
    if not files:
        print(f"No {label} merged result files found at {pattern}")
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading {label} predictions from: {os.path.basename(latest_file)}")
    return pd.read_csv(latest_file)


def calculate_confidence_correlation(predictions_df, human_df):
    join_keys = ['comment_id', 'annotator_id']
    for k in join_keys:
        if k not in predictions_df.columns:
            print(f"'{k}' missing from predictions — re-run merge_results.py")
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


def plot_comparison(vanilla_df, persona_df):
    if vanilla_df.empty and persona_df.empty:
        print("No metrics to plot for either mode.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(11, 8))
    sns.set_style("whitegrid")

    if not vanilla_df.empty:
        sns.scatterplot(
            data=vanilla_df, x='avg_confidence', y='correlation',
            s=140, color='steelblue', alpha=0.85, label='Standard (vanilla)',
            edgecolor='black', linewidth=0.5,
        )
        for _, row in vanilla_df.iterrows():
            plt.text(row['avg_confidence'] + 0.002, row['correlation'] + 0.002,
                     row['attribute'], fontsize=10, color='steelblue')

    if not persona_df.empty:
        sns.scatterplot(
            data=persona_df, x='avg_confidence', y='correlation',
            s=140, color='darkorange', alpha=0.85, label='Feature-based (persona)',
            edgecolor='black', linewidth=0.5, marker='^',
        )
        for _, row in persona_df.iterrows():
            plt.text(row['avg_confidence'] + 0.002, row['correlation'] - 0.012,
                     row['attribute'], fontsize=10, color='darkorange')

    common = set(vanilla_df['attribute']) & set(persona_df['attribute']) \
        if not vanilla_df.empty and not persona_df.empty else set()
    for attr in common:
        v = vanilla_df.loc[vanilla_df['attribute'] == attr].iloc[0]
        p = persona_df.loc[persona_df['attribute'] == attr].iloc[0]
        plt.plot(
            [v['avg_confidence'], p['avg_confidence']],
            [v['correlation'], p['correlation']],
            color='gray', linestyle='--', alpha=0.4, linewidth=1,
        )

    plt.title(
        f'{MODEL_NAME} — Model Confidence vs. Correlation with Human Annotations\n'
        'Standard vs. Feature-based (Persona)', fontsize=13,
    )
    plt.xlabel('Average Model Confidence', fontsize=12)
    plt.ylabel('Correlation with Human Annotations', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()

    mode_dir = "comparison"
    filename = f"confidence_vs_correlation_comparison_{timestamp}.png"

    local_path = os.path.join(config.VISUALIZATIONS_FOLDER, mode_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    plt.savefig(local_path, dpi=150)
    print(f"Plot saved to LOCAL: {local_path}")

    if os.path.isdir(GLOBAL_VIZ_ROOT):
        try:
            global_path = os.path.join(GLOBAL_VIZ_ROOT, MODEL_NAME, mode_dir, filename)
            os.makedirs(os.path.dirname(global_path), exist_ok=True)
            plt.savefig(global_path, dpi=150)
            print(f"Plot saved to GLOBAL: {global_path}")
        except OSError as e:
            print(f"Could not save to global path: {e}")

    plt.close()


if __name__ == "__main__":
    vanilla_pattern = os.path.join(
        config.RESULTS_FOLDER, "merged_standard_results_*.csv"
    )
    persona_pattern = os.path.join(
        config.RESULTS_FOLDER, "persona_results", "merged_persona_results_*.csv"
    )

    vanilla_preds = load_latest(vanilla_pattern, "STANDARD")
    persona_preds = load_latest(persona_pattern, "PERSONA")

    if vanilla_preds is None and persona_preds is None:
        print("No merged result files found for either mode.")
        sys.exit(1)

    try:
        human_df = load_dataset()
    except Exception as e:
        print(f"Failed to load human dataset: {e}")
        sys.exit(1)

    print("\nCalculating STANDARD metrics...")
    vanilla_metrics = (
        calculate_confidence_correlation(vanilla_preds, human_df)
        if vanilla_preds is not None else pd.DataFrame()
    )
    print("\nCalculating PERSONA metrics...")
    persona_metrics = (
        calculate_confidence_correlation(persona_preds, human_df)
        if persona_preds is not None else pd.DataFrame()
    )

    if not vanilla_metrics.empty:
        print("\nStandard results:")
        print(vanilla_metrics.sort_values('correlation', ascending=False).to_string(index=False))
    if not persona_metrics.empty:
        print("\nPersona results:")
        print(persona_metrics.sort_values('correlation', ascending=False).to_string(index=False))

    plot_comparison(vanilla_metrics, persona_metrics)
