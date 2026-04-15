"""
evaluate_feature.py

Standalone evaluation script for feature-based (dynamic) results.
Automatically loads the latest merged_persona_results_*.csv and produces:
  - Per-attribute correlations and MAE
  - Hate-speech classification metrics (accuracy, F1, MAE)
  - Visualisations (scatter plots, correlation bars, confusion matrix)
  - evaluation_metrics_persona_feature_<timestamp>.json  (local + global)

No interactive prompts — run directly:
    python src/evaluate_feature.py
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_dataset
from evaluation import evaluate_predictions   # reuse all metric/plot logic


def load_latest_persona_results():
    pattern = os.path.join(
        config.RESULTS_FOLDER, "persona_results", "merged_persona_results_*.csv"
    )
    files = glob.glob(pattern)

    if not files:
        print(f"❌ No merged persona result files found.")
        print(f"   Expected pattern: {pattern}")
        print("   Run first:  python src/merge_results.py dynamic")
        return None, None

    import pandas as pd
    latest = max(files, key=os.path.getmtime)
    print(f"📂 Loading: {os.path.basename(latest)}")
    return pd.read_csv(latest), latest


def main():
    print("🚀 FEATURE (PERSONA) EVALUATION")
    print("=" * 60)

    # 1. Load feature predictions
    llm_df, filepath = load_latest_persona_results()
    if llm_df is None:
        sys.exit(1)

    # 2. Load human annotations
    try:
        human_df = load_dataset()
    except Exception as e:
        print(f"❌ Failed to load human dataset: {e}")
        sys.exit(1)

    # 3. Run evaluation — persona="feature" routes JSON/plots to feature subfolder
    evaluate_predictions(llm_df, human_df, persona="feature")


if __name__ == "__main__":
    main()
