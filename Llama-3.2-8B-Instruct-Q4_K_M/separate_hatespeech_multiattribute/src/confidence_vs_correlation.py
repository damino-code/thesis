import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_dataset

def load_latest_merged_results():
    print("🔍 Searching for MERGED analysis results...")
    pattern = os.path.join(config.RESULTS_FOLDER, "merged_standard_results_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No merged result files found in {config.RESULTS_FOLDER}")
        return None
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Loading predictions from: {os.path.basename(latest_file)}")
    return pd.read_csv(latest_file)

def calculate_confidence_correlation(predictions_df, human_df):
    metrics = []
    
    # Ensure human_df is indexed by comment_id for fast lookup
    if 'comment_id' in human_df.columns:
        human_df = human_df.set_index('comment_id')
    
    # Iterate through potential attributes
    # We look for attributes that exist in config AND in the predictions
    for attr in config.ATTRIBUTES:
        pred_col = attr
        conf_col = f"{attr}_confidence"
        
        if pred_col not in predictions_df.columns:
            continue
        if conf_col not in predictions_df.columns:
            print(f"⚠️ Confidence column missing for {attr}")
            # If confidence is missing, we can't plot it, but maybe we could assume something? 
            # Better to skip or handle gracefully.
            continue
        if attr not in human_df.columns:
            print(f"⚠️ Human annotations missing for {attr}")
            continue
            
        # Prepare data for this attribute
        # We merge predictions with human data on comment_id
        # predictions_df should have comment_id
        if 'comment_id' not in predictions_df.columns:
            print("❌ 'comment_id' missing from predictions file.")
            return pd.DataFrame()

        # Extract relevant columns
        sub_df = predictions_df[['comment_id', pred_col, conf_col]].copy()
        
        # Map human values
        # We only keep rows where we have both prediction and human label
        sub_df['human'] = sub_df['comment_id'].map(human_df[attr])
        
        # Drop NaNs
        valid_df = sub_df.dropna()
        
        if len(valid_df) < 10:
            print(f"⚠️ Not enough matched data points for {attr} (n={len(valid_df)})")
            continue
            
        # Calculate Correlation
        correlation = valid_df[pred_col].corr(valid_df['human'])
        
        # Calculate Average Confidence
        avg_confidence = valid_df[conf_col].mean()
        
        metrics.append({
            'attribute': attr,
            'correlation': correlation,
            'avg_confidence': avg_confidence,
            'count': len(valid_df)
        })
        
    return pd.DataFrame(metrics)

def plot_all(metrics_df):
    if metrics_df.empty:
        print("❌ No metrics to plot.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Scatter plot
    sns.scatterplot(
        data=metrics_df, 
        x='avg_confidence', 
        y='correlation', 
        s=100, 
        color='blue', 
        alpha=0.7
    )
    
    # Add labels
    for i, row in metrics_df.iterrows():
        plt.text(
            row['avg_confidence'] + 0.002,
            row['correlation'] + 0.002,
            row['attribute'].replace('_', ' '),
            fontsize=18
        )

    plt.xlabel('Average Model Confidence', fontsize=15)
    plt.ylabel('Correlation with Human Annotations', fontsize=15)
    plt.tick_params(labelsize=14)

    # Add trend line?
    # sns.regplot(data=metrics_df, x='avg_confidence', y='correlation', scatter=False, color='red', line_kws={'linestyle':'--'})

    plt.tight_layout()
    
    output_path = os.path.join(config.VISUALIZATIONS_FOLDER, f"confidence_vs_correlation_{timestamp}.png")
    plt.savefig(output_path)
    print(f"\n💾 Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Load Data
    llm_df = load_latest_merged_results()
    if llm_df is None:
        sys.exit(1)
        
    try:
        human_df = load_dataset()
    except Exception as e:
        print(f"❌ Failed to load human dataset: {e}")
        sys.exit(1)
        
    # 2. Calculate
    print("\n📊 Calculating metrics...")
    metrics_df = calculate_confidence_correlation(llm_df, human_df)
    
    # 3. Display Results
    if not metrics_df.empty:
        print("\nResults:")
        print(metrics_df.sort_values('correlation', ascending=False).to_string(index=False))
        
        # 4. Plot
        plot_all(metrics_df)
    else:
        print("❌ Could not compute metrics for any attribute.")
