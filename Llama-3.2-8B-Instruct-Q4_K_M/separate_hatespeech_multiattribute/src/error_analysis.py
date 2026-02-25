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

def analyze_prediction_errors():
    print("🚀 STARTING GLOBAL ERROR ANALYSIS (Llama 3.2)")
    
    # 1. Find latest merged results
    pattern = os.path.join(config.RESULTS_FOLDER, "merged_single_attributes_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No merged result files found in {config.RESULTS_FOLDER}")
        return
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Loading predictions from: {os.path.basename(latest_file)}")
    llm_df = pd.read_csv(latest_file)
    
    # 2. Load human annotations
    human_df = load_dataset()
    if 'comment_id' in human_df.columns:
        human_df_indexed = human_df.set_index('comment_id')
    else:
        human_df_indexed = human_df

    analysis_results = []
    all_diffs = []

    # 3. Analyze each attribute
    print("\n📊 Computing Error Distribution...")
    for attr in config.ATTRIBUTES:
        if attr in llm_df.columns:
            # Map human values
            temp_df = llm_df[['comment_id', attr]].copy()
            temp_df['human'] = temp_df['comment_id'].map(human_df_indexed[attr])
            
            # Drop NaNs
            valid = temp_df.dropna()
            if len(valid) == 0:
                continue
            
            # Calculate Absolute Difference
            diff = (valid[attr] - valid['human']).abs()
            
            # Statistics
            count_total = len(diff)
            count_gt_2 = (diff > 2).sum()
            percent_gt_2 = (count_gt_2 / count_total) * 100
            mean_error = diff.mean()
            
            analysis_results.append({
                'Attribute': attr,
                'Total Sample': count_total,
                'Mean Abs Error': mean_error,
                'Err > 2 (Count)': count_gt_2,
                'Err > 2 (%)': percent_gt_2
            })
            
            # Collect all diffs for global histogram
            all_diffs.extend(diff.tolist())

    # 4. Display Summary Table
    results_df = pd.DataFrame(analysis_results)
    print("\n" + "="*60)
    print("LLM ERROR OVERVIEW (Absolute Difference vs Human)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    # 5. Visualization
    if all_diffs:
        plt.figure(figsize=(10, 6))
        sns.histplot(all_diffs, bins=20, kde=True, color='blue')
        plt.axvline(2, color='red', linestyle='--', label='Threshold (Diff > 2)')
        plt.title('Distribution of Absolute Differences (LLM vs Human) - All Attributes')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(config.VISUALIZATIONS_FOLDER, f"error_distribution_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"\n💾 Distribution plot saved to: {plot_path}")
        
        # Save summary CSV
        csv_path = os.path.join(config.RESULTS_FOLDER, f"error_analysis_summary_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"💾 Summary CSV saved to: {csv_path}")

if __name__ == "__main__":
    analyze_prediction_errors()
