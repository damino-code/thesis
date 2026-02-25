import pandas as pd
import os
import glob
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_dataset

def extract_negative_correlation_samples(attribute='respect'):
    print(f"🔍 Analyzing negative correlation samples for: {attribute}")
    
    # 1. Load human annotations
    human_df = load_dataset()
    if 'comment_id' in human_df.columns:
        human_df = human_df.set_index('comment_id')
    
    # 2. Find latest merged results
    pattern = os.path.join(config.RESULTS_FOLDER, "merged_single_attributes_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No merged result files found in {config.RESULTS_FOLDER}")
        return
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Loading predictions from: {os.path.basename(latest_file)}")
    llm_df = pd.read_csv(latest_file)
    
    # 3. Check if required columns exist
    pred_col = attribute
    conf_col = f"{attribute}_confidence"
    
    if pred_col not in llm_df.columns or conf_col not in llm_df.columns:
        print(f"❌ Required columns for {attribute} not found in results.")
        return
        
    # 4. Prepare matched dataframe
    llm_df['human_score'] = llm_df['comment_id'].map(human_df[attribute])
    
    # Drop rows without human scores
    matched_df = llm_df.dropna(subset=['human_score', pred_col]).copy()
    
    # 5. Extract samples where trend is contradictory
    # We define "negative correlation" samples as cases where 
    # the LLM score and Human score are on opposite sides of the median/mean or have large discrepancy
    # but the user specifically asked for "all the comments... where the correlation is negative".
    # Usually correlation is a global metric. Individual points contributing to negative correlation 
    # are those where (LLM - meanLLM) * (Human - meanHuman) < 0.
    
    llm_mean = matched_df[pred_col].mean()
    human_mean = matched_df['human_score'].mean()
    
    # Points that pull the correlation down (negative contribution)
    negative_contrib = matched_df[
        ((matched_df[pred_col] > llm_mean) & (matched_df['human_score'] < human_mean)) |
        ((matched_df[pred_col] < llm_mean) & (matched_df['human_score'] > human_mean))
    ].copy()
    
    # Prepare final output
    output_df = negative_contrib[[
        'comment_id', 
        'text', 
        pred_col, 
        conf_col, 
        'human_score'
    ]].rename(columns={
        pred_col: 'llm_score',
        conf_col: 'llm_confidence'
    })
    
    # Sort by absolute difference to show the most "wrong" ones first
    output_df['abs_diff'] = (output_df['llm_score'] - output_df['human_score']).abs()
    output_df = output_df.sort_values(by='abs_diff', ascending=False)
    
    # 6. Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.RESULTS_FOLDER, f"negative_correlation_{attribute}_{timestamp}.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Found {len(output_df)} samples contributing to negative correlation.")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    extract_negative_correlation_samples('respect')
