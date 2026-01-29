#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
import sys
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import config
from data_loader import load_dataset
from visualization import plot_scatter_plots, plot_correlation_matrix

def evaluate_single_attribute(attribute_name):
    print(f"📊 EVALUATING SINGLE ATTRIBUTE: {attribute_name.upper()}")
    
    # 1. Locate the specific result file
    base_results_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attribute_name)
    
    csv_files = glob.glob(os.path.join(base_results_dir, "*.csv"))
    if not csv_files:
        print(f"❌ No result files found for {attribute_name}")
        return

    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"📂 Loading results from: {os.path.basename(latest_file)}")
    
    llm_df = pd.read_csv(latest_file)
    
    # 2. Load Human Annotations
    human_df = load_dataset()
    
    # 3. Merge
    eval_df = llm_df.copy()
    human_df_indexed = human_df.set_index('comment_id') if 'comment_id' in human_df.columns else human_df

    matched_count = 0
    for idx, row in eval_df.iterrows():
        comment_id = row.get('comment_id', row.get('index'))
        
        if comment_id in human_df_indexed.index:
            human_row = human_df_indexed.loc[comment_id]
            if isinstance(human_row, pd.DataFrame):
                human_row = human_row.iloc[0]
            
            if attribute_name in human_row:
                eval_df.at[idx, f'{attribute_name}_human'] = human_row[attribute_name]
                matched_count += 1
    
    print(f"✅ Matched {matched_count} records with human annotations.")

    if matched_count == 0:
        print("⚠️ No matching human annotations found. Cannot calculate metrics.")
        return

    # 4. Calculate Metrics
    llm_col = attribute_name
    human_col = f'{attribute_name}_human'
    
    # Drop NaNs
    valid_df = eval_df.dropna(subset=[llm_col, human_col])
    
    if len(valid_df) > 0:
        # Correlation & MAE
        corr = valid_df[llm_col].corr(valid_df[human_col])
        mae = mean_absolute_error(valid_df[human_col], valid_df[llm_col])
        
        # Classification Metrics (Accuracy, F1)
        y_true = valid_df[human_col].round().astype(int)
        y_pred = valid_df[llm_col].round().astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"📊 Metrics for {attribute_name.upper()}:")
        print(f"   Correlation: {corr:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")

if __name__ == "__main__":
    attributes = ["sentiment", "topic", "emotion", "intent", "entity"]
    for attribute in attributes:
        evaluate_single_attribute(attribute)