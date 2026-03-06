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

def evaluate_single_attribute(attribute_name, persona=None):
    persona_label = f" [{persona.upper()}]" if persona else ""
    print(f"📊 EVALUATING SINGLE ATTRIBUTE: {attribute_name.upper()}{persona_label}")
    
    # 1. Locate the specific result file
    if persona:
        base_results_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", persona, attribute_name)
    else:
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
    # Ensure correct matching
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
        # Round predictions to nearest integer for classification comparison
        y_true = valid_df[human_col].round().astype(int)
        y_pred = valid_df[llm_col].round().astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        # Use 'weighted' average to account for class imbalance and multiclass/binary cases
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\n📈 Results for {attribute_name.upper()}:")
        print(f"   • Correlation: {corr:.4f}")
        print(f"   • MAE:         {mae:.4f}")
        print(f"   • Accuracy:    {accuracy:.4f}")
        print(f"   • F1 Score:    {f1:.4f} (weighted)")
        
        # 5. Save Metrics & Customize Visualization Folder
        # Create folder for this attribute
        if persona:
            attr_viz_folder = os.path.join(config.VISUALIZATIONS_FOLDER, "persona_results", persona, attribute_name)
        else:
            attr_viz_folder = os.path.join(config.VISUALIZATIONS_FOLDER, attribute_name)
        os.makedirs(attr_viz_folder, exist_ok=True)
        
        # Save Metrics to JSON
        metrics = {
            'attribute': attribute_name,
            'correlation': float(corr),
            'mae': float(mae),
            'accuracy': float(accuracy),
            'f1_score_weighted': float(f1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metrics_filename = f"metrics_{attribute_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics_path = os.path.join(attr_viz_folder, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n💾 Metrics saved to: {metrics_path}")

        # 6. Visualize
        try:
            print("\n🎨 Generating Visualizations...")
            
            # Temporarily redirect global visualization folder so plot_scatter_plots saves here
            original_viz_folder = config.VISUALIZATIONS_FOLDER
            config.VISUALIZATIONS_FOLDER = attr_viz_folder
            
            # 1. Scatter Plot
            plot_scatter_plots(eval_df, [attribute_name])
            print(f"🖼️  Scatter plot saved to {attr_viz_folder}")
            
            # Restore viz folder
            config.VISUALIZATIONS_FOLDER = original_viz_folder

        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
            config.VISUALIZATIONS_FOLDER = original_viz_folder
            
    else:
        print("⚠️ No valid numeric data pairs for evaluation.")



if __name__ == "__main__":
    # Check for Persona in arguments
    # Usage: python evaluate_single.py [attribute] [persona]
    # Or just walkthrough if no args
    
    attr_arg = sys.argv[1] if len(sys.argv) > 1 else None
    persona_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if attr_arg and attr_arg != "all":
        # Run for the specific attribute provided
        evaluate_single_attribute(attr_arg, persona=persona_arg)
    else:
        # Walkthrough or Auto-discover
        persona_base_dir = os.path.join(src_dir, 'persona_prompts')
        personas = []
        if os.path.exists(persona_base_dir):
            personas = [d for d in os.listdir(persona_base_dir) if os.path.isdir(os.path.join(persona_base_dir, d))]
        
        print("\nAvailable Personas for Evaluation:")
        print("0. Standard (No Persona)")
        for i, p in enumerate(personas, 1):
            print(f"{i}. {p.replace('_', ' ').title()}")
        
        choice = input("\nSelect persona (number) [Default 0]: ").strip()
        selected_persona = None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(personas):
                selected_persona = personas[idx-1]
        
        # 1. Determine base results dir
        if selected_persona:
            base_results_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", selected_persona)
        else:
            base_results_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser")
            
        print(f"🔍 Scanning for available results in: {base_results_dir}")
        
        if os.path.exists(base_results_dir):
            found_attrs = [d for d in os.listdir(base_results_dir) 
                          if os.path.isdir(os.path.join(base_results_dir, d))]
            
            if found_attrs:
                print(f"🎉 Found results for: {', '.join(found_attrs)}")
                for attr in found_attrs:
                    print(f"\n{'='*40}")
                    try:
                        evaluate_single_attribute(attr, persona=selected_persona)
                    except Exception as e:
                        print(f"❌ Failed to evaluate {attr}: {e}")
            else:
                print(f"❌ No attribute folders found in {base_results_dir}")
        else:
            print(f"❌ Results directory not found: {base_results_dir}")
