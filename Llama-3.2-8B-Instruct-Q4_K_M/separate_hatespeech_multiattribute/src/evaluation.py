import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix
from datetime import datetime
import config
from visualization import plot_correlation_matrix, plot_scatter_plots, plot_confusion_matrix, plot_correlation_bars
from data_loader import load_dataset

def load_predictions():
    print("🔍 Searching for MERGED analysis results...")
    
    # 1. Choose which results to evaluate
    print("\nSelect Results to Evaluate:")
    print("1. Standard (Vanilla prompts)")
    print("2. Persona (Feature-based dynamic prompts)")
    
    mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
    
    if mode_choice == '2':
        pattern = os.path.join(config.RESULTS_FOLDER, "persona_results", "merged_persona_results_*.csv")
        mode_name = "Persona"
    else:
        pattern = os.path.join(config.RESULTS_FOLDER, "merged_standard_results_*.csv")
        mode_name = "Standard"
    
    print(f"\n✓ Selected: {mode_name} results")
    
    files = glob.glob(pattern)
    unique_files = sorted(list(set(files)), key=os.path.getmtime, reverse=True)
    
    if not unique_files:
        print(f"❌ No merged {mode_name.lower()} result files found.")
        print(f"🔄 Attempting to run merge script...")
        
        try:
            from merge_results import merge_results
            use_dynamic = (mode_choice == '2')
            merge_results(use_dynamic=use_dynamic)
            # Re-search after merging
            files = glob.glob(pattern)
            unique_files = sorted(list(set(files)), key=os.path.getmtime, reverse=True)
            if not unique_files:
                raise FileNotFoundError(f"Failed to find merged file even after running merge_results.")
        except Exception as e:
            print(f"❌ Auto-merge failed: {e}")
            raise FileNotFoundError("No merged prediction files found.")
    
    print(f"\nRecent merged files found:")
    for i, f in enumerate(unique_files[:5], 1):
        timestamp = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i}. {os.path.basename(f)} ({timestamp})")
        
    choice = input("\nSelect file number to evaluate (default 1): ").strip()
    if not choice:
        selected_file = unique_files[0]
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(unique_files):
                selected_file = unique_files[idx]
            else:
                selected_file = unique_files[0]
        except ValueError:
            selected_file = unique_files[0]

    print(f"📂 Loading predictions from: {selected_file}")
    return pd.read_csv(selected_file), selected_file

def evaluate_predictions(llm_df, human_df, mode_dir="vanilla"):
    print(f"📈 EVALUATION: LLM vs Human Annotations")
    print("=" * 70)

    # Merge LLM predictions with human annotations
    eval_df = llm_df.copy()
    
    # Ensure matching by comment_id
    # human_df might need simple indexing
    human_df_indexed = human_df.set_index('comment_id') if 'comment_id' in human_df.columns else human_df

    # We augment eval_df with human columns
    matched_count = 0
    for idx, row in eval_df.iterrows():
        comment_id = row.get('comment_id', row.get('index'))
        
        if comment_id in human_df_indexed.index:
            human_row = human_df_indexed.loc[comment_id]
            if isinstance(human_row, pd.DataFrame):
                human_row = human_row.iloc[0]
            
            for attr in config.ATTRIBUTES:
                if attr in human_row:
                    eval_df.at[idx, f'{attr}_human'] = human_row[attr]
            matched_count += 1
            
    print(f"✅ Matched {matched_count} comments for evaluation.")

    correlations = {}
    maes = {}

    print("\n📈 CORRELATIONS & MAE (LLM vs Human):")
    print("-" * 70)
    print(f"{'Attribute':<20} {'Correlation':<15} {'MAE':<10} {'Strength'}")
    print("-" * 70)

    for attr in config.ATTRIBUTES:
        # Check if we have both LLM prediction and Human key for this attribute
        if attr in eval_df.columns and f'{attr}_human' in eval_df.columns:
            
            # Clean numeric data
            valid_data = eval_df.dropna(subset=[attr, f'{attr}_human'])
            
            llm_vals = valid_data[attr]
            human_vals = valid_data[f'{attr}_human']

            if len(llm_vals) > 0:
                corr = llm_vals.corr(human_vals)
                mae = mean_absolute_error(human_vals, llm_vals)

                correlations[attr] = corr
                maes[attr] = mae

                strength = "🟢 Strong" if abs(corr) >= 0.7 else "🟡 Moderate" if abs(corr) >= 0.4 else "🟠 Weak" if abs(corr) >= 0.2 else "🔴 Very Weak"
                print(f"{attr:<20} {corr:+.4f}          {mae:.4f}     {strength}")

    # --- Hate Speech Classification Metrics ---
    accuracy, f1, mae_hs = 0, 0, 0
    if 'hatespeech' in eval_df.columns and 'hatespeech_human' in eval_df.columns:
        hs_data = eval_df.dropna(subset=['hatespeech', 'hatespeech_human'])
        if len(hs_data) > 0:
            llm_classes = (hs_data['hatespeech'] > 0.5).astype(int)
            human_classes = (hs_data['hatespeech_human'] > 0.5).astype(int)
            
            accuracy = accuracy_score(human_classes, llm_classes)
            f1 = f1_score(human_classes, llm_classes, average='binary', zero_division=0)
            mae_hs = mean_absolute_error(hs_data['hatespeech_human'], hs_data['hatespeech'])
            
            print("\n🎯 HATE SPEECH CLASSIFICATION METRICS:")
            print(f"   • Accuracy:  {accuracy:.3f}")
            print(f"   • F1-Score:  {f1:.3f}")
            print(f"   • MAE:       {mae_hs:.3f}")
            
            cm = confusion_matrix(human_classes, llm_classes)
            # Visualize Confusion Matrix
            try:
                plot_confusion_matrix(cm, mode_dir=mode_dir)
            except Exception as e:
                print(f"⚠️ Failed to plot confusion matrix: {e}")

    # --- Visualizations ---
    print("\n🎨 Generating Visualizations...")
    
    # 1. Correlation Matrix of all attributes (Human vs LLM)
    cols_to_plot = []
    for attr in config.ATTRIBUTES:
        if attr in eval_df.columns: cols_to_plot.append(attr)
        if f'{attr}_human' in eval_df.columns: cols_to_plot.append(f'{attr}_human')
    
    if cols_to_plot:
        corr_data = eval_df[cols_to_plot].dropna()
        if len(corr_data) > 0:
            plot_correlation_matrix(corr_data, mode_dir=mode_dir)
        else:
            print("⚠️ Not enough data for Correlation Matrix.")

    # 2. Scatter Plots
    numeric_comparison_cols = [attr for attr in config.ATTRIBUTES if attr in correlations]
    if numeric_comparison_cols:
        plot_scatter_plots(eval_df, numeric_comparison_cols, mode_dir=mode_dir)

    # 3. Correlation Bar Chart
    if correlations:
        plot_correlation_bars(correlations, mode_dir=mode_dir)


    # Save Metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        'correlations': correlations,
        'maes': maes,
        'hate_speech_metrics': {'accuracy': accuracy, 'f1': f1, 'mae': mae_hs}
    }
    
    # Global Visualization folder setup
    global_viz_root = "/storage/home/amine/thesis/global_visualisation"
    model_name = "Llama-3.2-8B" # hardcoded for this folder
    
    mode_filename_part = "persona" if mode_dir == "persona" else "standard"
    metrics_filename = f"evaluation_metrics_{mode_filename_part}_{timestamp}.json"
    
    # Always save locally
    local_path = os.path.join(config.RESULTS_FOLDER, metrics_filename)
    with open(local_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Evaluation metrics saved to LOCAL: {local_path}")

    # Save to global path only when the server filesystem is reachable
    if os.path.isdir(global_viz_root):
        try:
            global_dir = os.path.join(global_viz_root, model_name, mode_dir)
            os.makedirs(global_dir, exist_ok=True)
            global_path = os.path.join(global_dir, metrics_filename)
            with open(global_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"💾 Evaluation metrics saved to GLOBAL: {global_path}")
        except OSError as e:
            print(f"⚠️  Could not save to global path: {e}")

def main():
    try:
        # 1. Load Merged Predictions
        result = load_predictions()
        if result is None:
            return
            
        llm_df, filename = result
        
        # Determine mode from filename
        mode_dir = "persona" if "persona" in filename.lower() else "vanilla"
        
        # 2. Load Human Data
        human_df = load_dataset()
        
        # 3. Run Evaluation
        evaluate_predictions(llm_df, human_df, mode_dir=mode_dir)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
