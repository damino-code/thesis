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
    
    # 1. Choose Persona (Recursive Search)
    persona_base_dir = os.path.join(os.path.dirname(__file__), 'persona_prompts')
    all_personas = []
    if os.path.exists(persona_base_dir):
        for root, dirs, files in os.walk(persona_base_dir):
            # Check for specific leaf personas (containing json prompts)
            if any(f.endswith('.json') for f in files):
                rel_path = os.path.relpath(root, persona_base_dir)
                all_personas.append(rel_path)
    
    all_personas.sort()
    
    print("\nAvailable Personas for Evaluation:")
    print("0. Standard (No Persona)")
    for i, p in enumerate(all_personas, 1):
        print(f"{i}. {p.replace('_', ' ').title()}")
    
    p_choice = input("\nSelect persona (number) [Default 0]: ").strip()
    selected_persona = None
    if p_choice.isdigit():
        idx = int(p_choice)
        if 1 <= idx <= len(all_personas):
            selected_persona = all_personas[idx-1]

    # 2. Look for merged files in the correct directory
    if selected_persona:
        search_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", selected_persona)
        pattern = os.path.join(search_dir, f"merged_{selected_persona.replace(os.sep, '_')}_*.csv")
    else:
        search_dir = config.RESULTS_FOLDER
        pattern = os.path.join(search_dir, "merged_single_attributes_*.csv")

    files = glob.glob(pattern)
    
    # Sort by modification time (newest first)
    unique_files = sorted(list(set(files)), key=os.path.getmtime, reverse=True)
    
    if not unique_files:
        print(f"❌ No merged result files found in {search_dir}")
        print(f"🔄 Attempting to run merge script for {selected_persona if selected_persona else 'Standard'}...")
        
        try:
            from merge_results import merge_results
            merge_results(persona=selected_persona)
            # Re-search after merging
            files = glob.glob(pattern)
            unique_files = sorted(list(set(files)), key=os.path.getmtime, reverse=True)
            if not unique_files:
                raise FileNotFoundError(f"Failed to find merged file even after running merge_results.")
        except Exception as e:
            print(f"❌ Auto-merge failed: {e}")
            if selected_persona:
                print(f"💡 Tip: Run 'python src/merge_results.py {selected_persona}' manually first!")
            else:
                print("💡 Tip: Run 'python src/merge_results.py' manually first!")
            raise FileNotFoundError("No merged prediction files found.")
    
    print(f"\nRecent merged files found for {selected_persona if selected_persona else 'Standard'}:")
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
                print("Invalid choice, using most recent.")
                selected_file = unique_files[0]
        except ValueError:
            print("Invalid input, using most recent.")
            selected_file = unique_files[0]

    print(f"📂 Loading predictions from: {selected_file}")
    return pd.read_csv(selected_file), selected_file, selected_persona

def evaluate_predictions(llm_df, human_df, persona=None):
    print(f"📊 EVALUATION: LLM ({persona if persona else 'Standard'}) vs Human Annotations")
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
                plot_confusion_matrix(cm, persona=persona)
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
            plot_correlation_matrix(corr_data, persona=persona)
        else:
            print("⚠️ Not enough data for Correlation Matrix.")

    # 2. Scatter Plots
    numeric_comparison_cols = [attr for attr in config.ATTRIBUTES if attr in correlations]
    if numeric_comparison_cols:
        plot_scatter_plots(eval_df, numeric_comparison_cols, persona=persona)

    # 3. Correlation Bar Chart
    if correlations:
        plot_correlation_bars(correlations, persona=persona)


    # Save Metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        'correlations': correlations,
        'maes': maes,
        'hate_speech_metrics': {'accuracy': accuracy, 'f1': f1, 'mae': mae_hs}
    }
    
    metrics_filename = f"evaluation_metrics_merged_{timestamp}.json"
    if persona:
        metrics_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", persona)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        # Also save visualizations to persona subfolders
        viz_dir = os.path.join(os.path.dirname(config.RESULTS_FOLDER), "visualizations", "persona_results", persona)
        os.makedirs(viz_dir, exist_ok=True)
        # Tip: Update plotting code if it exists to use viz_dir
    else:
        metrics_path = os.path.join(config.RESULTS_FOLDER, metrics_filename)
        
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Evaluation metrics saved to: {metrics_path}")

def main():
    try:
        # 1. Load Merged Predictions
        result = load_predictions()
        if result is None:
            return
            
        llm_df, filename, persona = result
        
        # 2. Load Human Data
        human_df = load_dataset()
        
        # 3. Run Evaluation
        evaluate_predictions(llm_df, human_df, persona)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
