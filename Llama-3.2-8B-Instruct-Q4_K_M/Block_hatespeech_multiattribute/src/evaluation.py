import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix
from datetime import datetime
import config
from visualization import plot_correlation_matrix, plot_scatter_plots, plot_confusion_matrix, plot_correlation_bars

def load_predictions():
    print("🔍 Searching for analysis results...")
    
    # Search for both types of analysis files
    patterns = [
        # Using glob recursion or multiple patterns
        os.path.join(config.RESULTS_FOLDER, "*analysis_*.csv")
    ]
    
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    
    # Sort by modification time (newest first)
    unique_files = sorted(list(set(all_files)), key=os.path.getmtime, reverse=True)
    
    if not unique_files:
        raise FileNotFoundError(f"No prediction files found in {config.RESULTS_FOLDER}")
    
    print("\nRecent files found:")
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
    return pd.read_csv(selected_file), selected_file

def evaluate_predictions(llm_df, human_df):
    print("📊 EVALUATION: LLM vs Human Annotations")
    print("=" * 70)

    # Merge LLM predictions with human annotations
    eval_df = llm_df.copy()
    
    # Ensure correct matching (by ID or index)
    human_df_indexed = human_df.set_index('comment_id') if 'comment_id' in human_df.columns else human_df

    for idx, row in eval_df.iterrows():
        comment_id = row.get('comment_id', row.get('index'))
        
        # Support both index based or column based matching
        if 'comment_id' in human_df.columns:
             if comment_id in human_df_indexed.index:
                human_row = human_df_indexed.loc[comment_id]
                # If duplicate indices exist, human_row might be a DataFrame, take first
                if isinstance(human_row, pd.DataFrame):
                    human_row = human_row.iloc[0]
                
                for attr in config.ATTRIBUTES:
                    if attr in human_row:
                        eval_df.at[idx, f'{attr}_human'] = human_row[attr]
        else:
             # Fallback to index matching if comment_id not available (risky but in original notebook)
             if comment_id in human_df.index:
                 human_row = human_df.iloc[comment_id]
                 for attr in config.ATTRIBUTES:
                     if attr in human_row:
                         eval_df.at[idx, f'{attr}_human'] = human_row[attr]

    correlations = {}
    maes = {}

    print("📈 CORRELATIONS & MAE (LLM vs Human):")
    print("-" * 70)
    print(f"{'Attribute':<20} {'Correlation':<15} {'MAE':<10} {'Strength'}")
    print("-" * 70)

    for attr in config.ATTRIBUTES:
        if f'{attr}_human' in eval_df.columns:
            llm_vals = eval_df[attr].dropna()
            human_vals = eval_df[f'{attr}_human'].dropna()

            if len(llm_vals) > 0 and len(human_vals) > 0:
                common_idx = llm_vals.index.intersection(human_vals.index)
                if len(common_idx) > 0:
                    corr = eval_df.loc[common_idx, attr].corr(eval_df.loc[common_idx, f'{attr}_human'])
                    mae = mean_absolute_error(eval_df.loc[common_idx, f'{attr}_human'], eval_df.loc[common_idx, attr])

                    correlations[attr] = corr
                    maes[attr] = mae

                    strength = "🟢 Strong" if abs(corr) >= 0.7 else "🟡 Moderate" if abs(corr) >= 0.4 else "🟠 Weak" if abs(corr) >= 0.2 else "🔴 Very Weak"
                    print(f"{attr:<20} {corr:+.4f}          {mae:.4f}     {strength}")
    
    # Calculate overall confidence
    if 'confidence' in eval_df.columns:
        avg_conf = eval_df['confidence'].mean()
        print(f"\n🧠 Average Confidence: {avg_conf:.3f}")

    # Hate Speech Classification
    accuracy, f1, mae_hs = 0, 0, 0
    if 'hatespeech_human' in eval_df.columns:
        llm_classes = (eval_df['hatespeech'] > 0.5).astype(int)
        human_classes = (eval_df['hatespeech_human'] > 0.5).astype(int)
        
        mask = llm_classes.notna() & human_classes.notna()
        llm_classes = llm_classes[mask]
        human_classes = human_classes[mask]

        if len(llm_classes) > 0:
            accuracy = accuracy_score(human_classes, llm_classes)
            f1 = f1_score(human_classes, llm_classes, average='binary', zero_division=0)
            mae_hs = mean_absolute_error(eval_df.loc[mask, 'hatespeech_human'], eval_df.loc[mask, 'hatespeech'])
            
            print("\n🎯 HATE SPEECH CLASSIFICATION METRICS:")
            print(f"   • Accuracy:  {accuracy:.3f}")
            print(f"   • F1-Score:  {f1:.3f}")
            print(f"   • MAE:       {mae_hs:.3f}")
            
            cm = confusion_matrix(human_classes, llm_classes)
            plot_confusion_matrix(cm)

    # Save Metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        'correlations': correlations,
        'maes': maes,
        'hate_speech_metrics': {'accuracy': accuracy, 'f1': f1, 'mae': mae_hs},
        'avg_confidence': float(avg_conf) if 'confidence' in eval_df.columns else 0
    }
    
    metrics_path = os.path.join(config.RESULTS_FOLDER, f"evaluation_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Evaluation metrics saved to: {metrics_path}")

    # Plotting
    plot_correlation_bars(correlations)
    
    # Correlation Matrix Data Preparation
    numeric_cols = config.ATTRIBUTES
    llm_cols = [col for col in numeric_cols if col in eval_df.columns]
    human_cols = [f'{col}_human' for col in numeric_cols if f'{col}_human' in eval_df.columns]
    corr_data = eval_df[llm_cols + human_cols].dropna() 
    rename_map = {col: f'{col}_LLM' for col in llm_cols}
    rename_map.update({col: f'{col}_Human' for col in human_cols})
    corr_data = corr_data.rename(columns=rename_map)
    
    plot_correlation_matrix(corr_data)
    plot_scatter_plots(eval_df, numeric_cols)

if __name__ == "__main__":
    try:
        llm_df, path = load_predictions()
        human_df = pd.read_csv(config.DATASET_PATH)
        evaluate_predictions(llm_df, human_df)
    except Exception as e:
        print(f"Error: {e}")
