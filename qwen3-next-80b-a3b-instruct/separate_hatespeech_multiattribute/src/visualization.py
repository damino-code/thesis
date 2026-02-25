import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
import config

def save_plot(filename):
    path = os.path.join(config.VISUALIZATIONS_FOLDER, filename)
    plt.savefig(path)
    print(f"💾 Plot saved to: {path}")
    plt.close()

def plot_correlation_matrix(corr_data):
    plt.figure(figsize=(16, 12))
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of LLM Predictions vs. Human Annotations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"correlation_matrix_{timestamp}.png")

def plot_scatter_plots(eval_df, numeric_cols):
    print("\n--- Scattered Dot Plots (LLM vs Human) ---\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for attr in numeric_cols:
        llm_col = attr
        human_col = f'{attr}_human'

        if llm_col in eval_df.columns and human_col in eval_df.columns:
            plt.figure(figsize=(7, 6))

            # Plot scatter points and regression line
            sns.regplot(x=human_col, y=llm_col, data=eval_df, scatter_kws={'alpha':0.6}, line_kws={'color':'green', 'label':'Regression Line'})

            # Add a diagonal line for perfect agreement
            min_val = min(eval_df[llm_col].min(), eval_df[human_col].min())
            max_val = max(eval_df[llm_col].max(), eval_df[human_col].max())
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Agreement')

            plt.title(f'{attr.capitalize()} LLM Predictions vs. Human Annotations')
            plt.xlabel(f'Human Annotation ({attr.capitalize()})')
            plt.ylabel(f'LLM Prediction ({attr.capitalize()})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_plot(f"scatter_plot_{attr}_{timestamp}.png")

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not HS Predicted', 'HS Predicted'],
                yticklabels=['Not HS Actual', 'HS Actual'])
    plt.title('Hate Speech Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"confusion_matrix_{timestamp}.png")

def plot_correlation_bars(correlations):
    if correlations:
        # Prepare data for plotting
        attrs = list(correlations.keys())
        corr_values = list(correlations.values())

        # Sort attributes by correlation value for better visualization
        sorted_indices = np.argsort(corr_values)
        sorted_attrs = [attrs[i] for i in sorted_indices]
        sorted_corr_values = [corr_values[i] for i in sorted_indices]

        # Create the bar plot
        plt.figure(figsize=(12, 7))
        bars = plt.barh(sorted_attrs, sorted_corr_values, color=['green' if c > 0 else 'red' for c in sorted_corr_values])
        plt.xlabel('Correlation Coefficient (LLM vs Human)')
        plt.ylabel('Attribute')
        plt.title('Correlation between LLM Predictions and Human Annotations (All Attributes)')
        plt.xlim([-1.0, 1.0])
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add correlation values to the bars
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.2f}',
                        va='center', ha='left' if bar.get_width() >= 0 else 'right')

        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_plot(f"correlation_bars_{timestamp}.png")

if __name__ == "__main__":
    import glob
    import pandas as pd
    from data_loader import load_dataset
    
    print("🎨 STARTING STANDALONE VISUALIZATION")
    
    # 1. Look for merged results
    merged_pattern = os.path.join(config.RESULTS_FOLDER, "merged_single_attributes_*.csv")
    merged_files = glob.glob(merged_pattern)
    
    if not merged_files:
        print(f"⚠️ No merged results found at: {merged_pattern}")
        print("💡 Run 'python src/merge_results.py' first.")
    else:
        latest_merged = max(merged_files, key=os.path.getmtime)
        print(f"📂 Loading merged results: {os.path.basename(latest_merged)}")
        llm_df = pd.read_csv(latest_merged)
        
        # 2. Load human annotations
        print("📂 Loading human annotations...")
        human_df = load_dataset()
        human_df_indexed = human_df.set_index('comment_id') if 'comment_id' in human_df.columns else human_df
        
        # 3. Merge for correlation
        eval_df = llm_df.copy()
        matched_count = 0
        correlations = {}
        
        for attr in config.ATTRIBUTES:
            if attr in eval_df.columns:
                # Map human annotation
                eval_df[f'{attr}_human'] = eval_df['comment_id'].map(human_df_indexed[attr])
                
                # Calculate correlation for this attribute
                valid = eval_df.dropna(subset=[attr, f'{attr}_human'])
                if len(valid) > 0:
                    correlations[attr] = valid[attr].corr(valid[f'{attr}_human'])
                    matched_count = len(valid) # Assumes same for all
                    
        print(f"✅ Matched {matched_count} records.")
        
        if correlations:
            print("\n📈 Generating Summary Plots...")
            
            # 1. Correlation Matrix
            # Include both predicted and human for the matrix
            corr_cols = []
            for attr in correlations.keys():
                corr_cols.extend([attr, f'{attr}_human'])
            
            plot_correlation_matrix(eval_df[corr_cols])
            
            # 2. Correlation Bars
            plot_correlation_bars(correlations)
            
            # 3. Scatter Plots (Optional: generate for top 3 correlations)
            sorted_attrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_attrs = [attr for attr, corr in sorted_attrs[:3]]
            print(f"🖼️ Generating scatter plots for top 3 correlates: {', '.join(top_attrs)}")
            plot_scatter_plots(eval_df, top_attrs)
            
            print("\n✅ Visualization complete. Check the 'visualizations' folder.")
        else:
            print("⚠️ No correlation data could be calculated. Check if attribute columns exist in the CSV.")
