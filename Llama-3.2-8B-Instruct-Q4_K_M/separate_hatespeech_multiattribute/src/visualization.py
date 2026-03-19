import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
import config

def save_plot(filename, persona=None):
    if persona:
        path = os.path.join(config.VISUALIZATIONS_FOLDER, "persona_results", persona, filename)
    else:
        path = os.path.join(config.VISUALIZATIONS_FOLDER, filename)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print(f"💾 Plot saved to: {path}")
    plt.close()

def plot_correlation_matrix(corr_data, persona=None):
    plt.figure(figsize=(16, 12))
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix ({"Standard" if not persona else persona.replace("_", " ").title()})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"correlation_matrix_{timestamp}.png", persona=persona)

def plot_scatter_plots(eval_df, numeric_cols, persona=None):
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

            plt.title(f'{attr.capitalize()} ({ "Standard" if not persona else persona.replace("_", " ").title()})')
            plt.xlabel(f'Human Annotation ({attr.capitalize()})')
            plt.ylabel(f'LLM Prediction ({attr.capitalize()})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_plot(f"scatter_plot_{attr}_{timestamp}.png", persona=persona)

def plot_confusion_matrix(cm, persona=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not HS Predicted', 'HS Predicted'],
                yticklabels=['Not HS Actual', 'HS Actual'])
    plt.title(f'Hate Speech Confusion Matrix ({"Standard" if not persona else persona.replace("_", " ").title()})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"confusion_matrix_{timestamp}.png", persona=persona)

def plot_correlation_bars(correlations, persona=None):
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
        plt.title(f'Correlation ({"Standard" if not persona else persona.replace("_", " ").title()})')
        plt.xlim([-1.0, 1.0])
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add correlation values to the bars
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.2f}',
                        va='center', ha='left' if bar.get_width() >= 0 else 'right')

        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_plot(f"correlation_bars_{timestamp}.png", persona=persona)
