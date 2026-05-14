import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
import config

MODEL_NAME = "Qwen2.5-72B"


def save_plot(filename, mode_dir="vanilla"):
    global_viz_root = "/storage/home/amine/thesis/global_visualisation"

    local_path = os.path.join(config.VISUALIZATIONS_FOLDER, mode_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    plt.savefig(local_path)
    print(f"Plot saved to LOCAL: {local_path}")

    if os.path.isdir(global_viz_root):
        try:
            path = os.path.join(global_viz_root, MODEL_NAME, mode_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print(f"Plot saved to GLOBAL: {path}")
        except OSError as e:
            print(f"Could not save to global path: {e}")

    plt.close()


def plot_correlation_matrix(corr_data, mode_dir="vanilla"):
    title_suffix = "Persona" if mode_dir == "persona" else "Standard"
    plt.figure(figsize=(16, 12))
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix ({title_suffix})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"correlation_matrix_{timestamp}.png", mode_dir)


def plot_scatter_plots(eval_df, numeric_cols, mode_dir="vanilla"):
    title_suffix = "Persona" if mode_dir == "persona" else "Standard"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for attr in numeric_cols:
        llm_col = attr
        human_col = f'{attr}_human'

        if llm_col in eval_df.columns and human_col in eval_df.columns:
            plt.figure(figsize=(7, 6))

            sns.regplot(x=human_col, y=llm_col, data=eval_df,
                        scatter_kws={'alpha': 0.6},
                        line_kws={'color': 'green', 'label': 'Regression Line'})

            min_val = min(eval_df[llm_col].min(), eval_df[human_col].min())
            max_val = max(eval_df[llm_col].max(), eval_df[human_col].max())
            plt.plot([min_val, max_val], [min_val, max_val],
                     color='red', linestyle='--', label='Perfect Agreement')

            plt.title(f'{attr.capitalize()} ({title_suffix})')
            plt.xlabel(f'Human Annotation ({attr.capitalize()})')
            plt.ylabel(f'LLM Prediction ({attr.capitalize()})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_plot(f"scatter_plot_{attr}_{timestamp}.png", mode_dir)


def plot_confusion_matrix(cm, mode_dir="vanilla"):
    title_suffix = "Persona" if mode_dir == "persona" else "Standard"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not HS Predicted', 'HS Predicted'],
                yticklabels=['Not HS Actual', 'HS Actual'])
    plt.title(f'Hate Speech Confusion Matrix ({title_suffix})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"confusion_matrix_{timestamp}.png", mode_dir)


def plot_correlation_bars(correlations, mode_dir="vanilla"):
    title_suffix = "Persona" if mode_dir == "persona" else "Standard"
    if correlations:
        attrs = list(correlations.keys())
        corr_values = list(correlations.values())

        sorted_indices = np.argsort(corr_values)
        sorted_attrs = [attrs[i] for i in sorted_indices]
        sorted_corr_values = [corr_values[i] for i in sorted_indices]

        plt.figure(figsize=(12, 7))
        bars = plt.barh(sorted_attrs, sorted_corr_values,
                        color=['green' if c > 0 else 'red' for c in sorted_corr_values])
        plt.xlabel('Correlation Coefficient (LLM vs Human)')
        plt.ylabel('Attribute')
        plt.title(f'Correlation ({title_suffix})')
        plt.xlim([-1.0, 1.0])
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{bar.get_width():.2f}',
                     va='center', ha='left' if bar.get_width() >= 0 else 'right')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_plot(f"correlation_bars_{timestamp}.png", mode_dir)


def plot_spearman_bars(correlations, mode_dir="vanilla"):
    title_suffix = "Persona" if mode_dir == "persona" else "Standard"
    if correlations:
        attrs = list(correlations.keys())
        corr_values = list(correlations.values())

        sorted_indices = np.argsort(corr_values)
        sorted_attrs = [attrs[i] for i in sorted_indices]
        sorted_corr_values = [corr_values[i] for i in sorted_indices]

        plt.figure(figsize=(12, 7))
        bars = plt.barh(sorted_attrs, sorted_corr_values,
                        color=['steelblue' if c > 0 else 'darkorange' for c in sorted_corr_values])
        plt.xlabel('Spearman Correlation (LLM vs Human)')
        plt.ylabel('Attribute')
        plt.title(f'Spearman Correlation ({title_suffix})')
        plt.xlim([-1.0, 1.0])
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{bar.get_width():.2f}',
                     va='center', ha='left' if bar.get_width() >= 0 else 'right')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_plot(f"spearman_bars_{timestamp}.png", mode_dir)
