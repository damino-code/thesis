import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_processed_demographics(csv_path, output_dir):
    # Load the full processed dataset
    print(f"📚 Loading full processed dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Target subfolder
    demographics_dir = os.path.join(output_dir, "processed_demographics")
    os.makedirs(demographics_dir, exist_ok=True)
    
    # Define attribute groups based on the full dataset structure
    annotator_attrs = [
        'annotator_gender', 'annotator_trans', 'annotator_educ', 
        'annotator_income', 'annotator_ideology', 'annotator_age', 
        'annotator_race', 'annotator_religion', 'annotator_sexuality'
    ]
    
    target_attrs = [
        'target_race', 'target_religion', 'target_origin', 
        'target_gender', 'target_sexuality', 'target_age', 'target_disability'
    ]
    
    # Set plotting style
    sns.set_theme(style="whitegrid")
    
    def plot_distributions(attrs, title_prefix, subfolder_name):
        target_subfolder = os.path.join(demographics_dir, subfolder_name)
        os.makedirs(target_subfolder, exist_ok=True)
        
        for attr in attrs:
            if attr not in df.columns:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Get data and handle potential NaNs
            data = df[attr].dropna()
            if data.empty:
                plt.close()
                continue
            
            # Sort by frequency for clarity
            order = data.value_counts().index
            
            # Create horizontal bar plot for better readability of long labels
            sns.countplot(y=data, order=order, hue=data, legend=False, palette="magma")
            
            # Add percentage labels
            total = len(data)
            for i, p in enumerate(plt.gca().patches):
                percentage = f'{100 * p.get_width() / total:.1f}%'
                x = p.get_width() + 0.02
                y = p.get_y() + p.get_height() / 2
                plt.gca().annotate(percentage, (x, y), va='center')

            plt.title(f'{title_prefix}: {attr.replace("_", " ").title()} (Full Processed Dataset)')
            plt.xlabel('Count')
            plt.ylabel(attr.replace("_", " ").title())
            plt.tight_layout()
            
            save_path = os.path.join(target_subfolder, f'{attr}_distribution.png')
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Saved: {save_path}")

    # Generate plots for both annotators and targets
    print("\n📊 Generating Annotator Background distributions...")
    plot_distributions(annotator_attrs, "Annotator Background", "annotator_background")
    
    print("\n📊 Generating Target Group distributions...")
    plot_distributions(target_attrs, "Target Group", "target_background")

if __name__ == "__main__":
    PROJECT_ROOT = "/storage/home/amine/thesis"
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset/processed_dataset.csv")
    VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "dataset-analysis/visualisations")
    
    visualize_processed_demographics(PROCESSED_DATA_PATH, VISUALIZATIONS_DIR)
