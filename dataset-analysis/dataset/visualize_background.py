import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_demographics(csv_path, output_dir):
    # Load dataset
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define attribute groups
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
    
    def plot_distributions(attrs, title_prefix, folder_name):
        target_subfolder = os.path.join(output_dir, folder_name)
        os.makedirs(target_subfolder, exist_ok=True)
        
        for attr in attrs:
            if attr not in df.columns:
                print(f"Skipping {attr}: not found in dataset.")
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Use value counts to handle categorical data nicely
            data = df[attr].dropna()
            if data.empty:
                print(f"Skipping {attr}: no data.")
                plt.close()
                continue
                
            order = data.value_counts().index
            sns.countplot(y=data, order=order, hue=data, legend=False, palette="viridis")
            
            plt.title(f'{title_prefix}: {attr.replace("_", " ").title()}')
            plt.xlabel('Count')
            plt.ylabel(attr.replace("_", " ").title())
            plt.tight_layout()
            
            save_path = os.path.join(target_subfolder, f'{attr}_distribution.png')
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()

    # Generate plots
    print("Generating Annotator distributions...")
    plot_distributions(annotator_attrs, "Annotator Background", "annotator_background")
    
    print("\nGenerating Target distributions...")
    plot_distributions(target_attrs, "Target Attribute", "target_background")

if __name__ == "__main__":
    PROJECT_ROOT = "/storage/home/amine/thesis"
    DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset/selected_comments.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset-analysis/visualisations/demographics")
    
    visualize_demographics(DATASET_PATH, OUTPUT_DIR)
