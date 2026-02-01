import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/storage/home/amine/thesis/Dataset/selected_comments.csv')

# Display the first few rows of the dataset
print(data.head())

# Output folder
output_dir = 'visualisation'
os.makedirs(output_dir, exist_ok=True)

# Get the list of numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 1) Distribution pages (histograms)
per_page = 10
total_pages = math.ceil(len(numerical_cols) / per_page)

for page in range(total_pages):
    start = page * per_page
    end = start + per_page
    cols = numerical_cols[start:end]

    plt.figure(figsize=(14, 3 * len(cols)))
    for i, col in enumerate(cols, 1):
        plt.subplot(len(cols), 1, i)
        sns.histplot(data[col].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attribute_distributions_page_{page + 1}.png'))
    plt.close()

# 2) Value counts per attribute (scale)
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    value_counts = data[col].value_counts().sort_index()
    sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, color='steelblue')
    plt.title(f'Value Counts for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'value_counts_{col}.png'))
    plt.close()

print(f"Saved visualizations to: {output_dir}/")