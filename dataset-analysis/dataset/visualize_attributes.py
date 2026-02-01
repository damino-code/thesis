import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/storage/home/amine/thesis/Dataset/selected_comments.csv'
data = pd.read_csv(file_path)

# Output folder
output_dir = 'visualisation'
os.makedirs(output_dir, exist_ok=True)

# Get numerical attributes
numerical_attributes = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 1) Distribution pages (histograms)
per_page = 10
total_pages = math.ceil(len(numerical_attributes) / per_page)

for page in range(total_pages):
    start = page * per_page
    end = start + per_page
    attrs = numerical_attributes[start:end]

    plt.figure(figsize=(14, 3 * len(attrs)))
    for i, attribute in enumerate(attrs, 1):
        plt.subplot(len(attrs), 1, i)
        sns.histplot(data[attribute].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attribute_distributions_page_{page + 1}.png'))
    plt.close()

# 2) Value counts per attribute (scale)
for attribute in numerical_attributes:
    plt.figure(figsize=(10, 4))
    value_counts = data[attribute].value_counts().sort_index()
    sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, color='steelblue')
    plt.title(f'Value Counts for {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'value_counts_{attribute}.png'))
    plt.close()

print(f"Saved visualizations to: {output_dir}/")