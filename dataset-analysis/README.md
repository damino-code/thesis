### Script: `visualize_attributes.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename of your dataset
data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Get the list of numerical attributes
numerical_attributes = data.select_dtypes(include=['float64', 'int64']).columns

# Create a figure to hold the subplots
plt.figure(figsize=(15, 10))

# Loop through each numerical attribute to create a subplot
for i, attribute in enumerate(numerical_attributes):
    plt.subplot(len(numerical_attributes), 1, i + 1)
    
    # Create a boxplot for the attribute
    sns.boxplot(y=data[attribute])
    plt.title(f'Boxplot of {attribute}')
    plt.xlabel(attribute)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('attribute_visualization.png')

# Show the plots
plt.show()
```

### Instructions to Run the Script

1. **Install Required Libraries**: Make sure you have the required libraries installed. You can install them using pip if you haven't already:

   ```bash
   pip install pandas matplotlib seaborn
   ```

2. **Save the Script**: Save the above script as `visualize_attributes.py` in your dataset folder.

3. **Run the Script**: Open a terminal or command prompt, navigate to the dataset folder, and run the script using Python:

   ```bash
   python visualize_attributes.py
   ```

4. **Check the Output**: The script will generate a boxplot for each numerical attribute in your dataset and save the visualization as `attribute_visualization.png` in the same folder. It will also display the plots on the screen.

### Notes

- Make sure to replace `'your_dataset.csv'` with the actual name of your dataset file.
- The script assumes that the dataset is in CSV format and contains numerical attributes. Adjust the data loading part if your dataset is in a different format.
- You can customize the visualizations further based on your specific requirements (e.g., changing plot types, colors, etc.).