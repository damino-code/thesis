import pandas as pd
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "processed_dataset.csv"
OUTPUT_PATH = Path(__file__).parent / "processed_dataset_modified.csv"

# 0 → 2, 1 → 0, 2 → 1
REMAP = {0.0: 2.0, 1.0: 0.0, 2.0: 1.0}

df = pd.read_csv(DATASET_PATH)

print("Before:")
print(df['hatespeech'].value_counts().sort_index())

df['hatespeech'] = df['hatespeech'].map(REMAP)

print("\nAfter:")
print(df['hatespeech'].value_counts().sort_index())

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")
