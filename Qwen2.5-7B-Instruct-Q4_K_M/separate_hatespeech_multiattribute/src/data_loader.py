import pandas as pd
import os
import config

def load_dataset(dataset_path=config.DATASET_PATH):
    print("🔍 SEARCHING FOR YOUR DATASET")
    if os.path.exists(dataset_path):
        print(f"\n🎉 FOUND DATASET: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"\n✅ Loaded {len(df)} comments")
        return df
    else:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

def get_text_column(df):
    text_columns = ['text', 'comment', 'content', 'message']
    for col in text_columns:
        if col in df.columns:
            return col
    return None
