import pandas as pd
import os
import config

def load_dataset(dataset_path=config.DATASET_PATH):
    """
    Loads the dataset from the specified path.
    """
    print("🔍 SEARCHING FOR YOUR DATASET")
    print("=" * 60)

    if os.path.exists(dataset_path):
        print(f"\n🎉 FOUND DATASET: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"\n✅ Loaded {len(df)} comments")
        print(f"📋 Columns: {list(df.columns)}")
        return df
    else:
        print(f"\n❌ DATASET NOT FOUND AT: {dataset_path}")
        # Only fallback if specifically requested or interactive, otherwise raise error
        # For script usage, failing fast is usually better unless we want to search
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

def get_text_column(df):
    """
    Identifies the text column in the dataframe.
    """
    text_columns = ['text', 'comment', 'content', 'message']
    for col in text_columns:
        if col in df.columns:
            return col
    return None
