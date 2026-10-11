import pandas as pd
import os
import config


def load_dataset(dataset_path=config.DATASET_PATH):
    if os.path.exists(dataset_path):
        print(f"Found dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} comments")
        return df
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")


def get_text_column(df):
    for col in ('text', 'comment', 'content', 'message'):
        if col in df.columns:
            return col
    return None
