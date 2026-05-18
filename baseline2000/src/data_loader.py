import pandas as pd
import config


def load_dataset(dataset_path=config.DATASET_PATH):
    df = pd.read_csv(dataset_path, on_bad_lines="skip", engine="python")
    df = df.reset_index(drop=True)
    df.index.name = "index"
    df = df.reset_index()   # 'index' column = row number, used as comment_id surrogate
    print(f"Loaded {len(df)} rows from {dataset_path}")
    return df


def get_text_column(df):
    for col in ("text", "comment", "content", "message"):
        if col in df.columns:
            return col
    return None
