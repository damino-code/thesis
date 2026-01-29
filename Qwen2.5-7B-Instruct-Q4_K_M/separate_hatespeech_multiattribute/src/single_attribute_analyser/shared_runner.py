import os
import pandas as pd
import re
from typing import Any

def analyze_attribute(text: str, attribute_name: str) -> dict:
    """
    Analyze an attribute from a text.
    """
    # Placeholder for actual analysis logic
    return {"text": text, "attribute": attribute_name}

def analyze_attributes(df_sample: pd.DataFrame, text_column: str, attribute_name: str) -> pd.DataFrame:
    """
    Analyze attributes from a DataFrame.
    """
    results = []
    for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
        comment = str(row[text_column])
        try:
            res = analyze_attribute(comment, attribute_name)
            # Always keep comment_id if present, else fallback to index
            if 'comment_id' in row:
                res['comment_id'] = row['comment_id']
            else:
                res['comment_id'] = idx
            res['index'] = idx
            results.append(res)
        except Exception as e:
            print(f"   Error on item {i}: {e}")
        if i % 10 == 0:
            print(f"   Processed {i}/{len(df_sample)}...")
    return pd.DataFrame(results)

def save_results(results: pd.DataFrame, attribute_name: str, save_path: str) -> None:
    """
    Save results to a CSV file.
    """
    out_df = pd.DataFrame(results)
    # Ensure comment_id is the first column if present
    if 'comment_id' in out_df.columns:
        cols = list(out_df.columns)
        cols.insert(0, cols.pop(cols.index('comment_id')))
        out_df = out_df[cols]
    filename = f"results_{attribute_name}.csv"
    out_df.to_csv(save_path, index=False)
    print(f"✅ Finished {attribute_name}. Saved to: {save_path}")