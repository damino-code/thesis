"""
Merge per-attribute annotation results into a single CSV.
Also attaches the original text and label columns from test_2000.csv.
"""

import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import config


def latest_file(attr_dir, attribute):
    files = glob.glob(str(attr_dir / f"results_{attribute}_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    df_test = pd.read_csv(config.TEST_CSV, on_bad_lines="skip", engine="python")
    df_test = df_test.reset_index(drop=True)
    df_test.index.name = "index"
    df_test = df_test.reset_index()  # 'index' column = row number

    merged = df_test[["index", "text", "label"]].copy()

    files_found = 0
    for attr in config.ATTRIBUTES:
        attr_dir = config.ATTR_RESULTS_DIR / attr
        path = latest_file(attr_dir, attr)
        if path is None:
            print(f"[skip] No results for {attr}")
            continue
        print(f"[merge] {attr}: {Path(path).name}")
        attr_df = pd.read_csv(path)
        attr_df = attr_df[["index", attr, f"{attr}_confidence"]]
        merged = merged.merge(attr_df, on="index", how="left")
        files_found += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.RESULTS_DIR / f"merged_annotations_{timestamp}.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged {files_found} attributes ({len(merged)} rows)")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
