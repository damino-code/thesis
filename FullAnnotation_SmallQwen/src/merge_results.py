import os
import pandas as pd
import glob
import sys
from datetime import datetime
import config
from data_loader import load_dataset


def merge_results(use_dynamic=False):
    mode_name = "Feature-based" if use_dynamic else "Standard"
    print(f"Preparing to merge attribute results ({mode_name})...")

    attributes = config.ATTRIBUTES
    merged_df = None
    files_found = 0

    if use_dynamic:
        search_base_dir = os.path.join(config.RESULTS_FOLDER, "persona_results")
    else:
        search_base_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser")

    for attr in attributes:
        attr_dir = os.path.join(search_base_dir, attr)

        if not os.path.exists(attr_dir):
            continue

        csv_files = glob.glob(os.path.join(attr_dir, f"results_{attr}_*.csv"))
        if not csv_files:
            continue

        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"  Found latest for {attr}: {os.path.basename(latest_file)}")

        df = pd.read_csv(latest_file)
        files_found += 1

        if 'confidence' in df.columns:
            df = df.rename(columns={'confidence': f'{attr}_confidence'})
        if 'raw_response' in df.columns:
            df = df.rename(columns={'raw_response': f'{attr}_raw_response'})

        # Merge on `index` because comment_id is non-unique
        # (same comment annotated by several annotators).
        if merged_df is None:
            merged_df = df
        else:
            drop_cols = ['comment_id', 'text', 'raw_output']
            cols_to_use = ['index'] + [c for c in df.columns if c not in drop_cols and c != 'index']
            merged_df = pd.merge(merged_df, df[cols_to_use], on='index', how='outer')

    if merged_df is None:
        print("\nNo data found to merge.")
        return

    # Attach annotator_id from the source dataset using the row index.
    print("\nAttaching annotator_id from dataset...")
    ds = load_dataset()
    ds = ds.reset_index().rename(columns={'index': '_ds_index'})
    keys = ds[['_ds_index', 'comment_id', 'annotator_id']]
    merged_df = pd.merge(
        merged_df,
        keys,
        left_on='index', right_on='_ds_index',
        how='left',
        suffixes=('', '_ds'),
    )
    if 'comment_id_ds' in merged_df.columns:
        merged_df['comment_id'] = merged_df['comment_id'].fillna(merged_df['comment_id_ds'])
        merged_df = merged_df.drop(columns=['comment_id_ds'])
    merged_df = merged_df.drop(columns=['_ds_index'])

    # Reorder so the join keys are at the front.
    front = ['index', 'comment_id', 'annotator_id']
    front = [c for c in front if c in merged_df.columns]
    other = [c for c in merged_df.columns if c not in front]
    merged_df = merged_df[front + other]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_dynamic:
        output_filename = f"merged_persona_results_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, "persona_results", output_filename)
    else:
        output_filename = f"merged_standard_results_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, output_filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully merged {files_found} attributes ({len(merged_df)} rows).")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        use_dynamic = sys.argv[1].lower() == 'dynamic'
        merge_results(use_dynamic=use_dynamic)
    else:
        merge_results()
