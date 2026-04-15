import sys
import os
import pandas as pd
import glob
from datetime import datetime
import config

def merge_results(persona=None, use_dynamic=False):
    """Merge per-attribute result CSVs into a single file.

    Parameters
    ----------
    persona : str or None
        Static persona path (legacy mode).  When set, reads from
        ``results/persona_results/{persona}/{attribute}/`` and saves to
        ``results/persona_results/{persona}/``.
    use_dynamic : bool
        Feature-based (dynamic) mode.  Reads from the flat
        ``results/persona_results/{attribute}/`` folders (same location the
        updated shared_runner writes to) and saves as
        ``merged_persona_results_TIMESTAMP.csv`` in
        ``results/persona_results/``.
    """
    if use_dynamic:
        print("Preparing to merge attribute results (Feature-based dynamic)...")
        search_base_dir = os.path.join(config.RESULTS_FOLDER, "persona_results")
        output_prefix = "merged_persona_results_"
        output_dir    = os.path.join(config.RESULTS_FOLDER, "persona_results")
    elif persona:
        print(f"Preparing to merge attribute results for persona: {persona.upper()}...")
        search_base_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", persona)
        output_prefix = f"merged_{persona.replace(os.sep, '_')}_"
        output_dir    = os.path.join(config.RESULTS_FOLDER, "persona_results", persona)
    else:
        print("Preparing to merge attribute results (Standard)...")
        search_base_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser")
        output_prefix = "merged_single_attributes_"
        output_dir    = config.RESULTS_FOLDER

    # Attributes to look for
    attributes = config.ATTRIBUTES

    merged_df = None
    files_found = 0

    for attr in attributes:
        if use_dynamic:
            attr_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", attr)
        elif persona:
            attr_dir = os.path.join(config.RESULTS_FOLDER, "persona_results", persona, attr)
        else:
            attr_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attr)

        if not os.path.exists(attr_dir):
            continue

        csv_files = glob.glob(os.path.join(attr_dir, f"results_{attr}_*.csv"))
        if not csv_files:
            continue

        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"   Found latest for {attr}: {os.path.basename(latest_file)}")

        df = pd.read_csv(latest_file)
        files_found += 1

        # Rename per-attribute columns to avoid collisions during merge
        if 'confidence' in df.columns:
            df = df.rename(columns={'confidence': f'{attr}_confidence'})
        if 'raw_response' in df.columns:
            df = df.rename(columns={'raw_response': f'{attr}_raw_response'})

        if merged_df is None:
            merged_df = df
        else:
            common_cols = ['comment_id', 'index', 'text', 'raw_output']
            cols_to_use = ['comment_id'] + [c for c in df.columns if c not in common_cols]
            merged_df = pd.merge(merged_df, df[cols_to_use], on='comment_id', how='outer')

    if merged_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_prefix}{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)

        os.makedirs(output_dir, exist_ok=True)

        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully merged {files_found} attributes.")
        print(f"💾 Saved to: {output_path}")
    else:
        print("\n❌ No data found to merge.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'dynamic':
        merge_results(use_dynamic=True)
    elif len(sys.argv) > 1:
        merge_results(persona=sys.argv[1])
    else:
        merge_results()
