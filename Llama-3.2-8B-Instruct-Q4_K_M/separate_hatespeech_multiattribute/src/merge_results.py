import os
import pandas as pd
import glob
import sys
from datetime import datetime
import config

def merge_results():
    print("Preparing to merge attribute results (Standard)...")
    
    # Attributes to look for
    attributes = config.ATTRIBUTES
    
    merged_df = None
    files_found = 0
    
    for attr in attributes:
        # Search results under standard folder
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
        
        # Rename confidence column to avoid collision/ambiguity
        if 'confidence' in df.columns:
            df = df.rename(columns={'confidence': f'{attr}_confidence'})
            
        if merged_df is None:
            merged_df = df
        else:
            common_cols = ['comment_id', 'index', 'text', 'raw_output']
            cols_to_use = ['comment_id'] + [c for c in df.columns if c not in common_cols]
            merged_df = pd.merge(merged_df, df[cols_to_use], on='comment_id', how='outer')

    if merged_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_standard_results_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, output_filename)
        
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully merged {files_found} attributes.")
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"\n❌ No data found to merge.")

if __name__ == "__main__":
    merge_results()
