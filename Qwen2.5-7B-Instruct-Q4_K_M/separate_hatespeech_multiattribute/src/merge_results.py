import os
import pandas as pd
import glob
from datetime import datetime
import config

def merge_results():
    print("Preparing to merge attribute results (Qwen)...")
    
    # Base folder for single attribute results
    base_results_dir = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser")
    
    # Attributes to look for
    attributes = [
        'sentiment', 'respect', 'insult', 'humiliate', 'status', 
        'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech'
    ]
    
    merged_df = None
    files_found = 0
    
    for attr in attributes:
        attr_dir = os.path.join(base_results_dir, attr)
        
        if not os.path.exists(attr_dir):
            print(f"⚠️  Directory not found for: {attr}")
            continue
            
        # Get list of CSVs, sort by modification time (latest first)
        csv_files = glob.glob(os.path.join(attr_dir, "*.csv"))
        if not csv_files:
            print(f"⚠️  No results found for: {attr}")
            continue
            
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"   Found latest for {attr}: {os.path.basename(latest_file)}")
        
        df = pd.read_csv(latest_file)
        files_found += 1
        
        # Rename confidence column to avoid collision/ambiguity
        if 'confidence' in df.columns:
            df = df.rename(columns={'confidence': f'{attr}_confidence'})
            
        if merged_df is None:
            # Initialize merged_df with the first attribute's data
            # Keep all columns (including text, comment_id, index)
            merged_df = df
        else:
            # Merge on comment_id
            # We want to add the specific attribute columns + its confidence
            # Common columns: comment_id, text, index, raw_output (maybe?)
            
            # Columns to merge: comment_id + unique columns
            common_cols = ['comment_id', 'index', 'text', 'raw_output']
            cols_to_use = ['comment_id'] + [c for c in df.columns if c not in common_cols]
            
            # Perform merge
            merged_df = pd.merge(merged_df, df[cols_to_use], on='comment_id', how='outer')

    if merged_df is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_single_attributes_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, output_filename)
        
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully merged {files_found} attributes.")
        print(f"💾 Saved to: {output_path}")
    else:
        print("\n❌ No data found to merge.")

if __name__ == "__main__":
    merge_results()
