import os
import pandas as pd
import glob
import sys
from datetime import datetime
import config

def get_available_personas():
    persona_results_dir = os.path.join(config.RESULTS_FOLDER, "persona_results")
    if not os.path.exists(persona_results_dir):
        return []
    
    personas = []
    # We look for directories that contain attribute subfolders
    for root, dirs, files in os.walk(persona_results_dir):
        # If this directory has any of our attributes as subdirectories, it's a persona result leaf
        if any(attr in dirs for attr in config.ATTRIBUTES):
            rel_path = os.path.relpath(root, persona_results_dir)
            personas.append(rel_path)
    return sorted(personas)

def merge_results(persona=None):
    if persona:
        print(f"Preparing to merge attribute results for persona: {persona.upper()}...")
        output_prefix = f"merged_{persona.replace(os.sep, '_')}_"
    else:
        print("Preparing to merge attribute results (Standard)...")
        output_prefix = "merged_single_attributes_"
    
    # Attributes to look for
    attributes = config.ATTRIBUTES
    
    merged_df = None
    files_found = 0
    
    for attr in attributes:
        # Search persona results under persona subfolder
        if persona:
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
        persona_tag = persona.replace(os.sep, "_") if persona else "standard"
        # Naming: merged_persona_[path_as_str]_results_[timestamp].csv
        output_filename = f"merged_persona_{persona_tag}_results_{timestamp}.csv"
        
        # Always save in a consistent place or relative to persona
        if persona:
            output_path = os.path.join(config.RESULTS_FOLDER, "persona_results", persona, output_filename)
        else:
            output_filename = f"merged_standard_results_{timestamp}.csv"
            output_path = os.path.join(config.RESULTS_FOLDER, output_filename)
        
        # Create dir if missing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully merged {files_found} attributes.")
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"\n❌ No data found to merge for {persona if persona else 'Standard'}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        merge_results(persona=sys.argv[1])
    else:
        print("\n--- Result Merger ---")
        print("0. Standard (No Persona)")
        personas = get_available_personas()
        for i, p in enumerate(personas, 1):
            print(f"{i}. Persona: {p}")
            
        choice = input("\nSelect mode (numbers e.g. 1,2,3, 'all', or 'q' to quit): ").strip().lower()
        
        if choice == '0':
            merge_results(None)
        elif choice == 'all':
            merge_results(None)
            for p in personas:
                merge_results(p)
        elif ',' in choice or choice.isdigit():
            # Handle comma separated list (e.g. 1,2,3)
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if idx == 0:
                        merge_results(None)
                    elif 1 <= idx <= len(personas):
                        merge_results(personas[idx-1])
            except ValueError:
                print("Invalid numeric selection.")
        elif choice == 'q':
            sys.exit()
