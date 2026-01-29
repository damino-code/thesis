import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import config
from model_loader import download_model, load_model
from data_loader import load_dataset, get_text_column
from single_attribute_analyzer import SingleAttributeAnalyzer

def run_attribute_analysis(attribute_name, sample_size='all'):
    print(f"🚀 STARTING ANALYSIS FOR: {attribute_name.upper()}")
    
    # 1. Setup
    model_path = download_model()
    llm = load_model(model_path)
    
    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    text_column = get_text_column(df)
    if not text_column:
        print("❌ Could not find text column automatically.")
        return

    # Sample Data
    if str(sample_size).lower() in ['all', 'a']:
        df_sample = df
    else:
        try:
            df_sample = df.sample(n=int(sample_size), random_state=42)
        except:
            df_sample = df

    print(f"   Analyzing {len(df_sample)} comments...")

    # 2. Logic
    analyzer = SingleAttributeAnalyzer(llm)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create attribute specific folder in results
    attr_result_folder = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attribute_name)
    os.makedirs(attr_result_folder, exist_ok=True)

    results = []
    
    for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
        comment = str(row[text_column])
        
        try:
            res = analyzer.analyze_attribute(comment, attribute_name)
            res['comment_id'] = row.get('comment_id', idx)
            res['index'] = idx
            
            # Request confidence (it's already in the logic? separate_hatespeech implementation check needed)
            # Inspecting user's SingleAttributeAnalyzer logic from previous turns... 
            # It uses `logprobs=1` and likely calculates confidence if updated.
            # Just relying on whatever analyze_attribute provides.
            
            results.append(res)
        except Exception as e:
            print(f"   Error on item {i}: {e}")

        if i % 10 == 0:
            print(f"   Processed {i}/{len(df_sample)}...")

    # 3. Save
    out_df = pd.DataFrame(results)
    filename = f"results_{attribute_name}_{timestamp}.csv"
    save_path = os.path.join(attr_result_folder, filename)
    
    out_df.to_csv(save_path, index=False)
    print(f"✅ Finished {attribute_name}. Saved to: {save_path}")
