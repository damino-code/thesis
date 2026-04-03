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

def run_attribute_analysis(attribute_name, sample_size='all', use_dynamic=False):
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

    # 2. Logic (initialize with or without dynamic features)
    analyzer = SingleAttributeAnalyzer(llm, use_dynamic=use_dynamic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Warmup call to initialize GPU/model on first inference
    print("\n🔥 Warming up model with test call...")
    try:
        warmup_result = analyzer.analyze_attribute("This is a test comment for warming up the model.", attribute_name, comment_id=None)
        print(f"✅ Warmup complete: {warmup_result}")
    except Exception as e:
        print(f"⚠️  Warmup failed (continuing anyway): {e}")
    
    # Create attribute specific folder in results - use persona_results for dynamic mode
    if use_dynamic:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "persona_results", attribute_name)
    else:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attribute_name)
        
    os.makedirs(attr_result_folder, exist_ok=True)

    results = []
    
    for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
        comment = str(row[text_column])
        comment_id = row.get('comment_id', idx) if use_dynamic else None
        
        try:
            res = analyzer.analyze_attribute(comment, attribute_name, comment_id)
            # Always keep comment_id if present, else fallback to index
            if 'comment_id' in row:
                res['comment_id'] = row['comment_id']
            else:
                res['comment_id'] = idx
            res['index'] = idx
            
            results.append(res)
            
            # Print progress for every comment
            if i % 10 == 0 or i == 1:
                print(f"   [{i:3d}/{len(df_sample)}] ✓ Processed - {attribute_name}: {res.get(attribute_name)}")
        except Exception as e:
            print(f"   [{i:3d}/{len(df_sample)}] ❌ ERROR on item: {e}")
            import traceback
            traceback.print_exc()
            # Continue processing even if one fails
            results.append({attribute_name: None, 'confidence': 0.0, 'comment_id': row.get('comment_id', idx), 'index': idx})

    # 3. Save
    out_df = pd.DataFrame(results)
    # Ensure comment_id is the first column if present
    if 'comment_id' in out_df.columns:
        cols = list(out_df.columns)
        cols.insert(0, cols.pop(cols.index('comment_id')))
        out_df = out_df[cols]
    filename = f"results_{attribute_name}_{timestamp}.csv"
    save_path = os.path.join(attr_result_folder, filename)
    
    out_df.to_csv(save_path, index=False)
    print(f"✅ Finished {attribute_name}. Saved to: {save_path}")
