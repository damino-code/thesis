import sys
import os
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import config
from model_loader import load_model
from data_loader import load_dataset, get_text_column
from single_attribute_analyzer import SingleAttributeAnalyzer


def run_attribute_analysis(attribute_name, sample_size='all', use_dynamic=False, llm=None):
    print(f"STARTING ANALYSIS FOR: {attribute_name.upper()}")

    if llm is None:
        llm = load_model()

    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    text_column = get_text_column(df)
    if not text_column:
        print("Could not find text column.")
        return

    if str(sample_size).lower() in ['all', 'a']:
        df_sample = df
    else:
        try:
            df_sample = df.sample(n=int(sample_size), random_state=42)
        except Exception:
            df_sample = df

    print(f"  Analyzing {len(df_sample)} comments...")

    analyzer = SingleAttributeAnalyzer(llm, use_dynamic=use_dynamic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_dynamic:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "persona_results", attribute_name)
    else:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attribute_name)

    os.makedirs(attr_result_folder, exist_ok=True)

    # --- Batch inference with vLLM ---
    texts = [str(row[text_column]) for _, row in df_sample.iterrows()]
    comment_ids = list(df_sample.get('comment_id', df_sample.index))
    annotator_ids = list(df_sample.get('annotator_id', [None] * len(df_sample))) if use_dynamic else None

    print(f"  Building prompts and running batch inference...")
    batch_results = analyzer.batch_analyze(
        texts, attribute_name,
        comment_ids=comment_ids if use_dynamic else None,
        annotator_ids=annotator_ids,
    )

    # Attach comment_id and index to results
    results = []
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        res = batch_results[i]
        res['comment_id'] = row.get('comment_id', idx)
        res['index'] = idx
        results.append(res)

    out_df = pd.DataFrame(results)
    if 'comment_id' in out_df.columns:
        cols = list(out_df.columns)
        cols.insert(0, cols.pop(cols.index('comment_id')))
        out_df = out_df[cols]

    # Save logprobs to a separate file, then drop the column from main results
    if 'label_logprobs' in out_df.columns:
        logprobs_df = out_df[['comment_id', 'index', 'label_logprobs']].copy() if 'comment_id' in out_df.columns else out_df[['index', 'label_logprobs']].copy()
        logprobs_path = os.path.join(attr_result_folder, f"logprobs_{attribute_name}_{timestamp}.csv")
        logprobs_df.to_csv(logprobs_path, index=False)
        print(f"  Logprobs saved to: {logprobs_path}")
        out_df = out_df.drop(columns=['label_logprobs'])

    filename = f"results_{attribute_name}_{timestamp}.csv"
    save_path = os.path.join(attr_result_folder, filename)
    out_df.to_csv(save_path, index=False)
    print(f"  Finished {attribute_name}. Saved to: {save_path}")
