import os
import sys
import json
import math
import glob
import pandas as pd
from datetime import datetime
import config


def softmax_dict(score_dict):
    m = max(score_dict.values())
    exps = {k: math.exp(v - m) for k, v in score_dict.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def compute_metrics(label_logprobs):
    if not label_logprobs:
        return {'confidence': None, 'margin': None, 'entropy': None}

    probs = softmax_dict(label_logprobs)
    ranked = sorted(probs.values(), reverse=True)

    confidence = ranked[0]
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
    entropy = -sum(p * math.log(p) for p in ranked if p > 0)

    return {'confidence': confidence, 'margin': margin, 'entropy': entropy}


def merge_logprobs(use_dynamic=False):
    mode_name = "Feature-based" if use_dynamic else "Standard"
    print(f"Preparing to merge logprobs ({mode_name})...")

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

        csv_files = glob.glob(os.path.join(attr_dir, f"logprobs_{attr}_*.csv"))
        if not csv_files:
            continue

        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"  Found latest for {attr}: {os.path.basename(latest_file)}")

        df = pd.read_csv(latest_file)
        files_found += 1

        metrics_rows = []
        for _, row in df.iterrows():
            try:
                raw = row['label_logprobs']
                logprobs = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
                logprobs = {k: float(v) for k, v in logprobs.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                logprobs = {}
            metrics_rows.append(compute_metrics(logprobs))

        metrics_df = pd.DataFrame(metrics_rows).rename(columns={
            'confidence': f'{attr}_confidence',
            'margin':     f'{attr}_margin',
            'entropy':    f'{attr}_entropy',
        })

        id_cols = [c for c in ['comment_id', 'index'] if c in df.columns]
        attr_df = pd.concat(
            [df[id_cols].reset_index(drop=True), metrics_df.reset_index(drop=True)],
            axis=1,
        )

        if merged_df is None:
            merged_df = attr_df
        else:
            cols_to_use = ['index'] + [c for c in attr_df.columns if c not in ('comment_id', 'index')]
            merged_df = pd.merge(merged_df, attr_df[cols_to_use], on='index', how='outer')

    if merged_df is None:
        print("\nNo logprobs data found to merge.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_dynamic:
        output_filename = f"merged_persona_logprobs_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, "persona_results", output_filename)
    else:
        output_filename = f"merged_standard_logprobs_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, output_filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully merged {files_found} attributes ({len(merged_df)} rows).")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        use_dynamic = sys.argv[1].lower() == 'dynamic'
        merge_logprobs(use_dynamic=use_dynamic)
    else:
        merge_logprobs()
