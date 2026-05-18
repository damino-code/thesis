"""
Run yes/no hate-speech classification on test_2000.csv for one or more
prompt strategies.

Usage (sbatch):  echo -e "2\nall\n" | python src/run_classification.py
Interactive:     python src/run_classification.py
"""

import os
import sys
from datetime import datetime

import pandas as pd

import config
from data_loader import load_dataset, get_text_column
from model_loader import load_model
from classifier_analyzer import HateSpeechClassifier

STRATEGIES = config.PROMPT_STRATEGIES


def _interactive_choices():
    print("\nAvailable prompt strategies:")
    for i, s in enumerate(STRATEGIES, 1):
        print(f"  {i}. {s}")
    print(f"  {len(STRATEGIES) + 1}. all")
    raw = input("\nChoose (e.g. '1,3' or 'all') [Default: all]: ").strip()
    if not raw or raw.lower() == "all":
        chosen = list(STRATEGIES)
    else:
        idxs = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        chosen = [STRATEGIES[i - 1] for i in idxs if 1 <= i <= len(STRATEGIES)]
        if not chosen:
            chosen = list(STRATEGIES)

    sample_size = input("\nEnter sample size (or 'all'): ").strip()
    return chosen, sample_size


def run_one(strategy, df_sample, text_column, llm):
    print("\n" + "#" * 70)
    print(f"#  STRATEGY: {strategy}")
    print("#" * 70)

    classifier = HateSpeechClassifier(llm, strategy)

    texts        = [str(row[text_column]) for _, row in df_sample.iterrows()]
    # Use 'index' column as comment_id surrogate (row number in test_2000.csv)
    comment_ids  = list(df_sample["index"])
    annotator_ids = [None] * len(df_sample)

    print(f"  Running batch on {len(texts)} comments…")
    batch = classifier.batch_analyze(
        texts,
        comment_ids=comment_ids,
        annotator_ids=annotator_ids,
    )

    rows = []
    for i, (_, src_row) in enumerate(df_sample.iterrows()):
        rec = dict(batch[i])
        rec["index"]    = int(src_row["index"])
        rec["label"]    = src_row.get("label")
        rec["split"]    = src_row.get("split")
        rows.append(rec)

    out = pd.DataFrame(rows)
    front = ["index", "label", "split"]
    other = [c for c in out.columns if c not in front]
    out = out[front + other]

    out_dir = os.path.join(config.RESULTS_FOLDER, strategy)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"results_{strategy}_{timestamp}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} predictions → {out_path}")
    return out_path


def main():
    chosen, sample_size = _interactive_choices()
    print(f"\nStrategies: {chosen}")

    print("\nLoading model (once for all strategies)…")
    llm = load_model()

    df = load_dataset()
    text_column = get_text_column(df)
    if text_column is None:
        print("Could not find text column.")
        sys.exit(1)

    if str(sample_size).lower() in ("all", "a", ""):
        df_sample = df
    else:
        try:
            df_sample = df.sample(n=int(sample_size), random_state=42)
        except Exception:
            df_sample = df

    for strategy in chosen:
        try:
            run_one(strategy, df_sample, text_column, llm)
        except Exception as e:
            print(f"[error:{strategy}] {e}")
            import traceback; traceback.print_exc()

    print("\nAll strategies done.")


if __name__ == "__main__":
    main()
