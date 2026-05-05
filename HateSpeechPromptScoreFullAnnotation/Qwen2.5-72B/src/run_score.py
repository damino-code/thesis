"""
Run hate-speech-score prompting over the full annotated dataset.

Usage (interactive):
  python src/run_score.py
    1. Vanilla   (Standard static prompt)
    2. Persona   (Annotator-specific dynamic prompt)
  + sample size or 'all'

Usage (non-interactive — used by sbatch):
  echo -e "1\\nall\\n" | python src/run_score.py     # vanilla, all rows
  echo -e "2\\nall\\n" | python src/run_score.py     # persona, all rows

Output:
  results/standard/results_hate_speech_score_<timestamp>.csv
  results/persona/results_hate_speech_score_<timestamp>.csv
Each row has:
  index, comment_id, annotator_id (if persona),
  hate_speech_score_pred, hate_speech_score_argmax,
  confidence_max_prob, confidence_entropy_certainty, n_digits_seen
"""

import os
import sys
from datetime import datetime

import pandas as pd

import config
from data_loader import load_dataset, get_text_column
from model_loader import load_model
from score_analyzer import HateSpeechScoreAnalyzer


def _interactive_choices():
    print("\nSelect Mode:")
    print("  1. Vanilla (Standard static prompt)")
    print("  2. Persona (Annotator-specific dynamic prompt)")
    mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
    use_dynamic = mode_choice == "2"

    sample_size = input("\nEnter sample size (or 'all'): ").strip()
    return use_dynamic, sample_size


def main():
    use_dynamic, sample_size = _interactive_choices()
    mode_label = "persona" if use_dynamic else "standard"
    print(f"\nMode: {mode_label.upper()}")

    print("\nLoading model...")
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

    print(f"\nAnalyzing {len(df_sample)} comments…")

    analyzer = HateSpeechScoreAnalyzer(llm, use_dynamic=use_dynamic)

    texts = [str(row[text_column]) for _, row in df_sample.iterrows()]
    comment_ids = list(df_sample.get("comment_id", df_sample.index))
    annotator_ids = list(df_sample.get("annotator_id", [None] * len(df_sample))) \
        if use_dynamic else None

    batch_results = analyzer.batch_analyze(
        texts,
        comment_ids=comment_ids if use_dynamic else None,
        annotator_ids=annotator_ids,
    )

    rows = []
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        rec = dict(batch_results[i])
        rec["comment_id"] = row.get("comment_id", idx)
        if use_dynamic:
            rec["annotator_id"] = row.get("annotator_id")
        rec["index"] = idx
        rows.append(rec)

    out = pd.DataFrame(rows)

    front = ["index", "comment_id"] + (["annotator_id"] if use_dynamic else [])
    other = [c for c in out.columns if c not in front]
    out = out[front + other]

    out_dir = os.path.join(config.RESULTS_FOLDER, mode_label)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"results_hate_speech_score_{timestamp}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} predictions → {out_path}")


if __name__ == "__main__":
    main()
