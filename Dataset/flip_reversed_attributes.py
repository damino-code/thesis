"""
The prompts asked for sentiment/respect/status with 0 = negative/disrespectful/
inferior, but the human labels use the opposite direction (0 = positive/
respectful/superior). Flip the LLM predictions (x -> 4 - x) so both scales agree.

Writes a *_flipped.csv copy of the latest merged standard + persona results for
each FullAnnotation_* model, and of the latest testRidge annotations per test
set. The copy is newest by mtime, so evaluation.py, the confidence scripts and
the ridge train/eval scripts pick it up automatically.
Confidence columns are untouched (prob of the chosen label doesn't change).
"""
import glob
import os
import pandas as pd

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIP = ['sentiment', 'respect', 'status']
MAX = 4

PATTERNS = ['FullAnnotation_*/results/merged_standard_results_*.csv',
            'FullAnnotation_*/results/persona_results/merged_persona_results_*.csv',
            'testRidge/results/merged_annotations_*.csv',
            'testRidge/results/test_2000_*/merged_annotations_*.csv']

for pattern in PATTERNS:
    by_dir = {}
    for f in glob.glob(os.path.join(THESIS, pattern)):
        if not f.endswith('_flipped.csv'):
            by_dir.setdefault(os.path.dirname(f), []).append(f)
    for files in by_dir.values():
        src = max(files)  # filename timestamp; mtime is unreliable after git checkout
        df = pd.read_csv(src, low_memory=False)
        for a in FLIP:
            df[a] = MAX - pd.to_numeric(df[a], errors='coerce')
        out = src[:-4] + '_flipped.csv'
        df.to_csv(out, index=False)
        print(f'{os.path.relpath(src, THESIS)} -> {os.path.basename(out)}')
