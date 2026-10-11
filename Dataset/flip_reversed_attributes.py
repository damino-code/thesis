"""
The prompts asked for sentiment/respect/status with 0 = negative/disrespectful/
inferior, but the human labels use the opposite direction (0 = positive/
respectful/superior). Flip the LLM predictions (x -> 4 - x) so both scales agree.

Writes a *_flipped.csv copy of the latest merged standard + persona results for
each FullAnnotation_* model. The copy is newest by mtime, so evaluation.py and
the confidence_vs_correlation*.py scripts pick it up automatically.
Confidence columns are untouched (prob of the chosen label doesn't change).
"""
import glob
import os
import pandas as pd

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIP = ['sentiment', 'respect', 'status']
MAX = 4

for model_dir in sorted(glob.glob(os.path.join(THESIS, 'FullAnnotation_*'))):
    for pattern in ['results/merged_standard_results_*.csv',
                    'results/persona_results/merged_persona_results_*.csv']:
        files = [f for f in glob.glob(os.path.join(model_dir, pattern))
                 if not f.endswith('_flipped.csv')]
        if not files:
            continue
        src = max(files, key=os.path.getmtime)
        df = pd.read_csv(src, low_memory=False)
        for a in FLIP:
            df[a] = MAX - pd.to_numeric(df[a], errors='coerce')
        out = src[:-4] + '_flipped.csv'
        df.to_csv(out, index=False)
        print(f'{os.path.relpath(src, THESIS)} -> {os.path.basename(out)}')
