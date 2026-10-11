"""
Inter-annotator agreement (Krippendorff's alpha) per attribute, and its
relationship to LLM-human alignment.

Question: do attributes where the LLM aligns well with humans also have higher
human-human agreement?  If so, the two-cluster finding is independently
validated by the humans alone (an attribute the LLM struggles on may simply be
one humans themselves disagree on -- an intrinsically fuzzy construct -- rather
than an LLM failure).

Human raw labels : Dataset/processed_dataset_original.csv  (long format:
                   one row per (comment_id, annotator_id); 10 attributes)
LLM alignment    : evaluation_metrics_*.json  (spearman_correlations per attr)

Krippendorff's alpha is computed directly from the per-unit value coincidences,
so we never materialise the huge sparse coders x units matrix.
"""

import argparse
import json
import numpy as np
import pandas as pd
from itertools import combinations

ATTRS = ['sentiment', 'respect', 'insult', 'humiliate', 'status',
         'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech']


# ---------------------------------------------------------------------------
# Krippendorff's alpha from a coincidence matrix (efficient, missing-data safe)
# ---------------------------------------------------------------------------
def coincidence_matrix(df, comment_col, attr, values):
    """Build the VxV coincidence matrix over all comments with >= 2 raters."""
    val_index = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V))
    sub = df[[comment_col, attr]].dropna()
    for _, g in sub.groupby(comment_col)[attr]:
        m_u = len(g)
        if m_u < 2:
            continue
        # every ordered pair of ratings within the unit, weighted 1/(m_u - 1)
        idx = [val_index[v] for v in g.values]
        w = 1.0 / (m_u - 1)
        for a, b in combinations(idx, 2):
            o[a, b] += w
            o[b, a] += w
    return o


def krippendorff_alpha(o, values, metric='ordinal'):
    """Alpha from coincidence matrix `o` (VxV) over ordered `values`."""
    n_c = o.sum(axis=1)          # marginal counts per value
    n = n_c.sum()                # total pairable values
    if n < 2:
        return np.nan
    V = len(values)

    # difference function delta(c,k)^2
    delta = np.zeros((V, V))
    if metric == 'nominal':
        for a in range(V):
            for b in range(V):
                delta[a, b] = (a != b)
    elif metric == 'interval':
        vals = np.asarray(values, dtype=float)
        for a in range(V):
            for b in range(V):
                delta[a, b] = (vals[a] - vals[b]) ** 2
    elif metric == 'ordinal':
        # delta = ( sum_{g=c..k} n_g  - (n_c + n_k)/2 )^2
        for a in range(V):
            for b in range(V):
                lo, hi = min(a, b), max(a, b)
                s = n_c[lo:hi + 1].sum() - (n_c[a] + n_c[b]) / 2.0
                delta[a, b] = s ** 2
    else:
        raise ValueError(metric)

    Do = (o * delta).sum()                                   # observed disagreement
    De = (np.outer(n_c, n_c) * delta).sum() / (n - 1)        # expected disagreement
    if De == 0:
        return np.nan
    return 1.0 - (n - 1) * Do / De  # note: standard alpha = 1 - Do/De with Do,De as means


def alpha_for_attr(df, comment_col, attr, metric='ordinal'):
    values = sorted(df[attr].dropna().unique().tolist())
    o = coincidence_matrix(df, comment_col, attr, values)
    # recompute alpha with the clean mean-form to avoid confusion
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return np.nan, 0
    V = len(values)
    delta = np.zeros((V, V))
    if metric == 'ordinal':
        for a in range(V):
            for b in range(V):
                lo, hi = min(a, b), max(a, b)
                delta[a, b] = (n_c[lo:hi + 1].sum() - (n_c[a] + n_c[b]) / 2.0) ** 2
    elif metric == 'interval':
        vals = np.asarray(values, dtype=float)
        delta = (vals[:, None] - vals[None, :]) ** 2
    elif metric == 'nominal':
        delta = 1.0 - np.eye(V)
    Do = (o * delta).sum() / n
    De = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    alpha = 1 - Do / De if De else np.nan
    n_units = df.groupby(comment_col)[attr].count()
    n_units = (n_units >= 2).sum()
    return alpha, n_units


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', default='Dataset/processed_dataset.csv')
    ap.add_argument('--llm', default='FullAnnotation_SmallQwen/results/'
                    'evaluation_metrics_standard_20261011_221624.json')
    ap.add_argument('--metric', default='ordinal',
                    choices=['ordinal', 'interval', 'nominal'])
    ap.add_argument('--out', default='dataset-analysis/agreement_vs_alignment.csv')
    ap.add_argument('--plot', default='dataset-analysis/agreement_vs_alignment.png')
    ap.add_argument('--alpha-plot', default='dataset-analysis/human_agreement_per_attribute.png')
    args = ap.parse_args()

    df = pd.read_csv(args.human)
    with open(args.llm) as f:
        llm = json.load(f)
    llm_r = llm['spearman_correlations']

    rows = []
    for a in ATTRS:
        alpha, n_units = alpha_for_attr(df, 'comment_id', a, args.metric)
        r = llm_r[a]
        rows.append(dict(attribute=a, alpha=alpha, n_units=n_units,
                         llm_spearman=r, llm_alignment=abs(r)))
    res = pd.DataFrame(rows).sort_values('alpha', ascending=False)

    # relationship across the 10 attributes
    from scipy.stats import spearmanr, pearsonr
    rho, p_rho = spearmanr(res.alpha, res.llm_alignment)
    r_p, p_p = pearsonr(res.alpha, res.llm_alignment)

    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(res.to_string(index=False))
    print(f'\nHuman alpha  vs  |LLM Spearman|  across {len(res)} attributes:')
    print(f'  Spearman rho = {rho:.3f}  (p = {p_rho:.3f})')
    print(f'  Pearson  r   = {r_p:.3f}  (p = {p_p:.3f})')
    res.to_csv(args.out, index=False)
    print(f'\nsaved -> {args.out}')

    make_plot(res, rho, p_rho, args.plot)
    print(f'saved -> {args.plot}')

    make_alpha_plot(res, args.alpha_plot)
    print(f'saved -> {args.alpha_plot}')


def make_alpha_plot(res, path):
    """Bar chart of human-human agreement (alpha) per attribute only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = res.sort_values('alpha', ascending=True)
    thr = np.median(res['alpha'])
    colors = ['#2a6f97' if a >= thr else '#c1121f' for a in r['alpha']]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(r['attribute'], r['alpha'], color=colors,
                   edgecolor='white', linewidth=0.8)
    for bar, a in zip(bars, r['alpha']):
        ax.text(a + 0.008, bar.get_y() + bar.get_height() / 2,
                f'{a:.2f}', va='center', fontsize=9)

    # 0.667 = conventional threshold for tentative conclusions
    ax.axvline(0.667, color='#555', ls='--', lw=1)
    ax.text(0.667, len(r) - 0.4, '0.667\n(min. acceptable)',
            fontsize=8, color='#555', ha='center', va='bottom')

    ax.set_xlabel("Human–human agreement  (Krippendorff's α, ordinal)",
                  fontsize=11)
    ax.set_title('How much do human annotators agree, per attribute',
                 fontsize=12)
    ax.set_xlim(0, 0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='s', ls='', mfc='#2a6f97', mec='white',
               ms=10, label='high agreement (well-defined)'),
        Line2D([0], [0], marker='s', ls='', mfc='#c1121f', mec='white',
               ms=10, label='low agreement (fuzzy construct)')],
        loc='lower right', fontsize=9, frameon=True)
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def make_plot(res, rho, p_rho, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = res['alpha'].values          # human-human agreement
    y = res['llm_alignment'].values  # |LLM-human Spearman|
    # split into the two clusters at the median of human agreement
    thr = np.median(x)
    hi = x >= thr
    colors = np.where(hi, '#2a6f97', '#c1121f')  # blue = agree, red = fuzzy

    fig, ax = plt.subplots(figsize=(7, 6))
    # trend line
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color='#888', lw=1.2, ls='--', zorder=1)

    ax.scatter(x, y, c=colors, s=90, zorder=3, edgecolor='white', linewidth=0.8)
    for xi, yi, name in zip(x, y, res['attribute']):
        ax.annotate(name, (xi, yi), xytext=(5, 4),
                    textcoords='offset points', fontsize=9)

    ax.set_xlabel("Human–human agreement  (Krippendorff's α)", fontsize=11)
    ax.set_ylabel('LLM–human alignment  (|Spearman|)', fontsize=11)
    ax.set_title('Attributes humans agree on are the ones the LLM matches',
                 fontsize=12)
    ax.text(0.03, 0.95, f'Spearman ρ = {rho:.2f}  (p = {p_rho:.3f})',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc'))
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', ls='', mfc='#2a6f97', mec='white',
               ms=9, label='high agreement (well-defined)'),
        Line2D([0], [0], marker='o', ls='', mfc='#c1121f', mec='white',
               ms=9, label='low agreement (fuzzy construct)')],
        loc='lower right', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
