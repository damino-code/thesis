"""
Ridge training on test_2000.csv annotations.

Loads the merged Llama-3.1-70B vanilla annotations from testRdige/results/,
uses label (0/1) as target, trains Ridge with 5-fold CV, saves weights and
OOF predictions.

Features: attr × attr_confidence  for each of 10 attributes  (same as
          HateSpeechScoreFull pipeline).
Target  : binary label column (0 = not hate, 1 = hate).
"""

import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
THESIS_DIR    = BASE_DIR.parent
ANNOTATIONS_DIR = THESIS_DIR / "testRdige" / "results"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTRIBUTES = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
]

ALPHAS       = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS     = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_annotations():
    files = glob.glob(str(ANNOTATIONS_DIR / "merged_annotations_*.csv"))
    if not files:
        raise FileNotFoundError(f"No merged_annotations_*.csv in {ANNOTATIONS_DIR}")
    path = max(files, key=os.path.getmtime)
    print(f"[data]  Annotations: {Path(path).name}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"[data]  {len(df)} samples  |  hate={df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    return df


def compute_features(df):
    feats = {}
    for attr in ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        a = pd.to_numeric(df.get(attr, 0), errors="coerce").fillna(0).values
        c = pd.to_numeric(df.get(conf_col, 0), errors="coerce").fillna(0).values
        feats[attr] = a * c
    return pd.DataFrame(feats, index=df.index)

# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def fold_metrics(y_true, y_pred):
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cv_ridge(X, y, alphas, kf):
    results = {a: [] for a in alphas}
    for tr_idx, te_idx in kf.split(X):
        for alpha in alphas:
            m = Ridge(alpha=alpha, fit_intercept=True).fit(X[tr_idx], y[tr_idx])
            results[alpha].append(fold_metrics(y[te_idx], m.predict(X[te_idx])))
    return results


def best_alpha(cv_results):
    return max(cv_results, key=lambda a: np.mean([f["r2"] for f in cv_results[a]]))


def summarise(folds):
    return {
        metric: {
            "mean": float(np.mean([f[metric] for f in folds])),
            "std":  float(np.std( [f[metric] for f in folds])),
            "folds": [f[metric] for f in folds],
        }
        for metric in ("r2", "mae", "rmse")
    }


def correlations(y_true, y_pred):
    r, p1 = pearsonr(y_true, y_pred)
    rho, p2 = spearmanr(y_true, y_pred)
    return {
        "pearson":  {"r": float(r),   "p_value": float(p1)},
        "spearman": {"rho": float(rho), "p_value": float(p2)},
    }

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cv(cv_results, alpha, out_dir):
    r2_means = [np.mean([f["r2"] for f in cv_results[a]]) for a in ALPHAS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ALPHAS, r2_means, "o-")
    ax.axvline(alpha, color="red", linestyle="--", label=f"best α={alpha}")
    ax.set_xlabel("alpha"); ax.set_ylabel("CV R²"); ax.set_title("Ridge CV — alpha selection")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "cv_alpha.png", dpi=120)
    plt.close(fig)


def plot_predictions(y_true, y_pred, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("True label"); ax.set_ylabel("Ridge prediction")
    ax.set_title("OOF predictions vs ground truth")
    fig.tight_layout()
    fig.savefig(out_dir / "oof_predictions.png", dpi=120)
    plt.close(fig)


def plot_weights(coef, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in coef]
    ax.barh(ATTRIBUTES, coef, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge coefficient"); ax.set_title("Ridge weights (label target)")
    fig.tight_layout()
    fig.savefig(out_dir / "weights.png", dpi=120)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#" * 60)
    print("#  trainRidge2000 — Ridge on test_2000 annotations")
    print("#" * 60)

    df = load_annotations()
    X  = compute_features(df).values
    y  = df["label"].astype(float).values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[cv]  Running {N_SPLITS}-fold CV over {len(ALPHAS)} alphas …")
    cv_results = cv_ridge(X, y, ALPHAS, kf)
    alpha       = best_alpha(cv_results)
    summary     = summarise(cv_results[alpha])

    print(f"[cv]  Best α = {alpha}")
    print(f"[cv]  R²  = {summary['r2']['mean']:.4f} ± {summary['r2']['std']:.4f}")
    print(f"[cv]  MAE = {summary['mae']['mean']:.4f} ± {summary['mae']['std']:.4f}")
    print(f"[cv]  RMSE= {summary['rmse']['mean']:.4f} ± {summary['rmse']['std']:.4f}")

    # OOF predictions
    oof = np.zeros_like(y)
    fold_assign = np.full(len(y), -1, dtype=int)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict(X[te_idx])
        fold_assign[te_idx] = fold_idx

    oof_df = pd.DataFrame({
        "index":     df.index.tolist(),
        "fold":      fold_assign,
        "y_true":    y,
        "y_pred":    oof,
    })
    oof_path = RESULTS_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[save]  OOF predictions → {oof_path.name}  ({len(oof_df)} rows)")

    # Final fit on all data
    final = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    fit_pred = final.predict(X)

    corr_oof = correlations(y, oof)
    corr_fit = correlations(y, fit_pred)
    print(f"[corr]  OOF   Pearson r={corr_oof['pearson']['r']:.4f}"
          f"  Spearman ρ={corr_oof['spearman']['rho']:.4f}")
    print(f"[corr]  Final Pearson r={corr_fit['pearson']['r']:.4f}"
          f"  Spearman ρ={corr_fit['spearman']['rho']:.4f}  (in-sample)")

    # Plots
    plot_cv(cv_results, alpha, RESULTS_DIR)
    plot_predictions(y, oof, RESULTS_DIR)
    plot_weights(final.coef_, RESULTS_DIR)

    ridge_coef = dict(zip(ATTRIBUTES, final.coef_.tolist()))

    results = {
        "timestamp":        datetime.now().isoformat(),
        "model_annotation": "Llama-3.1-70B vanilla",
        "n_samples":        len(df),
        "n_hate":           int(df["label"].sum()),
        "best_alpha":       alpha,
        "ridge_cv_per_alpha": {
            str(a): cv_results[a] for a in ALPHAS
        },
        "ridge_cv_summary": summary,
        "correlations": {"oof": corr_oof, "final": corr_fit},
        "weights": {
            "ridge": {
                "intercept":    round(float(final.intercept_), 8),
                "coefficients": {k: round(v, 8) for k, v in ridge_coef.items()},
            }
        },
    }

    weights_path = RESULTS_DIR / "weights.json"
    weights_path.write_text(json.dumps(results, indent=2))
    print(f"[save]  Weights → {weights_path.name}")

    # Report
    lines = [
        "=" * 64,
        "  RIDGE TRAINING — test_2000 (Llama-3.1-70B vanilla)",
        f"  Generated : {datetime.now().isoformat()}",
        "=" * 64,
        "",
        f"  Samples     : {len(df)}",
        f"  Hate (pos)  : {int(df['label'].sum())} ({100*df['label'].mean():.1f}%)",
        f"  Best alpha  : {alpha}",
        "",
        f"  CV R²   = {summary['r2']['mean']:.4f} ± {summary['r2']['std']:.4f}",
        f"  CV MAE  = {summary['mae']['mean']:.4f} ± {summary['mae']['std']:.4f}",
        f"  CV RMSE = {summary['rmse']['mean']:.4f} ± {summary['rmse']['std']:.4f}",
        "",
        f"  OOF  Pearson r = {corr_oof['pearson']['r']:.4f}  "
        f"Spearman ρ = {corr_oof['spearman']['rho']:.4f}",
        "",
        "  LEARNED RIDGE WEIGHTS",
        f"  {'Attribute':<20}  {'Coefficient':>12}",
        "  " + "-" * 34,
    ] + [
        f"  {attr:<20}  {final.coef_[i]:>12.6f}"
        for i, attr in enumerate(ATTRIBUTES)
    ] + [
        f"  {'intercept':<20}  {final.intercept_:>12.6f}",
        "=" * 64,
    ]
    report_path = RESULTS_DIR / "training_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"[save]  Report → {report_path.name}")

    print("\n" + "\n".join(lines))
    print("\nRun classification_metrics.py to get F1/AUROC.")


if __name__ == "__main__":
    main()
