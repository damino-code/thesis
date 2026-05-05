"""
Hate Speech Score Weight Training — Full Annotation (Qwen2.5-72B)
==================================================================
Learns per-attribute correlation weights from the full-annotation Qwen
predictions. Runs once for STANDARD (vanilla) merged predictions and
once for PERSONA (feature-based) merged predictions.

Formula (same as the small-model version in HateSpeechScore/):
  score_adj_i = Attribute_i × Confidence_i
  HateScore   = Σ ( score_adj_i × Correlation_i )

Aggregation:
  Both predictions and ground truth are reduced to one row per comment_id
  by averaging across annotators (matches the small-model script).

Training:
  Pure 5-fold cross-validation — each fold fits on 4/5 of the unique
  comments and tests on the remaining 1/5. Final weights are then re-fit
  on all unique comments.

Outputs go to results/standard/ and results/persona/.
"""

import json
import os
import glob
import sys
from pathlib import Path
from datetime import datetime

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
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
THESIS_DIR = BASE_DIR.parent.parent
FULL_ANNOTATION_DIR = THESIS_DIR / "FullAnnotation_Qwen"
RESULTS_BASE = FULL_ANNOTATION_DIR / "results"
GT_PATH = THESIS_DIR / "Dataset" / "processed_dataset.csv"

MODEL_LABEL = "Qwen2.5-72B"

ATTRIBUTES = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
]

ALPHAS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS = 5
RANDOM_STATE = 42

JOIN_KEY = "comment_id"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def latest_match(pattern: str) -> Path:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return Path(max(files, key=os.path.getmtime))


def load_predictions(mode: str) -> tuple[pd.DataFrame, Path]:
    if mode == "standard":
        pattern = str(RESULTS_BASE / "merged_standard_results_*.csv")
    elif mode == "persona":
        pattern = str(RESULTS_BASE / "persona_results" / "merged_persona_results_*.csv")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    path = latest_match(pattern)
    print(f"[data:{mode}]  Predictions file: {path.name}")
    return pd.read_csv(path, low_memory=False), path


def aggregate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per comment_id: mean of every numeric attribute / confidence column."""
    cols = [JOIN_KEY]
    for attr in ATTRIBUTES:
        for c in (attr, f"{attr}_confidence"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                cols.append(c)
    return df[cols].groupby(JOIN_KEY, as_index=False).mean()


def load_ground_truth() -> pd.DataFrame:
    """One row per comment_id with the IRT score averaged across annotators."""
    print(f"[data]  Ground truth: {GT_PATH}")
    df = pd.read_csv(GT_PATH, low_memory=False)
    df = df[[JOIN_KEY, "hate_speech_score"]].dropna(subset=["hate_speech_score"])
    agg = (
        df.groupby(JOIN_KEY)["hate_speech_score"]
        .mean()
        .reset_index()
        .rename(columns={"hate_speech_score": "target"})
    )
    return agg


def merge_with_gt(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    if JOIN_KEY not in pred_df.columns:
        raise ValueError(f"'{JOIN_KEY}' missing from predictions — re-run merge_results.py")
    merged = pred_df.merge(gt_df, on=JOIN_KEY, how="inner")
    print(f"[data]  Unique-comment predictions: {len(pred_df)}  "
          f"GT: {len(gt_df)}  Merged: {len(merged)}")
    return merged


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """score_adj_i = Attribute_i × Confidence_i"""
    features = {}
    for attr in ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        if attr not in df.columns or conf_col not in df.columns:
            print(f"[warn]  Skipping {attr}: missing column(s)")
            features[attr] = np.zeros(len(df))
            continue
        attr_vals = df[attr].fillna(0).values
        conf_vals = df[conf_col].fillna(0).values
        features[attr] = attr_vals * conf_vals
    return pd.DataFrame(features, index=df.index)


# ---------------------------------------------------------------------------
# Cross-validation (pure 5-fold, Ridge only)
# ---------------------------------------------------------------------------

def fold_metrics(y_true, y_pred) -> dict:
    return {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cv_ridge(X, y, alphas, kf):
    """For each α: per-fold metrics. Best α = highest mean R²."""
    cv_table = {}
    for alpha in alphas:
        per_fold = []
        for tr_idx, te_idx in kf.split(X):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X[tr_idx], y[tr_idx])
            per_fold.append(fold_metrics(y[te_idx], model.predict(X[te_idx])))
        cv_table[alpha] = per_fold
    best_alpha = max(cv_table, key=lambda a: np.mean([f["r2"] for f in cv_table[a]]))
    return best_alpha, cv_table


def summarise(per_fold: list) -> dict:
    keys = ["r2", "mae", "rmse"]
    return {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "std":  float(np.std([f[k] for f in per_fold])),
            "folds": [float(f[k]) for f in per_fold],
        }
        for k in keys
    }


def correlations(y_true, y_pred) -> dict:
    """Pearson + Spearman of predicted HateScore vs human hate_speech_score."""
    pr, pr_p = pearsonr(y_true, y_pred)
    sr, sr_p = spearmanr(y_true, y_pred)
    return {
        "pearson":  {"r":   float(pr), "p_value": float(pr_p)},
        "spearman": {"rho": float(sr), "p_value": float(sr_p)},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cv_results(ridge_cv, best_alpha, out_dir: Path, mode: str):
    alphas = list(ridge_cv.keys())
    means = [np.mean([f["r2"] for f in ridge_cv[a]]) for a in alphas]
    stds  = [np.std([f["r2"] for f in ridge_cv[a]])  for a in alphas]

    plt.figure(figsize=(8, 5))
    plt.errorbar(np.log10(alphas), means, yerr=stds,
                 marker="o", capsize=4, linewidth=2, label="Ridge CV R²")
    plt.axvline(np.log10(best_alpha), color="red", linestyle="--",
                label=f"best α={best_alpha}")
    plt.xlabel("log10(alpha)")
    plt.ylabel("CV R²")
    plt.title(f"Ridge α search (5-fold CV) — {mode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = out_dir / "cv_results.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


def plot_predictions(y_true, y_pred, out_dir: Path, mode: str):
    plt.figure(figsize=(7, 6))
    sample = np.random.RandomState(0).choice(
        len(y_true), size=min(5000, len(y_true)), replace=False,
    )
    plt.scatter(y_true[sample], y_pred[sample], alpha=0.4, s=10, edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) - 0.1
    hi = max(y_true.max(), y_pred.max()) + 0.1
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Ground Truth (mean IRT hate_speech_score)")
    plt.ylabel("Predicted HateScore (Ridge OOF)")
    plt.title(f"Ridge OOF predictions — {mode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = out_dir / "predictions_vs_actual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


def plot_weights(ridge_w, names, out_dir: Path, mode: str):
    plt.figure(figsize=(8, 5))
    colors = ["#d73027" if w >= 0 else "#4575b4" for w in ridge_w]
    bars = plt.barh(names, ridge_w, color=colors, edgecolor="k", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Weight (Correlation)")
    plt.title(f"Ridge weights — {mode}")
    plt.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, ridge_w):
        plt.text(val + np.sign(val) * 0.002,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()

    path = out_dir / "attribute_weights.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_report(results: dict, out_dir: Path):
    rs = results["ridge_cv_summary"]
    co = results["correlations"]["oof"]
    cf = results["correlations"]["final"]
    lines = [
        "=" * 64,
        f"  HATE SPEECH SCORE — FULL ANNOTATION ({results['model']})",
        f"  Mode      : {results['mode'].upper()}",
        f"  Generated : {results['timestamp']}",
        "=" * 64,
        "",
        "DATA",
        f"  Predictions file : {results['pred_file']}",
        f"  Unique comments  : {results['n_total']}",
        f"  CV scheme        : {N_SPLITS}-fold (4 folds fit / 1 fold test)",
        "",
        f"CROSS-VALIDATION (mean ± std across {N_SPLITS} folds)",
        f"  Ridge best α     : {results['best_alpha']}",
        f"  Ridge R²         : {rs['r2']['mean']:.4f} ± {rs['r2']['std']:.4f}",
        f"  Ridge MAE        : {rs['mae']['mean']:.4f} ± {rs['mae']['std']:.4f}",
        f"  Ridge RMSE       : {rs['rmse']['mean']:.4f} ± {rs['rmse']['std']:.4f}",
        "",
        "CORRELATION WITH HUMAN HATE SPEECH SCORE  (predicted vs hate_speech_score)",
        f"  OOF   Pearson r  : {co['pearson']['r']:.4f}   (p={co['pearson']['p_value']:.2e})",
        f"  OOF   Spearman ρ : {co['spearman']['rho']:.4f}   (p={co['spearman']['p_value']:.2e})",
        f"  Final Pearson r  : {cf['pearson']['r']:.4f}   (p={cf['pearson']['p_value']:.2e})   [in-sample]",
        f"  Final Spearman ρ : {cf['spearman']['rho']:.4f}   (p={cf['spearman']['p_value']:.2e})   [in-sample]",
        "",
        "FINAL WEIGHTS (re-fit on all unique comments)",
        f"  {'Attribute':<20}  {'Ridge':>10}",
        "  " + "-" * 32,
    ]
    for attr in ATTRIBUTES:
        rw = results["weights"]["ridge"]["coefficients"][attr]
        lines.append(f"  {attr:<20}  {rw:>10.6f}")
    lines += [
        "",
        f"  Ridge intercept : {results['weights']['ridge']['intercept']:.6f}",
        "",
        "INTERPRETATION",
        "  R² = 1.0  → predictions perfectly match hate_speech_score",
        "  R² = 0.0  → no better than predicting the mean",
        "  Positive weights expected on hate-related attributes",
        "  (insult, dehumanize, violence, genocide, hatespeech) and",
        "  negative weights on protective attributes (respect, sentiment).",
        "=" * 64,
    ]
    path = out_dir / "evaluation_report.txt"
    path.write_text("\n".join(lines))
    print(f"[report] Saved: {path}")


# ---------------------------------------------------------------------------
# Run a single mode
# ---------------------------------------------------------------------------

def run_mode(mode: str):
    print("\n" + "#" * 70)
    print(f"#  {MODEL_LABEL} — {mode.upper()}")
    print("#" * 70)

    out_dir = BASE_DIR / "results" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df_raw, pred_path = load_predictions(mode)
    pred_df = aggregate_predictions(pred_df_raw)
    print(f"[data:{mode}]  Aggregated to {len(pred_df)} unique comments "
          f"(from {len(pred_df_raw)} rows).")

    gt_df = load_ground_truth()
    merged = merge_with_gt(pred_df, gt_df)
    if len(merged) == 0:
        print(f"[skip:{mode}]  No overlapping comments — skipping.")
        return

    X = compute_features(merged).values
    y = merged["target"].astype(float).values

    print(f"[target] range [{y.min():.3f}, {y.max():.3f}]"
          f"  mean={y.mean():.3f}  std={y.std():.3f}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[cv]  Running {N_SPLITS}-fold CV …")
    best_alpha, ridge_cv = cv_ridge(X, y, ALPHAS, kf)
    ridge_summary = summarise(ridge_cv[best_alpha])

    print(f"[cv]  Ridge best α = {best_alpha}")
    print(f"[cv]  Ridge R²  = {ridge_summary['r2']['mean']:.4f}"
          f" ± {ridge_summary['r2']['std']:.4f}")
    print(f"[cv]  Ridge MAE = {ridge_summary['mae']['mean']:.4f}"
          f" ± {ridge_summary['mae']['std']:.4f}")
    print(f"[cv]  Ridge RMSE= {ridge_summary['rmse']['mean']:.4f}"
          f" ± {ridge_summary['rmse']['std']:.4f}")

    # OOF predictions — every comment is in the test fold exactly once.
    oof_ridge = np.zeros_like(y)
    fold_assign = np.full(len(y), -1, dtype=int)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        rm = Ridge(alpha=best_alpha, fit_intercept=True)
        rm.fit(X[tr_idx], y[tr_idx])
        oof_ridge[te_idx] = rm.predict(X[te_idx])
        fold_assign[te_idx] = fold_idx

    oof_df = pd.DataFrame({
        "row_index":    np.arange(len(merged)),
        "comment_id":   merged[JOIN_KEY].values,
        "fold":         fold_assign,
        "y_true":       y,
        "y_pred_ridge": oof_ridge,
    })
    oof_path = out_dir / "test_predictions_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[save]  OOF test predictions → {oof_path}  ({len(oof_df)} unique comments)")

    # Final fit on all unique comments
    final_ridge = Ridge(alpha=best_alpha, fit_intercept=True).fit(X, y)
    fit_ridge = final_ridge.predict(X)

    # Correlations between predicted HateScore and human hate_speech_score
    corr_oof = correlations(y, oof_ridge)
    corr_fit = correlations(y, fit_ridge)
    print(f"[corr]  OOF   Pearson r = {corr_oof['pearson']['r']:.4f}"
          f"  Spearman ρ = {corr_oof['spearman']['rho']:.4f}")
    print(f"[corr]  Final Pearson r = {corr_fit['pearson']['r']:.4f}"
          f"  Spearman ρ = {corr_fit['spearman']['rho']:.4f}  (in-sample)")

    plot_cv_results(ridge_cv, best_alpha, out_dir, mode)
    plot_predictions(y, oof_ridge, out_dir, mode)
    plot_weights(final_ridge.coef_, ATTRIBUTES, out_dir, mode)

    ridge_coef = dict(zip(ATTRIBUTES, final_ridge.coef_.tolist()))

    results = {
        "timestamp":  datetime.now().isoformat(),
        "model":      MODEL_LABEL,
        "mode":       mode,
        "pred_file":  pred_path.name,
        "n_total":    len(merged),
        "best_alpha": best_alpha,
        "ridge_cv_per_alpha": {
            str(a): [f for f in v] for a, v in ridge_cv.items()
        },
        "ridge_cv_summary": ridge_summary,
        "correlations": {
            "oof":   corr_oof,
            "final": corr_fit,
        },
        "weights": {
            "ridge": {
                "intercept":    round(float(final_ridge.intercept_), 8),
                "coefficients": {k: round(v, 8) for k, v in ridge_coef.items()},
            },
        },
    }

    weights_path = out_dir / "weights.json"
    weights_path.write_text(json.dumps(results, indent=2))
    print(f"[save]  Weights → {weights_path}")
    save_report(results, out_dir)

    print("\n" + "=" * 45)
    print(f"  LEARNED RIDGE WEIGHTS — {mode.upper()}")
    print(f"  {'Attribute':<20}  {'Ridge':>10}")
    print("  " + "-" * 32)
    for attr, rw in zip(ATTRIBUTES, final_ridge.coef_):
        print(f"  {attr:<20}  {rw:>10.6f}")
    print(f"  {'intercept':<20}  {final_ridge.intercept_:>10.6f}")
    print("=" * 45)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["standard", "persona"]
    for mode in modes:
        try:
            run_mode(mode)
        except FileNotFoundError as e:
            print(f"[skip:{mode}]  {e}")
        except Exception as e:
            print(f"[error:{mode}]  {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
