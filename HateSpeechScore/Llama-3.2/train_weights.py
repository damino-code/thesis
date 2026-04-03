"""
Hate Speech Score Weight Training
==================================
Learn per-attribute correlation weights from Llama-3.2 model predictions.

Formula:
  score_adj_i  = Attribute_i × Confidence_i          (normalised attribute)
  HateScore    = Σ ( score_adj_i × Correlation_i )   (weighted sum)

Training:
  - 75 % of data   → 5-fold cross-validation to select the best model
  - 25 % of data   → held-out evaluation (never seen during training)

Ground truth: hate_speech_score (IRT-calibrated) from selected_comments.csv
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import nnls
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

LLAMA_CSV = BASE_DIR / "merged_standard_results_20260403_034909.csv"
GT_CSV    = BASE_DIR.parent / "selected_comments.csv"
RESULTS_DIR = BASE_DIR / "results"

ATTRIBUTES = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
]

ALPHAS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]   # Ridge regularisation grid
N_SPLITS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.25


# ---------------------------------------------------------------------------
# 1. Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Merge Llama-3.2 predictions with ground-truth hate_speech_score.
    Returns a DataFrame with score_adj features and target column.
    """
    llama_df = pd.read_csv(LLAMA_CSV)
    gt_df    = pd.read_csv(GT_CSV)

    # Aggregate ground truth: mean IRT score per comment_id
    # (handles multiple annotator rows if present)
    gt_agg = (
        gt_df.groupby("comment_id")["hate_speech_score"]
        .mean()
        .reset_index()
        .rename(columns={"hate_speech_score": "target"})
    )

    merged = llama_df.merge(gt_agg, on="comment_id", how="inner")

    print(f"[data]  Llama rows  : {len(llama_df)}")
    print(f"[data]  GT rows     : {len(gt_df)}  →  {len(gt_agg)} unique comments")
    print(f"[data]  Merged rows : {len(merged)}")

    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows – check that comment_ids overlap.")

    return merged


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    score_adj_i = Attribute_i × Confidence_i

    Missing attribute values are filled with 0 (no signal → no contribution).
    """
    features = {}
    for attr in ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        attr_vals = df[attr].fillna(0).values
        conf_vals = df[conf_col].fillna(0).values
        features[attr] = attr_vals * conf_vals

    return pd.DataFrame(features, index=df.index)


# ---------------------------------------------------------------------------
# 2. Cross-validated model comparison
# ---------------------------------------------------------------------------

def cv_ridge(X: np.ndarray, y: np.ndarray, alphas, kf: KFold):
    """
    5-fold CV for Ridge regression across a grid of regularisation strengths.
    Returns: best_alpha, per-fold R² scores for best alpha, all cv results.
    """
    cv_table = {}   # alpha → [fold R² scores]

    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            fold_scores.append(r2_score(y[val_idx], y_pred))
        cv_table[alpha] = fold_scores

    best_alpha = max(cv_table, key=lambda a: np.mean(cv_table[a]))
    return best_alpha, cv_table


def cv_nnls(X: np.ndarray, y: np.ndarray, kf: KFold):
    """
    5-fold CV for Non-Negative Least Squares (no intercept, weights ≥ 0).
    Returns per-fold R² scores.
    """
    fold_scores = []
    for train_idx, val_idx in kf.split(X):
        weights, _ = nnls(X[train_idx], y[train_idx])
        y_pred = X[val_idx] @ weights
        fold_scores.append(r2_score(y[val_idx], y_pred))
    return fold_scores


# ---------------------------------------------------------------------------
# 3. Final model training
# ---------------------------------------------------------------------------

def train_final_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return model


def train_final_nnls(X: np.ndarray, y: np.ndarray):
    weights, residual = nnls(X, y)
    return weights


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

def eval_metrics(y_true, y_pred, label="") -> dict:
    metrics = {
        "r2":   round(r2_score(y_true, y_pred), 6),
        "mae":  round(mean_absolute_error(y_true, y_pred), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
    }
    if label:
        print(f"\n[{label}]")
        print(f"  R²   = {metrics['r2']:.4f}   (1.0 = perfect)")
        print(f"  MAE  = {metrics['mae']:.4f}   (lower = better)")
        print(f"  RMSE = {metrics['rmse']:.4f}   (lower = better)")
    return metrics


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_cv_results(cv_table: dict, nnls_scores: list, best_alpha: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ridge CV R² across alphas
    alphas = list(cv_table.keys())
    means  = [np.mean(cv_table[a]) for a in alphas]
    stds   = [np.std(cv_table[a])  for a in alphas]
    axes[0].errorbar(
        np.log10(alphas), means, yerr=stds,
        marker="o", capsize=4, linewidth=2,
        label="Ridge CV R²"
    )
    axes[0].axvline(np.log10(best_alpha), color="red", linestyle="--",
                    label=f"best α={best_alpha}")
    axes[0].set_xlabel("log10(alpha)")
    axes[0].set_ylabel("CV R²")
    axes[0].set_title("Ridge Regularisation Search (5-fold CV)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # NNLS per-fold R²
    folds = range(1, len(nnls_scores) + 1)
    axes[1].bar(folds, nnls_scores, color="steelblue", alpha=0.7)
    axes[1].axhline(np.mean(nnls_scores), color="red", linestyle="--",
                    label=f"mean={np.mean(nnls_scores):.3f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("R²")
    axes[1].set_title("NNLS per-Fold R² (5-fold CV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "cv_results.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


def plot_predictions(y_true, y_pred_ridge, y_pred_nnls, split_label="Test"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_ridge, y_pred_nnls],
        [f"Ridge – {split_label}", f"NNLS – {split_label}"],
    ):
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
        lo = min(y_true.min(), y_pred.min()) - 0.1
        hi = max(y_true.max(), y_pred.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Ground Truth (IRT hate_speech_score)")
        ax.set_ylabel("Predicted HateScore")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "predictions_vs_actual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


def plot_weights(ridge_weights, nnls_weights, attribute_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_r = ["#d73027" if w >= 0 else "#4575b4" for w in ridge_weights]
    colors_n = ["#d73027" if w >= 0 else "#4575b4" for w in nnls_weights]

    for ax, weights, colors, title in zip(
        axes,
        [ridge_weights, nnls_weights],
        [colors_r, colors_n],
        ["Ridge Weights (Correlations)", "NNLS Weights (Correlations)"],
    ):
        bars = ax.barh(attribute_names, weights, color=colors, edgecolor="k",
                       linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Weight (Correlation)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        # Annotate values
        for bar, val in zip(bars, weights):
            ax.text(
                val + np.sign(val) * 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

    plt.tight_layout()
    path = RESULTS_DIR / "attribute_weights.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot]  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------

def save_report(results: dict):
    path = RESULTS_DIR / "evaluation_report.txt"
    lines = [
        "=" * 60,
        "  HATE SPEECH SCORE – WEIGHT TRAINING REPORT",
        f"  Generated: {results['timestamp']}",
        "=" * 60,
        "",
        "DATA SPLIT",
        f"  Total samples  : {results['n_total']}",
        f"  Train (75%)    : {results['n_train']}",
        f"  Test  (25%)    : {results['n_test']}",
        "",
        "CROSS-VALIDATION (5-fold on training set)",
        f"  Ridge best α   : {results['best_alpha']}",
        f"  Ridge CV R²    : {np.mean(results['ridge_cv'][str(results['best_alpha'])]):.4f}"
        f" ± {np.std(results['ridge_cv'][str(results['best_alpha'])]):.4f}",
        f"  NNLS  CV R²    : {np.mean(results['nnls_cv']):.4f}"
        f" ± {np.std(results['nnls_cv']):.4f}",
        "",
        "TEST SET EVALUATION",
        "  Ridge:",
        f"    R²   = {results['test_metrics']['ridge']['r2']:.4f}",
        f"    MAE  = {results['test_metrics']['ridge']['mae']:.4f}",
        f"    RMSE = {results['test_metrics']['ridge']['rmse']:.4f}",
        "  NNLS:",
        f"    R²   = {results['test_metrics']['nnls']['r2']:.4f}",
        f"    MAE  = {results['test_metrics']['nnls']['mae']:.4f}",
        f"    RMSE = {results['test_metrics']['nnls']['rmse']:.4f}",
        "",
        "LEARNED WEIGHTS (Correlations)",
        f"  {'Attribute':<20}  {'Ridge':>10}  {'NNLS':>10}",
        "  " + "-" * 44,
    ]
    for attr in ATTRIBUTES:
        rw = results["weights"]["ridge"]["coefficients"][attr]
        nw = results["weights"]["nnls"][attr]
        lines.append(f"  {attr:<20}  {rw:>10.6f}  {nw:>10.6f}")

    if "ridge" in results["weights"]:
        lines += [
            "",
            f"  Ridge intercept : {results['weights']['ridge']['intercept']:.6f}",
        ]

    lines += [
        "",
        "HOW TO INTERPRET RESULTS",
        "  R² = 1.0  → model perfectly explains variance in hate speech scores",
        "  R² = 0.0  → model no better than predicting the mean",
        "  R² < 0    → model worse than predicting the mean (bad fit)",
        "",
        "  Good weights show POSITIVE values for hate-related attributes",
        "  (insult, dehumanize, violence, genocide, hatespeech)",
        "  and NEGATIVE values for protective attributes (respect, sentiment).",
        "",
        "  Convergence between Ridge and NNLS weights = more stable estimates.",
        "  Low CV std-dev relative to mean R² = stable across folds (good).",
        "=" * 60,
    ]
    path.write_text("\n".join(lines))
    print(f"[report] Saved: {path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load & prepare -----------------------------------------------
    df = load_data()
    X  = compute_features(df)
    y  = df["target"]

    print(f"\n[target] hate_speech_score range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"[target] mean: {y.mean():.3f}  std: {y.std():.3f}")

    # ---- Train / Test split -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[split]  Train: {len(X_train)}  Test: {len(X_test)}")

    X_tr = X_train.values
    y_tr = y_train.values
    X_te = X_test.values
    y_te = y_test.values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Cross-validation --------------------------------------------
    print(f"\n[cv]  Running {N_SPLITS}-fold CV …")

    best_alpha, ridge_cv_table = cv_ridge(X_tr, y_tr, ALPHAS, kf)
    nnls_cv_scores = cv_nnls(X_tr, y_tr, kf)

    print(f"[cv]  Ridge best α  = {best_alpha}")
    print(f"[cv]  Ridge mean R² = {np.mean(ridge_cv_table[best_alpha]):.4f}"
          f" ± {np.std(ridge_cv_table[best_alpha]):.4f}")
    print(f"[cv]  NNLS  mean R² = {np.mean(nnls_cv_scores):.4f}"
          f" ± {np.std(nnls_cv_scores):.4f}")

    # ---- Final training ----------------------------------------------
    ridge_model   = train_final_ridge(X_tr, y_tr, best_alpha)
    nnls_weights  = train_final_nnls(X_tr, y_tr)

    # ---- Test evaluation --------------------------------------------
    y_pred_ridge = ridge_model.predict(X_te)
    y_pred_nnls  = X_te @ nnls_weights

    ridge_metrics = eval_metrics(y_te, y_pred_ridge, label="Ridge  – TEST SET")
    nnls_metrics  = eval_metrics(y_te, y_pred_nnls,  label="NNLS   – TEST SET")

    # ---- Plots -------------------------------------------------------
    plot_cv_results(ridge_cv_table, nnls_cv_scores, best_alpha)
    plot_predictions(y_te, y_pred_ridge, y_pred_nnls)
    plot_weights(ridge_model.coef_, nnls_weights, ATTRIBUTES)

    # ---- Save weights -----------------------------------------------
    ridge_coef_dict = dict(zip(ATTRIBUTES, ridge_model.coef_.tolist()))
    nnls_coef_dict  = dict(zip(ATTRIBUTES, nnls_weights.tolist()))

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "best_alpha": best_alpha,
        "ridge_cv": {str(a): v for a, v in ridge_cv_table.items()},
        "nnls_cv":  nnls_cv_scores,
        "weights": {
            "ridge": {
                "intercept":    round(float(ridge_model.intercept_), 8),
                "coefficients": {k: round(v, 8) for k, v in ridge_coef_dict.items()},
            },
            "nnls": {k: round(v, 8) for k, v in nnls_coef_dict.items()},
        },
        "test_metrics": {
            "ridge": ridge_metrics,
            "nnls":  nnls_metrics,
        },
    }

    weights_path = RESULTS_DIR / "weights.json"
    weights_path.write_text(json.dumps(results, indent=2))
    print(f"\n[save]  Weights → {weights_path}")

    save_report(results)

    # ---- Summary print -----------------------------------------------
    print("\n" + "=" * 55)
    print("  LEARNED WEIGHTS (Correlations)")
    print(f"  {'Attribute':<20}  {'Ridge':>10}  {'NNLS':>10}")
    print("  " + "-" * 44)
    for attr, rw, nw in zip(ATTRIBUTES, ridge_model.coef_, nnls_weights):
        print(f"  {attr:<20}  {rw:>10.6f}  {nw:>10.6f}")
    print(f"  {'intercept':<20}  {ridge_model.intercept_:>10.6f}  {'(none)':>10}")
    print("=" * 55)

    print("\nDone. Check results/ for weights.json, plots, and evaluation_report.txt")


if __name__ == "__main__":
    main()
