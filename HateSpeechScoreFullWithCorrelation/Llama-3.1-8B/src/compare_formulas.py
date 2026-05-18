"""
Five-way formula comparison for Llama-3.3-70B:

  A) ĥₙ = Σ Sₙ,ᵢ · Cₙ,ᵢ · ρᵢ       no Ridge, with confidence, with Spearman
  B) ĥₙ = Σ Sₙ,ᵢ · Cₙ,ᵢ · wᵢ        Ridge, with confidence, no Spearman
  C) ĥₙ = Σ Sₙ,ᵢ · Cₙ,ᵢ · ρᵢ · wᵢ   Ridge, with confidence, with Spearman
  D) ĥₙ = Σ Sₙ,ᵢ · ρᵢ               no Ridge, no confidence, with Spearman
  E) ĥₙ = Σ Sₙ,ᵢ · wᵢ               Ridge, no confidence, no Spearman

For A/D : computed directly (no CV), evaluated on full dataset.
For B/C : OOF predictions loaded from saved results.
For E   : Ridge trained inline with 5-fold CV (no confidence features).

Saves: results/formula_comparison.txt, .json, .png
"""

import json
import glob
import os
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
# Paths
# ---------------------------------------------------------------------------

SRC_DIR    = Path(__file__).parent
BASE_DIR   = SRC_DIR.parent
THESIS_DIR = BASE_DIR.parent.parent

FULL_ANNOTATION_DIR = THESIS_DIR / "FullAnnotation_SmallLlama"
RESULTS_BASE        = FULL_ANNOTATION_DIR / "results"
GT_PATH             = THESIS_DIR / "Dataset" / "processed_dataset.csv"

RIDGE_ONLY_DIR = THESIS_DIR / "HateSpeechScoreFull" / "Llama-3.1-8B" / "results"
RIDGE_CORR_DIR = BASE_DIR / "results"
OUT_DIR        = BASE_DIR / "results"

MODEL_LABEL = "Llama-3.1-8B"

ATTRIBUTES = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
]

JOIN_KEY      = "comment_id"
ALPHAS        = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS      = 5
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_match(pattern: str) -> Path:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return Path(max(files, key=os.path.getmtime))


def load_ground_truth() -> pd.DataFrame:
    df = pd.read_csv(GT_PATH, low_memory=False)
    df = df[[JOIN_KEY, "hate_speech_score"]].dropna(subset=["hate_speech_score"])
    return (df.groupby(JOIN_KEY)["hate_speech_score"]
              .mean().reset_index()
              .rename(columns={"hate_speech_score": "target"}))


def load_spearman(mode: str) -> dict:
    pattern = str(RESULTS_BASE / f"evaluation_metrics_{mode}_*.json")
    path = latest_match(pattern)
    with open(path) as f:
        return json.load(f)["spearman_correlations"]


def load_predictions(mode: str) -> pd.DataFrame:
    if mode == "standard":
        pattern = str(RESULTS_BASE / "merged_standard_results_*.csv")
    else:
        pattern = str(RESULTS_BASE / "persona_results" / "merged_persona_results_*.csv")
    path = latest_match(pattern)
    df = pd.read_csv(path, low_memory=False)
    cols = [JOIN_KEY]
    for attr in ATTRIBUTES:
        for c in (attr, f"{attr}_confidence"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                cols.append(c)
    return df[cols].groupby(JOIN_KEY, as_index=False).mean()


def metrics_block(y_true, y_pred) -> dict:
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "r2":       float(r2_score(y_true, y_pred)),
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson":  float(pr),
        "spearman": float(sr),
    }


def load_oof(results_dir: Path, mode: str) -> tuple[np.ndarray, np.ndarray]:
    path = results_dir / mode / "test_predictions_oof.csv"
    df = pd.read_csv(path)
    return df["y_true"].values, df["y_pred_ridge"].values


# ---------------------------------------------------------------------------
# Formula A — Σ Sᵢ · Cᵢ · ρᵢ  (no Ridge, with confidence)
# ---------------------------------------------------------------------------

def run_A(merged: pd.DataFrame, rho: dict) -> dict:
    score = np.zeros(len(merged))
    for attr in ATTRIBUTES:
        conf_col = f"{attr}_confidence"
        if attr not in merged.columns or conf_col not in merged.columns:
            continue
        score += merged[attr].fillna(0).values * merged[conf_col].fillna(0).values * rho[attr]
    return metrics_block(merged["target"].values, score)


# ---------------------------------------------------------------------------
# Formula D — Σ Sᵢ · ρᵢ  (no Ridge, no confidence)
# ---------------------------------------------------------------------------

def run_D(merged: pd.DataFrame, rho: dict) -> dict:
    score = np.zeros(len(merged))
    for attr in ATTRIBUTES:
        if attr not in merged.columns:
            continue
        score += merged[attr].fillna(0).values * rho[attr]
    return metrics_block(merged["target"].values, score)


# ---------------------------------------------------------------------------
# Formula E — Ridge on raw Sᵢ only  (no confidence)
# ---------------------------------------------------------------------------

def run_E(merged: pd.DataFrame) -> dict:
    X = np.column_stack([
        merged[attr].fillna(0).values for attr in ATTRIBUTES
        if attr in merged.columns
    ])
    y = merged["target"].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # pick best alpha by CV R²
    best_alpha, best_r2 = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        fold_r2 = []
        for tr, te in kf.split(X):
            m = Ridge(alpha=alpha, fit_intercept=True).fit(X[tr], y[tr])
            fold_r2.append(r2_score(y[te], m.predict(X[te])))
        mean_r2 = np.mean(fold_r2)
        if mean_r2 > best_r2:
            best_r2, best_alpha = mean_r2, alpha

    print(f"    [E] best α={best_alpha}  CV R²={best_r2:.4f}")

    oof = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = Ridge(alpha=best_alpha, fit_intercept=True).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])

    return metrics_block(y, oof)


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def fmt(m):
    return (f"  R²={m['r2']:>8.4f}  MAE={m['mae']:>7.4f}  RMSE={m['rmse']:>7.4f}"
            f"  Pearson={m['pearson']:>7.4f}  Spearman={m['spearman']:>7.4f}")


def run_mode(mode: str, gt_df: pd.DataFrame) -> dict:
    print(f"\n{'='*65}\n  MODE: {mode.upper()}\n{'='*65}")

    rho     = load_spearman(mode)
    pred_df = load_predictions(mode)
    merged  = pred_df.merge(gt_df, on=JOIN_KEY, how="inner")

    m_a = run_A(merged, rho)
    print(f"[A] Σ Sᵢ·Cᵢ·ρᵢ       (no Ridge, conf, Spearman):{fmt(m_a)}")

    y_b, yp_b = load_oof(RIDGE_ONLY_DIR, mode)
    m_b = metrics_block(y_b, yp_b)
    print(f"[B] Σ Sᵢ·Cᵢ·wᵢ        (Ridge, conf, no Spearman):{fmt(m_b)}")

    y_c, yp_c = load_oof(RIDGE_CORR_DIR, mode)
    m_c = metrics_block(y_c, yp_c)
    print(f"[C] Σ Sᵢ·Cᵢ·ρᵢ·wᵢ     (Ridge, conf, Spearman)   :{fmt(m_c)}")

    m_d = run_D(merged, rho)
    print(f"[D] Σ Sᵢ·ρᵢ           (no Ridge, no conf, Spearman):{fmt(m_d)}")

    print(f"[E] Σ Sᵢ·wᵢ           (Ridge, no conf, no Spearman):")
    m_e = run_E(merged)
    print(f"    {fmt(m_e)}")

    return {"A": m_a, "B": m_b, "C": m_c, "D": m_d, "E": m_e}


# ---------------------------------------------------------------------------
# Report + plot
# ---------------------------------------------------------------------------

FORMULA_LABELS = {
    "A": "A: Σ Sᵢ·Cᵢ·ρᵢ       (no Ridge, conf+Spearman)",
    "B": "B: Σ Sᵢ·Cᵢ·wᵢ        (Ridge, conf only)",
    "C": "C: Σ Sᵢ·Cᵢ·ρᵢ·wᵢ     (Ridge, conf+Spearman)",
    "D": "D: Σ Sᵢ·ρᵢ            (no Ridge, Spearman only)",
    "E": "E: Σ Sᵢ·wᵢ             (Ridge, no conf no Spearman)",
}


def save_report(all_results: dict):
    lines = [
        "=" * 98,
        f"  FORMULA COMPARISON — {MODEL_LABEL}",
        "=" * 98,
        "",
    ]
    for k, v in FORMULA_LABELS.items():
        lines.append(f"  {v}")
    lines += [
        "",
        f"  {'Formula':<4}  {'Mode':<10}  {'R²':>8}  {'MAE':>8}  {'RMSE':>8}  {'Pearson':>9}  {'Spearman':>10}",
        "  " + "-" * 62,
    ]
    for mode, results in all_results.items():
        for key in ["A", "B", "C", "D", "E"]:
            m = results[key]
            lines.append(
                f"  {key:<4}  {mode:<10}  {m['r2']:>8.4f}  {m['mae']:>8.4f}"
                f"  {m['rmse']:>8.4f}  {m['pearson']:>9.4f}  {m['spearman']:>10.4f}"
            )
        lines.append("  " + "-" * 62)
    lines += ["", "=" * 98]

    txt_path  = OUT_DIR / "formula_comparison.txt"
    json_path = OUT_DIR / "formula_comparison.json"
    txt_path.write_text("\n".join(lines))
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")


def plot_comparison(all_results: dict):
    metrics      = ["r2", "pearson", "spearman", "mae"]
    metric_labels = ["R²", "Pearson r", "Spearman ρ", "MAE (lower=better)"]
    formula_keys  = ["A", "B", "C", "D", "E"]
    short_labels  = ["A: S·C·ρ\n(no Ridge)", "B: S·C·w\n(Ridge)", "C: S·C·ρ·w\n(Ridge+ρ)",
                     "D: S·ρ\n(no Ridge,\nno conf)", "E: S·w\n(Ridge,\nno conf)"]
    modes = list(all_results.keys())

    fig, axes = plt.subplots(len(modes), len(metrics), figsize=(5 * len(metrics), 4 * len(modes)))
    if len(modes) == 1:
        axes = [axes]

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    for row, mode in enumerate(modes):
        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row][col]
            vals = [all_results[mode][k][metric] for k in formula_keys]
            bars = ax.bar(short_labels, vals, color=colors, edgecolor="k", linewidth=0.5)
            ax.set_title(f"{mlabel} — {mode}", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(abs(v) for v in vals),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.tick_params(axis="x", labelsize=7)

    plt.suptitle(f"Formula Comparison — {MODEL_LABEL}", fontsize=13)
    plt.tight_layout()
    path = OUT_DIR / "formula_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    gt_df = load_ground_truth()
    all_results = {}
    for mode in ["standard", "persona"]:
        try:
            all_results[mode] = run_mode(mode, gt_df)
        except Exception as e:
            print(f"[error:{mode}] {e}")
            import traceback; traceback.print_exc()

    save_report(all_results)
    plot_comparison(all_results)


if __name__ == "__main__":
    main()
