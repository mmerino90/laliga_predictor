"""
La Liga Predictor — Visualisation Suite
========================================
Generates diagnostic plots into plots/ and saves them as PNG files.

Run:
    python visualise.py

Requires the pipeline to have been run first:
    python data_engineering/build_model_dataset.py
    python prediction/train_predict.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
PALETTE   = {"H": "#1a73e8", "D": "#fbbc04", "A": "#ea4335"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 130


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — re-run a lightweight version of the pipeline for chart data
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline_data():
    """Return (played_df, train, test, clf, features, medians)."""
    import sys
    sys.path.insert(0, str(ROOT / "prediction"))
    from train_predict import (
        FEATURES, XGB_PARAMS, GOALS_PARAMS,
        add_season_progress, engineer_features,
        recency_weights, fit_imputer, apply_imputer,
        train_result_model, RESULT_TO_INT,
    )

    df = pd.read_csv(DATA_DIR / "model_ready_matches_25_26.csv", encoding="utf-8-sig")
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], format="mixed")
    df = df.sort_values("match_datetime").reset_index(drop=True)
    df = add_season_progress(df)
    df = engineer_features(df)

    played  = df[df["is_future_match"] == 0].copy().reset_index(drop=True)
    current = played[played["is_current_season"] == 1].copy().reset_index(drop=True)
    hist    = played[played["is_current_season"] == 0].copy().reset_index(drop=True)

    cur_split = int(len(current) * 0.83)
    train = pd.concat([hist, current.iloc[:cur_split]], ignore_index=True).sort_values("match_datetime").reset_index(drop=True)
    test  = current.iloc[cur_split:].copy().reset_index(drop=True)

    medians = fit_imputer(train[FEATURES])
    X_train = apply_imputer(train[FEATURES], medians)
    X_test  = apply_imputer(test[FEATURES],  medians)
    y_train = train["target_ftr"].map(RESULT_TO_INT).values
    y_test  = test["target_ftr"].map(RESULT_TO_INT).values
    w_train = recency_weights(len(train))

    clf = train_result_model(X_train, y_train, X_test, y_test, w_train)
    prob_test = clf.predict_proba(X_test)

    return played, train, test, clf, FEATURES, medians, X_test, y_test, prob_test


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Feature importance (horizontal bar, top 20)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(clf, features: list[str]) -> None:
    fi = pd.Series(clf.feature_importances_, index=features).sort_values()
    fi_top = fi.tail(20)

    # Colour by feature group
    def group_colour(name: str) -> str:
        if "prob" in name:         return "#1a73e8"
        if "elo" in name:          return "#e37400"
        if "form" in name:         return "#34a853"
        if "venue" in name:        return "#a142f4"
        if "h2h" in name:          return "#ea4335"
        if "rest" in name:         return "#80868b"
        return "#5f6368"

    colours = [group_colour(n) for n in fi_top.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(fi_top.index, fi_top.values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature importance (gain)", labelpad=10)
    ax.set_title("Top 20 Features — Result Model", fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#1a73e8", label="Bookmaker odds"),
        Patch(color="#e37400", label="ELO"),
        Patch(color="#34a853", label="Rolling form"),
        Patch(color="#a142f4", label="Venue-specific form"),
        Patch(color="#ea4335", label="Head-to-head"),
        Patch(color="#80868b", label="Rest days"),
        Patch(color="#5f6368", label="Table stats"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = PLOTS_DIR / "feature_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Calibration curves (H / D / A)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(y_test: np.ndarray, prob_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    labels = ["Home win (H)", "Draw (D)", "Away win (A)"]
    colours = ["#1a73e8", "#fbbc04", "#ea4335"]

    for i, (label, colour, ax) in enumerate(zip(labels, colours, axes)):
        y_bin = (y_test == i).astype(int)
        prob  = prob_test[:, i]
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, prob, n_bins=8, strategy="quantile")
        except ValueError:
            continue
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color=colour, lw=2, label="Model")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Predicted probability")
        if i == 0:
            ax.set_ylabel("Actual frequency")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Probability Calibration — Are Predicted Probs Reliable?", fontweight="bold", y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "calibration_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Confusion matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test: np.ndarray, prob_test: np.ndarray) -> None:
    pred = prob_test.argmax(axis=1)
    labels = ["H", "D", "A"]
    cm = confusion_matrix(y_test, pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Row %"},
        ax=ax,
    )
    # Annotate raw counts in smaller text
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.72, f"n={cm[i,j]}", ha="center", va="center",
                    fontsize=8, color="grey")

    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Actual", labelpad=10)
    ax.set_title("Confusion Matrix — Current-Season Holdout\n(row % — how often each actual result was predicted)", fontweight="bold")
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — ELO trajectory for top 6 teams (current season)
# ─────────────────────────────────────────────────────────────────────────────

def plot_elo_trajectory(played: pd.DataFrame) -> None:
    current = played[played["is_current_season"] == 1].copy()

    # Identify top 6 teams by final ELO
    latest_elo = (
        current.groupby("home_team")["home_elo_pre"].last()
        .combine_first(current.groupby("away_team")["away_elo_pre"].last())
    )
    top6 = latest_elo.nlargest(6).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 5))

    for team in top6:
        home_rows = current[current["home_team"] == team][["match_datetime", "home_elo_pre"]].rename(columns={"home_elo_pre": "elo"})
        away_rows = current[current["away_team"] == team][["match_datetime", "away_elo_pre"]].rename(columns={"away_elo_pre": "elo"})
        trajectory = pd.concat([home_rows, away_rows]).sort_values("match_datetime")
        ax.plot(trajectory["match_datetime"], trajectory["elo"], marker="o", markersize=3, lw=1.8, label=team)

    ax.set_title("ELO Rating Trajectory — Top 6 Teams (2025/26)", fontweight="bold", pad=12)
    ax.set_ylabel("ELO Rating")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.axhline(1500, color="grey", linestyle="--", alpha=0.4, label="League average")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = PLOTS_DIR / "elo_trajectory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Score probability heatmap (one fixture example)
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_heatmap(lam_h: float, lam_a: float,
                       home: str, away: str) -> None:
    import math

    def pmf(k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(k)

    n = 6
    grid = np.array([[pmf(h, lam_h) * pmf(a, lam_a) for a in range(n)] for h in range(n)])
    grid /= grid.sum()  # normalise

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        grid * 100, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=range(n), yticklabels=range(n),
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Probability (%)"},
        ax=ax,
    )
    ax.set_xlabel(f"Away goals  ({away})", labelpad=10)
    ax.set_ylabel(f"Home goals  ({home})", labelpad=10)
    ax.set_title(f"Score Probability Matrix\n{home}  vs  {away}\n"
                 f"(xG  {lam_h:.2f} – {lam_a:.2f})",
                 fontweight="bold")
    # shade diagonal (draws) differently
    for i in range(min(n, n)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#1a73e8", lw=2))

    fig.tight_layout()
    out = PLOTS_DIR / "score_heatmap_example.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading pipeline data ...")
    played, train, test, clf, features, medians, X_test, y_test, prob_test = _load_pipeline_data()

    print("\nGenerating plots ...")
    plot_feature_importance(clf, features)
    plot_calibration(y_test, prob_test)
    plot_confusion_matrix(y_test, prob_test)
    plot_elo_trajectory(played)

    # Score heatmap — use first future fixture
    preds = pd.read_csv(DATA_DIR / "predictions_future_matches.csv", encoding="utf-8-sig")
    row = preds.iloc[0]
    plot_score_heatmap(row["xg_home"], row["xg_away"], row["home_team"], row["away_team"])

    print(f"\nAll plots saved to  {PLOTS_DIR}/")
