"""README / site figures (PNG, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate as E, model as M

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#8f8e88"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def per_season_accuracy(preds: pd.DataFrame, out: Path) -> None:
    names = {"XGBoost": SERIES[0], "Bookmakers": SERIES[1], "ELO logistic": SERIES[2], "Base rates": GREY}
    acc = E.per_season(preds, list(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(acc))
    for name, color in names.items():
        label = "Bookmakers' odds (market forecast)" if name == "Bookmakers" else name
        ax.plot(x, acc[name] * 100, color=color, marker="o", markersize=5, label=label,
                linestyle="--" if name == "Base rates" else "-")
    ax.set_xticks(x, acc.index)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Walk-forward accuracy per season (each season predicted by a model trained on earlier seasons)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower right")
    ax.set_ylim(36, None)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def calibration(preds: pd.DataFrame, name: str, out: Path, bins: int = 8) -> None:
    y, p = M.labels(preds), E.probs(preds, name)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for c, (ax, label) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
        edges = np.quantile(p[:, c], np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, p[:, c], side="right") - 1, 0, bins - 1)
        pred = [p[idx == b, c].mean() for b in range(bins)]
        obs = [(y[idx == b] == c).mean() for b in range(bins)]
        ax.plot([0, 0.85], [0, 0.85], color=GREY, linestyle="--", linewidth=1)
        ax.plot(pred, obs, color=SERIES[0], marker="o", markersize=5)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle(f"Calibration of {name} on {len(y):,} walk-forward matches (dashed = perfect)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def elo_race(features: pd.DataFrame, season: str, out: Path, top: int = 6) -> None:
    s = features[(features["season"] == season) & (features["is_future_match"] == 0)]
    long = pd.concat([
        s[["match_datetime", "home_team", "home_elo_pre"]].set_axis(["date", "team", "elo"], axis=1),
        s[["match_datetime", "away_team", "away_elo_pre"]].set_axis(["date", "team", "elo"], axis=1),
    ]).sort_values("date")
    teams = long.groupby("team")["elo"].last().nlargest(top).index
    fig, ax = plt.subplots(figsize=(8, 4))
    for team, color in zip(teams, SERIES):
        t = long[long["team"] == team]
        ax.plot(t["date"], t["elo"], color=color, label=team)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_ylabel("ELO (pre-match)")
    ax.set_title(f"ELO ratings {season}: top {top} teams", loc="left", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
