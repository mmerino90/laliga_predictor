"""
Walk-forward evaluation.

For every test season S (2018-19 onwards) each model is trained only on seasons
before S and then predicts every match of S. Features are pre-match, so each
prediction uses only information available before kick-off. The model is frozen
at the start of S; the features (ELO, form, odds) still update match by match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import model as M

FIRST_TEST_SEASON = "2018-19"

VARIANTS = {
    "Base rates":               lambda tr: M.fit_base_rates(tr),
    "Bookmakers":               lambda tr: M.bookmakers,
    "ELO logistic":             lambda tr: M.fit_logistic(tr, ["diff_elo_pre"]),
    "Logistic (all features)":  lambda tr: M.fit_logistic(tr, M.FEATURES),
    "XGBoost (no odds)":        lambda tr: M.fit_xgb_result(tr, M.FORM_FEATURES),
    "XGBoost":                  lambda tr: M.fit_xgb_result(tr),
    "XGBoost + isotonic":       lambda tr: M.fit_xgb_result(tr, calibrate=True),
    "XGBoost tuned + isotonic": lambda tr: M.fit_xgb_result(tr, calibrate=True, tune=True),
}


def rps(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Ranked probability score per match (H < D < A are ordered). Lower is better."""
    cum_p = np.cumsum(p, axis=1)[:, :2]
    cum_y = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return ((cum_p - cum_y) ** 2).sum(axis=1) / 2


def season_folds(df: pd.DataFrame):
    """Yield (season, train, test): train = all played matches of earlier seasons."""
    played = df[df["is_future_match"] == 0]
    for season in sorted(played["season"].unique()):
        if season >= FIRST_TEST_SEASON:
            yield season, played[played["season"] < season], played[played["season"] == season]


def walk_forward(df: pd.DataFrame, variants: dict = VARIANTS) -> pd.DataFrame:
    """One row per test match: identifiers, result, and H/D/A probabilities per variant."""
    out = []
    for season, train, test in season_folds(df):
        print(f"  {season}: train {len(train)}  test {len(test)}")
        rows = test[["match_datetime", "season", "home_team", "away_team", "target_ftr",
                     "target_home_goals", "target_away_goals"]].copy()
        for name, fit in variants.items():
            p = fit(train)(test)
            for i, r in enumerate(M.RESULTS):
                rows[f"{name}|{r}"] = p[:, i].round(4)
        goals = M.fit_goals(train)(test)
        rows["xg_home"], rows["xg_away"] = goals[0].round(3), goals[1].round(3)
        rows["xg_home_baseline"] = train["target_home_goals"].mean()
        rows["xg_away_baseline"] = train["target_away_goals"].mean()
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def probs(preds: pd.DataFrame, name: str) -> np.ndarray:
    p = preds[[f"{name}|{r}" for r in M.RESULTS]].to_numpy()
    return p / p.sum(axis=1, keepdims=True)  # undo rounding drift


def variant_names(preds: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(c.split("|")[0] for c in preds.columns if "|" in c))


def summarise(preds: pd.DataFrame, reference: str = "Bookmakers",
              n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Accuracy, log-loss, RPS and draw share per variant, with a bootstrap 95% CI
    for the RPS difference against the reference (negative = better than reference)."""
    y = M.labels(preds)
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(y), (n_boot, len(y)))
    ref_rps = rps(y, probs(preds, reference))
    rows = []
    for name in variant_names(preds):
        p = probs(preds, name)
        r = rps(y, p)
        diff = (r - ref_rps)[boot].mean(axis=1)
        rows.append({
            "model": name,
            "accuracy": (p.argmax(1) == y).mean(),
            "log_loss": log_loss(y, p, labels=[0, 1, 2]),
            "rps": r.mean(),
            "rps_vs_bookmakers": r.mean() - ref_rps.mean(),
            "ci_low": np.percentile(diff, 2.5),
            "ci_high": np.percentile(diff, 97.5),
            "draw_picks": (p.argmax(1) == 1).mean(),
        })
    return pd.DataFrame(rows).set_index("model")


def per_season(preds: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    y = M.labels(preds)
    data = {n: (probs(preds, n).argmax(1) == y) for n in names}
    return pd.DataFrame(data).groupby(preds["season"].to_numpy()).mean()


def draw_rule(preds: pd.DataFrame, name: str, margins=np.arange(0, 0.21, 0.01)) -> dict:
    """
    Draw-aware decision rule, evaluated walk-forward: the margin used in season S is the
    one that maximised accuracy over all *earlier* test seasons (0 for the first one).
    """
    y, p = M.labels(preds), probs(preds, name)
    seasons = preds["season"].to_numpy()
    picks = np.empty_like(y)
    for s in np.unique(seasons):
        past, now = seasons < s, seasons == s
        best = 0.0
        if past.any():
            best = max(margins, key=lambda m: (M.draw_aware_picks(p[past], m) == y[past]).mean())
        picks[now] = M.draw_aware_picks(p[now], best)
    return {
        "accuracy_argmax": (p.argmax(1) == y).mean(),
        "accuracy_draw_rule": (picks == y).mean(),
        "draw_recall_argmax": (p.argmax(1)[y == 1] == 1).mean(),
        "draw_recall_draw_rule": (picks[y == 1] == 1).mean(),
        "draw_picks_draw_rule": (picks == 1).mean(),
    }


def forward_test(snapshot: pd.DataFrame, features: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """
    Score predictions that were committed to git *before* the matches were played
    against what actually happened, next to the bookmakers on the same matches.
    """
    # Match on (season, home, away): each pairing happens once per season, and some
    # kick-offs were rescheduled after the snapshot was taken.
    season = "2025-26"
    keys = ["home_team", "away_team"]
    played = features[(features["season"] == season) & (features["is_future_match"] == 0)]
    m = snapshot[keys + ["prob_home_win", "prob_draw", "prob_away_win"]].merge(played, on=keys).merge(
        preds.loc[preds["season"] == season, keys + [f"XGBoost|{r}" for r in M.RESULTS]], on=keys)
    y = M.labels(m)
    out = {"n": len(m), "of": len(snapshot), "first": str(m["match_datetime"].min().date()),
           "last": str(m["match_datetime"].max().date())}
    for name, p in [
        ("committed", m[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy()),
        ("bookmakers", M.bookmakers(m)),
        ("corrected_xgboost", probs(m, "XGBoost")),
    ]:
        p = p / p.sum(axis=1, keepdims=True)
        out[name] = {"accuracy": (p.argmax(1) == y).mean(), "rps": rps(y, p).mean(),
                     "log_loss": log_loss(y, p, labels=[0, 1, 2]), "home_picks": (p.argmax(1) == 0).mean()}
    return out


def goals_mae(preds: pd.DataFrame) -> dict:
    out = {}
    for side in ("home", "away"):
        actual = preds[f"target_{side}_goals"]
        out[f"{side}_mae_model"] = (preds[f"xg_{side}"] - actual).abs().mean()
        out[f"{side}_mae_baseline"] = (preds[f"xg_{side}_baseline"] - actual).abs().mean()
    return out
