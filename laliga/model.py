"""Models: XGBoost result classifier (+ isotonic calibration), Poisson goals models, baselines."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = ["H", "D", "A"]
ODDS_FEATURES = ["consensus_home_prob", "consensus_draw_prob", "consensus_away_prob"]
FORM_FEATURES = [
    # table strength
    "home_matches_played_pre", "away_matches_played_pre",
    "home_ppm_pre", "away_ppm_pre", "diff_ppm_pre",
    "home_gdpm_pre", "away_gdpm_pre", "diff_gdpm_pre",
    "home_home_ppm_pre", "away_away_ppm_pre",
    # ELO: only the gap matters
    "diff_elo_pre",
    # rolling form (3 / 5 / 10 matches)
    "home_form3_pts_pre", "away_form3_pts_pre", "diff_form3_pts",
    "home_form5_pts_pre", "away_form5_pts_pre", "diff_form5_pts",
    "home_form10_pts_pre", "away_form10_pts_pre", "diff_form10_pts",
    "home_form5_gf_pre", "away_form5_gf_pre", "diff_form5_gf",
    "home_form5_ga_pre", "away_form5_ga_pre", "diff_form5_ga",
    # shots on target
    "home_form5_sot_for_pre", "away_form5_sot_for_pre", "diff_sot_pre",
    "home_form5_sot_ag_pre", "away_form5_sot_ag_pre", "diff_sot_ag",
    # venue-specific form
    "home_venue_pts5_pre", "away_venue_pts5_pre", "diff_venue_pts5",
    "home_venue_gf5_pre", "away_venue_gf5_pre", "diff_venue_gf5",
    "home_venue_ga5_pre", "away_venue_ga5_pre", "diff_venue_ga5",
    "home_venue_sot5_pre", "away_venue_sot5_pre",
    "home_venue_sotag5_pre", "away_venue_sotag5_pre",
    # rest, head-to-head, season progress
    "home_rest_days_pre", "away_rest_days_pre", "diff_rest_days_pre",
    "h2h_pts_pre", "h2h_gf_pre", "h2h_ga_pre",
    "season_progress",
]
FEATURES = FORM_FEATURES + ODDS_FEATURES

BASE_PARAMS = dict(
    n_estimators=800, max_depth=4, learning_rate=0.025, subsample=0.75,
    colsample_bytree=0.70, min_child_weight=10, reg_alpha=0.8, reg_lambda=3.0,
    random_state=42, n_jobs=4, verbosity=0,
)
TUNING_GRID = dict(max_depth=[2, 3, 4], min_child_weight=[10, 40])
SEASON = 380  # matches per La Liga season

Predictor = Callable[[pd.DataFrame], np.ndarray]


def labels(df: pd.DataFrame) -> np.ndarray:
    return df["target_ftr"].map({r: i for i, r in enumerate(RESULTS)}).to_numpy()


def recency_weights(n: int, half_life: int = SEASON) -> np.ndarray:
    """Exponential decay: a match one season old counts half as much."""
    return 0.5 ** ((n - 1 - np.arange(n)) / half_life)


def _fit_xgb(train: pd.DataFrame, features: list[str], y: np.ndarray,
             objective: str, **params) -> xgb.XGBModel:
    """Early-stop on the last 10% of train (by time), then refit on all of it."""
    cls = xgb.XGBClassifier if objective == "multi:softprob" else xgb.XGBRegressor
    p = {**BASE_PARAMS, **params, "objective": objective}
    X, w, cut = train[features], recency_weights(len(train)), int(len(train) * 0.9)
    probe = cls(**p, early_stopping_rounds=50)
    probe.fit(X[:cut], y[:cut], sample_weight=w[:cut], eval_set=[(X[cut:], y[cut:])], verbose=False)
    model = cls(**{**p, "n_estimators": probe.best_iteration + 1})
    return model.fit(X, y, sample_weight=w)


def fit_xgb_result(train: pd.DataFrame, features: list[str] = FEATURES,
                   calibrate: bool = False, tune: bool = False) -> Predictor:
    """
    XGBoost H/D/A classifier. Missing values are handled natively by XGBoost.

    calibrate: fit per-class isotonic regression on out-of-sample predictions for the
               last season of the training data, then renormalise.
    tune:      pick max_depth / min_child_weight by log-loss on that same last season.
    """
    y = labels(train)
    inner_tr, inner_va, y_va = train.iloc[:-SEASON], train.iloc[-SEASON:], y[-SEASON:]
    params: dict = {}
    if tune:
        grid = [dict(zip(TUNING_GRID, v)) for v in itertools.product(*TUNING_GRID.values())]
        scores = [log_loss(y_va, _fit_xgb(inner_tr, features, y[:-SEASON], "multi:softprob", **g)
                           .predict_proba(inner_va[features]), labels=[0, 1, 2]) for g in grid]
        params = grid[int(np.argmin(scores))]

    isotonic = []
    if calibrate:
        raw_va = _fit_xgb(inner_tr, features, y[:-SEASON], "multi:softprob", **params) \
            .predict_proba(inner_va[features])
        isotonic = [IsotonicRegression(out_of_bounds="clip").fit(raw_va[:, c], y_va == c)
                    for c in range(3)]

    model = _fit_xgb(train, features, y, "multi:softprob", **params)

    def predict(df: pd.DataFrame) -> np.ndarray:
        p = model.predict_proba(df[features])
        if isotonic:
            p = np.column_stack([iso.predict(p[:, c]) for c, iso in enumerate(isotonic)]) + 1e-6
            p /= p.sum(axis=1, keepdims=True)
        return p

    return predict


def fit_logistic(train: pd.DataFrame, features: list[str]) -> Predictor:
    """Multinomial logistic regression (median-imputed, standardised)."""
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))
    pipe.fit(train[features], labels(train))
    return lambda df: pipe.predict_proba(df[features])


def fit_base_rates(train: pd.DataFrame) -> Predictor:
    """Always predicts the training-set frequency of H / D / A."""
    freq = np.bincount(labels(train), minlength=3) / len(train)
    return lambda df: np.tile(freq, (len(df), 1))


def bookmakers(df: pd.DataFrame) -> np.ndarray:
    """Margin-free consensus of Bet365 and the market average (no training)."""
    p = df[ODDS_FEATURES].to_numpy()
    return p / p.sum(axis=1, keepdims=True)


def fit_goals(train: pd.DataFrame, features: list[str] = FEATURES) -> Callable:
    """Two Poisson XGBoost regressors -> expected home and away goals."""
    home = _fit_xgb(train, features, train["target_home_goals"].to_numpy(), "count:poisson")
    away = _fit_xgb(train, features, train["target_away_goals"].to_numpy(), "count:poisson")
    return lambda df: (home.predict(df[features]), away.predict(df[features]))


def score_grid(lam_home: float, lam_away: float, max_goals: int = 7) -> np.ndarray:
    """P(home scores i, away scores j), independent Poissons, renormalised to the grid."""
    g = np.outer(poisson.pmf(range(max_goals), lam_home), poisson.pmf(range(max_goals), lam_away))
    return g / g.sum()


def draw_aware_picks(probs: np.ndarray, margin: float) -> np.ndarray:
    """Pick D when P(D) is within `margin` of the favourite, else the argmax."""
    picks = probs.argmax(axis=1)
    picks[probs[:, 1] >= probs.max(axis=1) - margin] = 1
    return picks
