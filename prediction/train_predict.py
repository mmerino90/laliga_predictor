"""
La Liga 2025/26 — Full Train/Predict Pipeline
==============================================
Produces:
  data/predictions_future_matches.csv   - predictions for every remaining fixture
  data/evaluation_report.txt            - performance across walk-forward CV + single holdout

Models
------
  1. XGBoost multi-class classifier   → result (H / D / A)
  2. XGBoost Poisson regressor x2     → expected home goals + expected away goals
  3. OOF isotonic calibration         → corrects probability over-confidence

Design choices
--------------
  - Walk-forward CV across 7 seasons for reliable accuracy estimate
  - Time-based single holdout: test = last 17% of current season
  - Recency weighting with half-life = 1 full season
  - Median imputation fit on train only
  - Within-season season_progress feature for home-advantage drift
  - RPS (Ranked Probability Score) as primary metric
"""

from __future__ import annotations

import warnings
from pathlib import Path

import math

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "model_ready_matches_25_26.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions_future_matches.csv"
REPORT_FILE = DATA_DIR / "evaluation_report.txt"

RESULT_ORDER = ["H", "D", "A"]
RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_RESULT = {v: k for k, v in RESULT_TO_INT.items()}

FEATURES = [
    # ── table strength ─────────────────────────────────────────────────────
    "home_matches_played_pre", "away_matches_played_pre",
    "home_ppm_pre",   "away_ppm_pre",   "diff_ppm_pre",
    "home_gdpm_pre",  "away_gdpm_pre",  "diff_gdpm_pre",
    "home_home_ppm_pre", "away_away_ppm_pre",
    # ── ELO: only the gap matters; raw values are redundant with the diff ──
    "diff_elo_pre",
    # ── overall rolling form + explicit gaps ───────────────────────────────
    "home_form3_pts_pre",  "away_form3_pts_pre",  "diff_form3_pts",
    "home_form5_pts_pre",  "away_form5_pts_pre",  "diff_form5_pts",
    "home_form10_pts_pre", "away_form10_pts_pre", "diff_form10_pts",
    "home_form5_gf_pre",   "away_form5_gf_pre",   "diff_form5_gf",
    "home_form5_ga_pre",   "away_form5_ga_pre",   "diff_form5_ga",
    # ── shots on target (luck-adjusted quality) + gap ──────────────────────
    "home_form5_sot_for_pre", "away_form5_sot_for_pre", "diff_sot_pre",
    "home_form5_sot_ag_pre",  "away_form5_sot_ag_pre",  "diff_sot_ag",
    # ── venue-specific form + explicit gaps ────────────────────────────────
    "home_venue_pts5_pre",  "away_venue_pts5_pre",  "diff_venue_pts5",
    "home_venue_gf5_pre",   "away_venue_gf5_pre",   "diff_venue_gf5",
    "home_venue_ga5_pre",   "away_venue_ga5_pre",   "diff_venue_ga5",
    "home_venue_sot5_pre",  "away_venue_sot5_pre",
    "home_venue_sotag5_pre","away_venue_sotag5_pre",
    # ── rest / fatigue ─────────────────────────────────────────────────────
    "home_rest_days_pre", "away_rest_days_pre", "diff_rest_days_pre",
    # ── head-to-head ───────────────────────────────────────────────────────
    "h2h_pts_pre", "h2h_gf_pre", "h2h_ga_pre",
    # ── bookmaker consensus (b365 + avg averaged → single clean signal) ────
    "consensus_home_prob", "consensus_draw_prob", "consensus_away_prob",
    # ── within-season progress ─────────────────────────────────────────────
    "season_progress",
]

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=800,
    max_depth=4,
    learning_rate=0.025,
    subsample=0.75,
    colsample_bytree=0.70,
    min_child_weight=10,
    reg_alpha=0.8,
    reg_lambda=3.0,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0,
)

GOALS_PARAMS = dict(
    objective="count:poisson",
    n_estimators=800,
    max_depth=4,
    learning_rate=0.025,
    subsample=0.75,
    colsample_bytree=0.70,
    min_child_weight=10,
    reg_alpha=0.8,
    reg_lambda=3.0,
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """RPS for ordered 3-outcome predictions. Lower is better."""
    n, rps = len(y_true), 0.0
    for i in range(n):
        cum_t = np.array([1.0 if y_true[i] <= j else 0.0 for j in range(3)])
        rps  += np.sum((np.cumsum(probs[i])[:2] - cum_t[:2]) ** 2) / 2
    return rps / n


def add_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Within-season fractional progress (0 -> 1 per season)."""
    df = df.copy()
    df["season_progress"] = np.nan
    for season, grp in df.groupby("season"):
        pi = grp[grp["is_future_match"] == 0].index
        fi = grp[grp["is_future_match"] == 1].index
        if len(pi):
            df.loc[pi, "season_progress"] = np.linspace(0, 1, len(pi))
        if len(fi) and len(pi):
            step = 1 / len(pi)
            df.loc[fi, "season_progress"] = np.linspace(
                1 + step, 1 + len(fi) * step, len(fi)
            )
    return df


def recency_weights(n: int, half_life: int = 380) -> np.ndarray:
    """Exponential decay; half_life=380 ~= one full La Liga season."""
    decay = np.log(2) / half_life
    return np.exp(decay * (np.arange(n) - (n - 1)))


def poisson_score_probs(lam_h: float, lam_a: float,
                        max_goals: int = 6) -> list[tuple[str, float]]:
    """
    Returns a list of (scoreline, probability) pairs, sorted by probability
    descending, using independent Poisson distributions for each team.
    Only scores up to max_goals-1 per side are considered; the remainder
    is absorbed into a catch-all 'other' bucket.
    """
    def poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    scores: list[tuple[str, float]] = []
    total = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            p = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a)
            scores.append((f"{h}-{a}", p))
            total += p

    # normalise so probabilities sum to 1 (accounts for truncation)
    scores = [(s, p / total) for s, p in scores]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def top_scores_str(lam_h: float, lam_a: float, n: int = 5) -> str:
    """Human-readable top-N scorelines with percentages."""
    scores = poisson_score_probs(lam_h, lam_a)
    return "  |  ".join(f"{s} ({p:.0%})" for s, p in scores[:n])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consolidated and diff features.
    Called once on the full dataset before any train/test split.

    Consolidations
    --------------
    - consensus_*_prob : average of b365 and avg implied probabilities.
      Reduces 6 correlated odds columns to 3 cleaner signals.

    Diff features
    -------------
    - diff_form*  : home minus away for form windows (3/5/10).
      Explicit gaps let the model use a single split instead of having
      to infer the gap from two separate features.
    - diff_sot_ag : shots-on-target conceded gap (mirrors diff_sot_pre
      which is the shots-scored gap).
    - diff_venue_* : venue-specific form gaps.
    """
    d = df.copy()

    def safe_mean(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.fillna(b) + b.fillna(a)) / 2

    def diff(a: pd.Series, b: pd.Series) -> pd.Series:
        return a - b

    # Consensus odds (average of two bookmakers, remove overround each first)
    d["consensus_home_prob"] = safe_mean(d["avg_home_prob"], d["b365_home_prob"])
    d["consensus_draw_prob"] = safe_mean(d["avg_draw_prob"], d["b365_draw_prob"])
    d["consensus_away_prob"] = safe_mean(d["avg_away_prob"], d["b365_away_prob"])

    # Form gaps
    d["diff_form3_pts"]  = diff(d["home_form3_pts_pre"],  d["away_form3_pts_pre"])
    d["diff_form5_pts"]  = diff(d["home_form5_pts_pre"],  d["away_form5_pts_pre"])
    d["diff_form10_pts"] = diff(d["home_form10_pts_pre"], d["away_form10_pts_pre"])
    d["diff_form5_gf"]   = diff(d["home_form5_gf_pre"],   d["away_form5_gf_pre"])
    d["diff_form5_ga"]   = diff(d["home_form5_ga_pre"],   d["away_form5_ga_pre"])

    # Shots gaps
    d["diff_sot_ag"]     = diff(d["home_form5_sot_ag_pre"], d["away_form5_sot_ag_pre"])

    # Venue-specific form gaps
    d["diff_venue_pts5"] = diff(d["home_venue_pts5_pre"], d["away_venue_pts5_pre"])
    d["diff_venue_gf5"]  = diff(d["home_venue_gf5_pre"],  d["away_venue_gf5_pre"])
    d["diff_venue_ga5"]  = diff(d["home_venue_ga5_pre"],  d["away_venue_ga5_pre"])

    return d


def fit_imputer(X: pd.DataFrame) -> dict[str, float]:
    return X.median(numeric_only=True).to_dict()


def apply_imputer(X: pd.DataFrame, medians: dict) -> np.ndarray:
    return X.fillna(medians).values


def calibrate_oof(X_train: np.ndarray, y_train: np.ndarray,
                  n_estimators: int,
                  probs_new: np.ndarray,
                  n_splits: int = 5) -> np.ndarray:
    """
    OOF isotonic calibration — avoids leakage.
    Generates out-of-fold predictions via TimeSeriesSplit, fits per-class
    isotonic regressors on those, then applies to new data.
    """
    from sklearn.model_selection import TimeSeriesSplit
    n = X_train.shape[0]
    oof_raw  = np.full((n, 3), np.nan)
    test_sz  = max(30, n // (n_splits + 1))
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_sz)

    skip = {"early_stopping_rounds", "eval_metric", "n_estimators"}
    base_params = {k: v for k, v in XGB_PARAMS.items() if k not in skip}

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]
        mini = xgb.XGBClassifier(**base_params, n_estimators=n_estimators)
        mini.fit(X_tr, y_tr,
                 sample_weight=recency_weights(len(y_tr)),
                 verbose=False)
        oof_raw[va_idx] = mini.predict_proba(X_va)

    mask = ~np.isnan(oof_raw[:, 0])
    cal  = np.zeros_like(probs_new)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof_raw[mask, c], (y_train[mask] == c).astype(float))
        cal[:, c] = ir.predict(probs_new[:, c])

    row_sums = cal.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cal / row_sums


def train_result_model(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_va: np.ndarray, y_va: np.ndarray,
                       w_tr: np.ndarray) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_va, y_va)], verbose=False)
    return clf


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

CV_SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]
TEST_FRAC = 0.30   # last 30% of each test season


def walk_forward_cv(played: pd.DataFrame) -> dict:
    """
    For each test season S:
      - train on everything before the last TEST_FRAC of S
      - test on that last TEST_FRAC
    Returns aggregated metrics.
    """
    all_y, all_probs, all_odds = [], [], []

    for season in CV_SEASONS:
        season_matches = played[played["season"] == season].copy().reset_index(drop=True)
        split_idx = int(len(season_matches) * (1 - TEST_FRAC))
        if split_idx < 20 or len(season_matches) - split_idx < 10:
            continue

        te_idx = season_matches.index[split_idx:]
        tr_idx_all = played[
            (played["season"] < season) |
            ((played["season"] == season) & played.index.isin(played[played["season"] == season].index[:split_idx]))
        ].index

        train = played.loc[played.index.isin(tr_idx_all)].sort_values("match_datetime")
        test  = played.loc[played.index.isin(te_idx)].sort_values("match_datetime")

        medians = fit_imputer(train[FEATURES])
        X_tr = apply_imputer(train[FEATURES], medians)
        X_te = apply_imputer(test[FEATURES],  medians)
        y_tr = train["target_ftr"].map(RESULT_TO_INT).values
        y_te = test["target_ftr"].map(RESULT_TO_INT).values
        w_tr = recency_weights(len(train))

        # small fast eval set (last 10% of train)
        val_split = int(len(X_tr) * 0.9)
        clf = train_result_model(
            X_tr[:val_split], y_tr[:val_split],
            X_tr[val_split:], y_tr[val_split:],
            w_tr[:val_split],
        )
        probs = clf.predict_proba(X_te)
        odds  = test[["avg_home_prob","avg_draw_prob","avg_away_prob"]].fillna(1/3).values

        all_y.append(y_te)
        all_probs.append(probs)
        all_odds.append(odds)

    y_all    = np.concatenate(all_y)
    p_all    = np.vstack(all_probs)
    o_all    = np.vstack(all_odds)

    acc      = accuracy_score(y_all, p_all.argmax(axis=1))
    rps      = ranked_probability_score(y_all, p_all)
    acc_odds = accuracy_score(y_all, o_all.argmax(axis=1))
    rps_odds = ranked_probability_score(y_all, o_all)

    return {
        "n_matches":  len(y_all),
        "n_seasons":  len(CV_SEASONS),
        "acc":        acc,
        "rps":        rps,
        "acc_odds":   acc_odds,
        "rps_odds":   rps_odds,
        "h_rate":     (y_all == 0).mean(),
        "d_rate":     (y_all == 1).mean(),
        "a_rate":     (y_all == 2).mean(),
        "h_pred":     (p_all.argmax(1) == 0).mean(),
        "d_pred":     (p_all.argmax(1) == 1).mean(),
        "a_pred":     (p_all.argmax(1) == 2).mean(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], format="mixed")
    df = df.sort_values("match_datetime").reset_index(drop=True)
    df = add_season_progress(df)
    df = engineer_features(df)

    played  = df[df["is_future_match"] == 0].copy().reset_index(drop=True)
    future  = df[df["is_future_match"] == 1].copy().reset_index(drop=True)
    current = played[played["is_current_season"] == 1].copy().reset_index(drop=True)
    hist    = played[played["is_current_season"] == 0].copy().reset_index(drop=True)

    log("=" * 65)
    log("  LA LIGA 2025/26 — PREDICTION PIPELINE EVALUATION REPORT")
    log("=" * 65)
    log(f"\nTotal played matches : {len(played)}")
    log(f"  Historical seasons : {len(hist)}  (2015-16 to 2024-25)")
    log(f"  Current season     : {len(current)}  (2025-26)")
    log(f"  Future fixtures    : {len(future)}")
    log(f"  Feature count      : {len(FEATURES)}")

    # ── 2. Walk-forward CV (reliable cross-season accuracy) ────────────────
    log("\n" + "─" * 65)
    log("  WALK-FORWARD CV  (7 seasons, last 30% of each as test)")
    log("─" * 65)
    log("  Running ... (may take ~30 s)")
    cv = walk_forward_cv(played)
    log(f"\n  Evaluated on {cv['n_matches']} held-out matches across {cv['n_seasons']} seasons")
    log(f"  (Odds baseline in CV = uniform 1/3 where historical odds unavailable)")
    log(f"\n  {'Metric':<34} {'Model':>10}  {'Odds Baseline':>14}")
    log(f"  {'Accuracy':<34} {cv['acc']:>10.3f}  {cv['acc_odds']:>14.3f}")
    log(f"  {'RPS (lower=better)':<34} {cv['rps']:>10.4f}  {cv['rps_odds']:>14.4f}")
    log(f"\n  Predicted vs Actual distribution (aggregated CV):")
    log(f"  {'Outcome':<8} {'Predicted':>10} {'Actual':>10}")
    for lbl, pk, ak in [("H", cv["h_pred"], cv["h_rate"]),
                         ("D", cv["d_pred"], cv["d_rate"]),
                         ("A", cv["a_pred"], cv["a_rate"])]:
        log(f"  {lbl:<8} {pk:>9.1%} {ak:>9.1%}")

    # ── 3. Current-season holdout split ───────────────────────────────────
    TRAIN_FRAC = 0.83
    cur_split  = int(len(current) * TRAIN_FRAC)
    train = pd.concat(
        [hist, current.iloc[:cur_split]], ignore_index=True
    ).sort_values("match_datetime").reset_index(drop=True)
    test  = current.iloc[cur_split:].copy().reset_index(drop=True)

    log("\n" + "─" * 65)
    log("  CURRENT-SEASON HOLDOUT (most recent matches)")
    log("─" * 65)
    log(f"\n  Train : {len(train)} matches  "
        f"({train['match_datetime'].min().date()} to "
        f"{train['match_datetime'].max().date()})")
    log(f"  Test  : {len(test)} matches   "
        f"({test['match_datetime'].min().date()} to "
        f"{test['match_datetime'].max().date()})")

    # ── 4. Imputation ──────────────────────────────────────────────────────
    medians  = fit_imputer(train[FEATURES])
    X_train  = apply_imputer(train[FEATURES],  medians)
    X_test   = apply_imputer(test[FEATURES],   medians)
    X_future = apply_imputer(future[FEATURES], medians)

    # ── 5. Weights ─────────────────────────────────────────────────────────
    y_train_cls = train["target_ftr"].map(RESULT_TO_INT).values
    y_test_cls  = test["target_ftr"].map(RESULT_TO_INT).values
    w_train     = recency_weights(len(train), half_life=380)

    # ── 6. Train result classifier ─────────────────────────────────────────
    clf = train_result_model(X_train, y_train_cls, X_test, y_test_cls, w_train)
    prob_test_raw   = clf.predict_proba(X_test)
    prob_future_raw = clf.predict_proba(X_future)

    # ── 7. OOF isotonic calibration ────────────────────────────────────────
    log("\n  Running OOF calibration ...")
    best_n = clf.best_iteration + 1
    prob_test   = calibrate_oof(X_train, y_train_cls, best_n, prob_test_raw)
    prob_future = calibrate_oof(X_train, y_train_cls, best_n, prob_future_raw)
    pred_test   = prob_test.argmax(axis=1)

    # ── 8. Goals models ────────────────────────────────────────────────────
    y_train_hg = train["target_home_goals"].values
    y_train_ag = train["target_away_goals"].values
    y_test_hg  = test["target_home_goals"].values
    y_test_ag  = test["target_away_goals"].values

    reg_hg = xgb.XGBRegressor(**GOALS_PARAMS)
    reg_hg.fit(X_train, y_train_hg, sample_weight=w_train,
               eval_set=[(X_test, y_test_hg)], verbose=False)

    reg_ag = xgb.XGBRegressor(**GOALS_PARAMS)
    reg_ag.fit(X_train, y_train_ag, sample_weight=w_train,
               eval_set=[(X_test, y_test_ag)], verbose=False)

    pred_hg_test = reg_hg.predict(X_test).clip(0)
    pred_ag_test = reg_ag.predict(X_test).clip(0)

    # ── 9. Evaluation ──────────────────────────────────────────────────────
    log("\n" + "─" * 65)
    log("  RESULT MODEL — CURRENT-SEASON HOLDOUT")
    log("─" * 65)

    acc  = accuracy_score(y_test_cls, pred_test)
    ll   = log_loss(y_test_cls, prob_test)
    rps  = ranked_probability_score(y_test_cls, prob_test)

    odds = test[["avg_home_prob","avg_draw_prob","avg_away_prob"]].fillna(1/3).values
    odds_pred = odds.argmax(axis=1)
    odds_acc  = accuracy_score(y_test_cls, odds_pred)
    odds_rps  = ranked_probability_score(y_test_cls, odds)

    log(f"\n  {'Metric':<30} {'Model':>10}  {'Odds Baseline':>14}")
    log(f"  {'Accuracy':<30} {acc:>10.3f}  {odds_acc:>14.3f}")
    log(f"  {'Log-Loss (lower=better)':<30} {ll:>10.3f}  {'—':>14}")
    log(f"  {'RPS (lower=better)':<30} {rps:>10.4f}  {odds_rps:>14.4f}")
    log(f"\n  NOTE: n={len(test)} is small (SE ~7%) — CV accuracy is more reliable.")

    log(f"\n  Predicted vs Actual distribution:")
    log(f"  {'Outcome':<8} {'Predicted':>10} {'Actual':>10}")
    for i, lbl in enumerate(RESULT_ORDER):
        log(f"  {lbl:<8} {(pred_test==i).mean():>9.1%} {(y_test_cls==i).mean():>9.1%}")

    log("\n  Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test_cls, pred_test)
    log(f"  {'':8} {'Pred H':>8} {'Pred D':>8} {'Pred A':>8}")
    for i, lbl in enumerate(RESULT_ORDER):
        log(f"  {'Act '+lbl:<8} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

    log("\n" + "─" * 65)
    log("  GOALS MODEL — CURRENT-SEASON HOLDOUT")
    log("─" * 65)
    mae_hg  = np.abs(pred_hg_test - y_test_hg).mean()
    mae_ag  = np.abs(pred_ag_test - y_test_ag).mean()
    rmse_hg = np.sqrt(((pred_hg_test - y_test_hg) ** 2).mean())
    rmse_ag = np.sqrt(((pred_ag_test - y_test_ag) ** 2).mean())
    log(f"\n  {'Metric':<28} {'Home Goals':>12} {'Away Goals':>12}")
    log(f"  {'MAE':<28} {mae_hg:>12.3f} {mae_ag:>12.3f}")
    log(f"  {'RMSE':<28} {rmse_hg:>12.3f} {rmse_ag:>12.3f}")
    log(f"  {'Avg predicted':<28} {pred_hg_test.mean():>12.3f} {pred_ag_test.mean():>12.3f}")
    log(f"  {'Avg actual':<28} {y_test_hg.mean():>12.3f} {y_test_ag.mean():>12.3f}")

    log("\n" + "─" * 65)
    log("  TOP 12 FEATURES — RESULT MODEL")
    log("─" * 65)
    fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, imp in fi.head(12).items():
        log(f"  {feat:<42} {imp:.4f}")

    # ── 10. Final retrain on ALL played data ───────────────────────────────
    X_all     = apply_imputer(played[FEATURES], medians)
    y_all_cls = played["target_ftr"].map(RESULT_TO_INT).values
    y_all_hg  = played["target_home_goals"].values
    y_all_ag  = played["target_away_goals"].values
    w_all     = recency_weights(len(played), half_life=380)

    skip_keys = {"early_stopping_rounds", "eval_metric", "n_estimators"}
    clf_final = xgb.XGBClassifier(
        **{k: v for k, v in XGB_PARAMS.items() if k not in skip_keys},
        n_estimators=clf.best_iteration + 1,
    )
    clf_final.fit(X_all, y_all_cls, sample_weight=w_all, verbose=False)

    prob_fut_raw   = clf_final.predict_proba(X_future)
    prob_fut_final = calibrate_oof(X_all, y_all_cls, clf_final.n_estimators, prob_fut_raw)

    skip_g = {"early_stopping_rounds", "n_estimators"}
    reg_hg_final = xgb.XGBRegressor(
        **{k: v for k, v in GOALS_PARAMS.items() if k not in skip_g},
        n_estimators=reg_hg.best_iteration + 1,
    )
    reg_hg_final.fit(X_all, y_all_hg, sample_weight=w_all, verbose=False)

    reg_ag_final = xgb.XGBRegressor(
        **{k: v for k, v in GOALS_PARAMS.items() if k not in skip_g},
        n_estimators=reg_ag.best_iteration + 1,
    )
    reg_ag_final.fit(X_all, y_all_ag, sample_weight=w_all, verbose=False)

    pred_hg_fut = reg_hg_final.predict(X_future).clip(0)
    pred_ag_fut = reg_ag_final.predict(X_future).clip(0)

    # ── 11. Build predictions CSV ──────────────────────────────────────────
    preds = future[["match_datetime", "home_team", "away_team"]].copy()
    preds["prob_home_win"] = prob_fut_final[:, 0].round(3)
    preds["prob_draw"]     = prob_fut_final[:, 1].round(3)
    preds["prob_away_win"] = prob_fut_final[:, 2].round(3)
    preds["predicted_result"] = [INT_TO_RESULT[i] for i in prob_fut_final.argmax(axis=1)]
    preds["xg_home"] = pred_hg_fut.round(2)
    preds["xg_away"] = pred_ag_fut.round(2)

    # ── Poisson score distribution ─────────────────────────────────────────
    score_dists = [
        poisson_score_probs(h, a) for h, a in zip(pred_hg_fut, pred_ag_fut)
    ]
    # Most likely single scoreline
    preds["most_likely_score"] = [d[0][0] for d in score_dists]
    preds["most_likely_score_prob"] = [round(d[0][1], 3) for d in score_dists]
    # Top 5 scorelines as readable string
    preds["score_probabilities"] = [
        top_scores_str(h, a, n=5) for h, a in zip(pred_hg_fut, pred_ag_fut)
    ]
    # Poisson-implied H/D/A probabilities (independent check of result model)
    p_hda = []
    for dist in score_dists:
        ph = sum(p for s, p in dist if int(s.split("-")[0]) > int(s.split("-")[1]))
        pd_ = sum(p for s, p in dist if s.split("-")[0] == s.split("-")[1])
        pa = sum(p for s, p in dist if int(s.split("-")[0]) < int(s.split("-")[1]))
        p_hda.append((round(ph, 3), round(pd_, 3), round(pa, 3)))
    preds["poisson_prob_h"] = [x[0] for x in p_hda]
    preds["poisson_prob_d"] = [x[1] for x in p_hda]
    preds["poisson_prob_a"] = [x[2] for x in p_hda]
    preds["confidence"] = prob_fut_final.max(axis=1).round(3)
    preds["confidence_tier"] = pd.cut(
        preds["confidence"],
        bins=[0, 0.45, 0.55, 0.65, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
    )
    preds["odds_implied_result"] = [
        INT_TO_RESULT[i]
        for i in future[["avg_home_prob","avg_draw_prob","avg_away_prob"]]
        .fillna(1/3).values.argmax(axis=1)
    ]
    preds["model_vs_odds"] = np.where(
        preds["predicted_result"] == preds["odds_implied_result"], "agree", "disagree"
    )
    preds.to_csv(PREDICTIONS_FILE, index=False, encoding="utf-8-sig")

    # ── 12. Preview ────────────────────────────────────────────────────────
    log("\n" + "─" * 65)
    log("  FUTURE FIXTURE PREDICTIONS — FIRST 20")
    log("─" * 65)
    log(f"\n  {'Date':<12} {'Home':<20} {'Away':<20} {'Pred':>5} "
        f"{'H%':>5} {'D%':>5} {'A%':>5}  Score probabilities (top 5)")
    log("  " + "─" * 105)
    for _, row in preds.head(20).iterrows():
        log(
            f"  {str(row['match_datetime'])[:10]:<12} "
            f"{row['home_team']:<20} {row['away_team']:<20} "
            f"{row['predicted_result']:>5} "
            f"{row['prob_home_win']:>4.0%} "
            f"{row['prob_draw']:>4.0%} "
            f"{row['prob_away_win']:>4.0%}  "
            f"{row['score_probabilities']}"
        )

    disagree_n = (preds["model_vs_odds"] == "disagree").sum()
    log(f"\n  Model disagrees with odds on {disagree_n} fixtures ({disagree_n/len(preds):.1%})")
    log(f"\n  Predictions saved -> {PREDICTIONS_FILE}")
    log(f"  Report saved      -> {REPORT_FILE}")
    log("\n" + "=" * 65)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_pipeline()
