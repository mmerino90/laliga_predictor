"""
La Liga match predictor: one command, end to end.

    python run.py            # features -> walk-forward evaluation -> predictions -> plots -> site data
    python run.py --update   # download the latest results and fixtures first
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from laliga import evaluate as E, model as M, plots
from laliga.data import ROOT, download_latest, load_matches
from laliga.features import build_features

RESULTS_DIR = ROOT / "results"
SITE_DATA = ROOT / "site" / "data.json"
FORWARD_SNAPSHOT = ROOT / "data" / "forward_test" / "predictions_made_2026-04-09.csv"
PRODUCTION = "XGBoost"  # best ML variant on walk-forward log-loss/RPS (see results/metrics.json)
SHOWN = ["Base rates", "ELO logistic", "Logistic (all features)", "XGBoost (no odds)",
         "XGBoost", "XGBoost + isotonic", "XGBoost tuned + isotonic", "Bookmakers"]


def r(x, nd=4):
    return round(float(x), nd)


def predict_upcoming(features: pd.DataFrame) -> list[dict]:
    played = features[features["is_future_match"] == 0]
    future = features[features["is_future_match"] == 1]
    if future.empty:
        return []
    p = M.fit_xgb_result(played)(future)
    xg_home, xg_away = M.fit_goals(played)(future)
    book = M.bookmakers(future)
    out = []
    for i, row in enumerate(future.itertuples()):
        grid = M.score_grid(xg_home[i], xg_away[i])
        top = np.argsort(grid, axis=None)[::-1][:3]
        out.append({
            "kickoff": row.match_datetime.strftime("%Y-%m-%d %H:%M"),
            "home": row.home_team, "away": row.away_team,
            "model": [r(v, 3) for v in p[i]],
            "bookmakers": None if np.isnan(book[i]).any() else [r(v, 3) for v in book[i]],
            "xg": [r(xg_home[i], 2), r(xg_away[i], 2)],
            "scores": [[f"{k // grid.shape[1]}-{k % grid.shape[1]}", r(grid.flat[k], 3)] for k in top],
        })
    return out


def track_record(preds: pd.DataFrame, season: str) -> dict:
    s = preds[preds["season"] == season].sort_values("match_datetime", ascending=False)
    if s.empty:
        return {}
    y = M.labels(s)
    model, book = E.probs(s, PRODUCTION), E.probs(s, "Bookmakers")
    return {
        "season": season, "n": len(s),
        "model_accuracy": r((model.argmax(1) == y).mean()),
        "bookmaker_accuracy": r((book.argmax(1) == y).mean()),
        "model_rps": r(E.rps(y, model).mean()), "bookmaker_rps": r(E.rps(y, book).mean()),
        "matches": [{
            "date": row.match_datetime.strftime("%Y-%m-%d"), "home": row.home_team, "away": row.away_team,
            "score": f"{int(row.target_home_goals)}-{int(row.target_away_goals)}", "result": row.target_ftr,
            "model": [r(v, 3) for v in model[i]], "bookmakers": [r(v, 3) for v in book[i]],
            "pick": M.RESULTS[model[i].argmax()],
        } for i, row in enumerate(s.itertuples())],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="download the latest results and fixtures first")
    args = ap.parse_args()

    if args.update:
        download_latest()

    features = build_features(load_matches())
    played = features[features["is_future_match"] == 0]
    current = played["season"].max()
    print(f"{len(played)} played matches ({played['season'].min()} to {current}), "
          f"{int(features['is_future_match'].sum())} upcoming")

    print("Walk-forward evaluation ...")
    preds = E.walk_forward(features)
    complete = preds[preds["season"] < current]
    summary = E.summarise(complete).loc[SHOWN]
    print(f"\n{len(complete)} matches, {complete['season'].nunique()} complete seasons\n")
    print(summary.round(4).to_string())

    forward = E.forward_test(pd.read_csv(FORWARD_SNAPSHOT, encoding="utf-8-sig"), features, preds) \
        if FORWARD_SNAPSHOT.exists() else None

    RESULTS_DIR.mkdir(exist_ok=True)
    preds.to_csv(RESULTS_DIR / "walk_forward_predictions.csv", index=False)
    metrics = {
        "walk_forward": {"matches": len(complete), "seasons": sorted(complete["season"].unique()),
                         "models": summary.round(4).reset_index().to_dict("records")},
        "per_season_accuracy": E.per_season(complete, ["XGBoost", "Bookmakers", "ELO logistic"]).round(4).to_dict("index"),
        "draw_rule": {k: r(v) for k, v in E.draw_rule(complete, PRODUCTION).items()},
        "goals_mae": {k: r(v) for k, v in E.goals_mae(complete).items()},
        "forward_test": forward,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    plots.per_season_accuracy(complete, RESULTS_DIR / "per_season_accuracy.png")
    plots.calibration(complete, PRODUCTION, RESULTS_DIR / "calibration.png")
    plots.elo_race(features, current, RESULTS_DIR / "elo_current_season.png")

    print("\nPredicting upcoming fixtures ...")
    site = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": PRODUCTION,
        "upcoming": predict_upcoming(features),
        "track_record": track_record(preds, current),
        "walk_forward": metrics["walk_forward"],
        "forward_test": forward,
    }
    SITE_DATA.write_text(json.dumps(site, indent=1, default=float), encoding="utf-8")
    for m in site["upcoming"]:
        print(f"  {m['kickoff']}  {m['home']:>14} v {m['away']:<14}  H/D/A {m['model']}")
    print(f"\nWrote {RESULTS_DIR.relative_to(ROOT)}/ and {SITE_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
