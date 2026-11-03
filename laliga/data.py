"""Load and download raw match data from football-data.co.uk."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIXTURES_FILE = RAW_DIR / "fixtures.csv"

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Older seasons name the market-average odds BbAv*, newer ones Avg*.
ODDS_ALIASES = {"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"}
KEEP = ["FTHG", "FTAG", "HST", "AST", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"]


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _fetch(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return pd.read_csv(io.BytesIO(resp.read()), encoding="utf-8-sig")


def download_latest() -> None:
    """Refresh the current season's results and the upcoming fixtures."""
    start = current_season_start()
    code = f"{start % 100:02d}{(start + 1) % 100:02d}"
    try:
        results = _fetch(SEASON_URL.format(code=code)).dropna(subset=["HomeTeam"])
        results.to_csv(RAW_DIR / f"{season_label(start)}.csv", index=False)
        print(f"Downloaded {len(results)} results for {season_label(start)}")
    except urllib.error.HTTPError as e:  # pre-season: the new file doesn't exist yet
        print(f"No results file for {season_label(start)} yet ({e.code})")
    fixtures = _fetch(FIXTURES_URL)
    fixtures = fixtures[fixtures["Div"] == "SP1"]
    fixtures.to_csv(FIXTURES_FILE, index=False)
    print(f"Downloaded {len(fixtures)} La Liga fixtures")


def _standardise(raw: pd.DataFrame, season: str, future: bool) -> pd.DataFrame:
    raw = raw.rename(columns=ODDS_ALIASES).dropna(subset=["HomeTeam"])
    when = raw["Date"].astype(str) + " " + raw.get("Time", pd.Series("00:00", index=raw.index)).astype(str)
    out = pd.DataFrame({
        "match_datetime": pd.to_datetime(when, dayfirst=True, format="mixed", errors="coerce"),
        "home_team": raw["HomeTeam"].str.strip(),
        "away_team": raw["AwayTeam"].str.strip(),
        "FTR": raw["FTR"] if "FTR" in raw else np.nan,
    })
    for c in KEEP:
        out[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else np.nan
    out["season"] = season
    out["is_future_match"] = int(future)
    return out


def load_matches() -> pd.DataFrame:
    """All played matches (every season file) plus upcoming fixtures not yet played."""
    frames = [
        _standardise(pd.read_csv(p, encoding="utf-8-sig"), p.stem, future=False)
        for p in sorted(RAW_DIR.glob("20??-??.csv"))
    ]
    played = pd.concat(frames, ignore_index=True)
    if FIXTURES_FILE.exists():
        fix = pd.read_csv(FIXTURES_FILE, encoding="utf-8-sig")
        fix = _standardise(fix[fix["Div"] == "SP1"], "", future=True)
        fix["season"] = [season_label(d.year if d.month >= 7 else d.year - 1) for d in fix["match_datetime"]]
        # fixtures.csv can still list matches that are already in the results file
        done = set(zip(played["home_team"], played["away_team"], played["season"]))
        fix = fix[[k not in done for k in zip(fix["home_team"], fix["away_team"], fix["season"])]]
        played = pd.concat([played, fix], ignore_index=True)
    return played.sort_values(["match_datetime", "home_team"]).reset_index(drop=True)
