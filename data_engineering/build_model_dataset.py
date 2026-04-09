from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIXTURES_FILE = DATA_DIR / "fixtures_25_26.csv"
OUTPUT_FILE = DATA_DIR / "model_ready_matches_25_26.csv"

SEASON_FILES = [
    (DATA_DIR / "SP1 2015.csv",         "2015-16"),
    (DATA_DIR / "SP1 2016.csv",         "2016-17"),
    (DATA_DIR / "SP1 2017.csv",         "2017-18"),
    (DATA_DIR / "SP1 2018.csv",         "2018-19"),
    (DATA_DIR / "SP1 2019.csv",         "2019-20"),
    (DATA_DIR / "SP1 2020.csv",         "2020-21"),
    (DATA_DIR / "SP1 2021.csv",         "2021-22"),
    (DATA_DIR / "SP1 2022.csv",         "2022-23"),
    (DATA_DIR / "SP1 2023.csv",         "2023-24"),
    (DATA_DIR / "SP1 2024.csv",         "2024-25"),
    (DATA_DIR / "SP1 2025 updated.csv", "2025-26"),
]

TEAM_NAME_MAP = {
    "Deportivo Alavés": "Alaves",
    "Athletic Club":    "Ath Bilbao",
    "Atlético de Madrid": "Ath Madrid",
    "FC Barcelona":     "Barcelona",
    "Real Betis":       "Betis",
    "Elche CF":         "Elche",
    "Getafe CF":        "Getafe",
    "Girona FC":        "Girona",
    "Levante UD":       "Levante",
    "RCD Espanyol de Barcelona": "Espanol",
    "RCD Mallorca":     "Mallorca",
    "CA Osasuna":       "Osasuna",
    "Real Oviedo":      "Oviedo",
    "Real Sociedad":    "Sociedad",
    "Sevilla FC":       "Sevilla",
    "Valencia CF":      "Valencia",
    "Rayo Vallecano":   "Vallecano",
    "Villarreal CF":    "Villarreal",
}

# ELO constants
ELO_START   = 1500.0
ELO_K       = 20.0
HOME_ADV_ELO = 60.0   # home ground advantage in ELO points


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamState:
    """Season-scoped league-table stats (resets each season)."""
    matches: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0
    home_matches: int = 0
    home_points: int = 0
    home_gf: int = 0
    home_ga: int = 0
    away_matches: int = 0
    away_points: int = 0
    away_gf: int = 0
    away_ga: int = 0


def _deque5()  -> deque: return deque(maxlen=5)
def _deque3()  -> deque: return deque(maxlen=3)
def _deque10() -> deque: return deque(maxlen=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_team(team: Any) -> str:
    if pd.isna(team):
        return team
    t = str(team).strip()
    return TEAM_NAME_MAP.get(t, t)


def implied_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    if any(pd.isna(v) or v <= 0 for v in [h, d, a]):
        return np.nan, np.nan, np.nan
    inv = np.array([1/h, 1/d, 1/a], dtype=float)
    s = inv.sum()
    return float(inv[0]/s), float(inv[1]/s), float(inv[2]/s)


def rmean(q: deque) -> float:
    return float(np.mean(q)) if q else np.nan


def elo_expected(home_elo: float, away_elo: float) -> float:
    """Probability of home win implied by ELO difference + home advantage."""
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - HOME_ADV_ELO) / 400.0))


def elo_update(home_elo: float, away_elo: float, ftr: str) -> tuple[float, float]:
    """Return updated (home_elo, away_elo) after result ftr ∈ {H, D, A}."""
    exp = elo_expected(home_elo, away_elo)
    actual = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
    delta = ELO_K * (actual - exp)
    return home_elo + delta, away_elo - delta


# ---------------------------------------------------------------------------
# Season file loader
# ---------------------------------------------------------------------------

def load_season_file(path: Path, season: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")

    if "Time" in raw.columns:
        dt = pd.to_datetime(
            raw["Date"].astype(str) + " " + raw["Time"].astype(str),
            dayfirst=True, errors="coerce",
        )
    else:
        dt = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")

    def col(name: str) -> pd.Series:
        return pd.to_numeric(raw[name], errors="coerce") if name in raw.columns else pd.Series(np.nan, index=raw.index)

    return pd.DataFrame({
        "match_datetime":  dt,
        "home_team":  raw["HomeTeam"].map(normalize_team),
        "away_team":  raw["AwayTeam"].map(normalize_team),
        "FTHG":  col("FTHG"),
        "FTAG":  col("FTAG"),
        "FTR":   raw["FTR"],
        "HS":    col("HS"),   # home shots
        "AS":    col("AS"),   # away shots
        "HST":   col("HST"),  # home shots on target
        "AST":   col("AST"),  # away shots on target
        "AvgH":  col("AvgH"),
        "AvgD":  col("AvgD"),
        "AvgA":  col("AvgA"),
        "B365H": col("B365H"),
        "B365D": col("B365D"),
        "B365A": col("B365A"),
        "season":            season,
        "is_current_season": int(season == "2025-26"),
        "is_future_match":   0,
    })


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:

    season_frames = []
    for path, label in SEASON_FILES:
        if path.exists():
            season_frames.append(load_season_file(path, label))
        else:
            print(f"  [warn] {path.name} not found — skipping")

    _fix = pd.read_csv(FIXTURES_FILE, encoding="utf-8-sig")
    fixtures = pd.DataFrame({
        "match_datetime":  pd.to_datetime(_fix["Date"], dayfirst=True, errors="coerce"),
        "home_team":  _fix["Home Team"].map(normalize_team),
        "away_team":  _fix["Away Team"].map(normalize_team),
        "FTHG": np.nan, "FTAG": np.nan, "FTR": np.nan,
        "HS":   np.nan, "AS":   np.nan,
        "HST":  np.nan, "AST":  np.nan,
        "AvgH": np.nan, "AvgD": np.nan, "AvgA": np.nan,
        "B365H":np.nan, "B365D":np.nan, "B365A":np.nan,
        "season":            "2025-26",
        "is_current_season": 1,
        "is_future_match":   1,
    })

    all_matches = pd.concat(season_frames + [fixtures], ignore_index=True)
    all_matches = all_matches.sort_values(
        ["match_datetime", "home_team", "away_team"]
    ).reset_index(drop=True)

    # ── State that RESETS each season ─────────────────────────────────────
    team_state:       dict[str, TeamState] = defaultdict(TeamState)
    # overall rolling windows
    t_pts5:           dict[str, deque]     = defaultdict(_deque5)
    t_pts3:           dict[str, deque]     = defaultdict(_deque3)
    t_pts10:          dict[str, deque]     = defaultdict(_deque10)
    t_gf5:            dict[str, deque]     = defaultdict(_deque5)
    t_ga5:            dict[str, deque]     = defaultdict(_deque5)
    t_sot_for5:       dict[str, deque]     = defaultdict(_deque5)   # shots-on-target scored
    t_sot_ag5:        dict[str, deque]     = defaultdict(_deque5)   # shots-on-target conceded
    # VENUE-SPECIFIC rolling (home team at home / away team away)
    t_home_pts5:      dict[str, deque]     = defaultdict(_deque5)
    t_home_gf5:       dict[str, deque]     = defaultdict(_deque5)
    t_home_ga5:       dict[str, deque]     = defaultdict(_deque5)
    t_home_sot5:      dict[str, deque]     = defaultdict(_deque5)
    t_home_sotag5:    dict[str, deque]     = defaultdict(_deque5)
    t_away_pts5:      dict[str, deque]     = defaultdict(_deque5)
    t_away_gf5:       dict[str, deque]     = defaultdict(_deque5)
    t_away_ga5:       dict[str, deque]     = defaultdict(_deque5)
    t_away_sot5:      dict[str, deque]     = defaultdict(_deque5)
    t_away_sotag5:    dict[str, deque]     = defaultdict(_deque5)

    last_date:        dict[str, pd.Timestamp] = {}
    cur_season:       str | None = None

    # ── State that PERSISTS across seasons ────────────────────────────────
    elo:              dict[str, float]     = defaultdict(lambda: ELO_START)
    h2h_pts:          dict[tuple, deque]   = defaultdict(_deque5)
    h2h_gf:           dict[tuple, deque]   = defaultdict(_deque5)
    h2h_ga:           dict[tuple, deque]   = defaultdict(_deque5)

    rows: list[dict[str, Any]] = []

    for _, m in all_matches.iterrows():
        home   = m["home_team"]
        away   = m["away_team"]
        dt     = m["match_datetime"]
        season = m["season"]

        # Season reset
        if season != cur_season:
            team_state    = defaultdict(TeamState)
            t_pts5        = defaultdict(_deque5);  t_pts3   = defaultdict(_deque3)
            t_pts10       = defaultdict(_deque10)
            t_gf5         = defaultdict(_deque5);  t_ga5    = defaultdict(_deque5)
            t_sot_for5    = defaultdict(_deque5);  t_sot_ag5 = defaultdict(_deque5)
            t_home_pts5   = defaultdict(_deque5);  t_home_gf5  = defaultdict(_deque5)
            t_home_ga5    = defaultdict(_deque5);  t_home_sot5 = defaultdict(_deque5)
            t_home_sotag5 = defaultdict(_deque5)
            t_away_pts5   = defaultdict(_deque5);  t_away_gf5  = defaultdict(_deque5)
            t_away_ga5    = defaultdict(_deque5);  t_away_sot5 = defaultdict(_deque5)
            t_away_sotag5 = defaultdict(_deque5)
            last_date     = {}
            cur_season    = season

        hs  = team_state[home]
        as_ = team_state[away]

        # Points per match
        home_ppm = hs.points / hs.matches        if hs.matches       > 0 else np.nan
        away_ppm = as_.points / as_.matches       if as_.matches      > 0 else np.nan
        # Goal difference per match
        home_gdpm = (hs.gf - hs.ga) / hs.matches if hs.matches       > 0 else np.nan
        away_gdpm = (as_.gf-as_.ga) / as_.matches if as_.matches      > 0 else np.nan
        # Venue-specific PPM
        h_home_ppm = hs.home_points / hs.home_matches   if hs.home_matches  > 0 else np.nan
        a_away_ppm = as_.away_points / as_.away_matches  if as_.away_matches > 0 else np.nan

        # Rest days
        home_rest = (dt - last_date[home]).days if home in last_date and pd.notna(dt) else np.nan
        away_rest = (dt - last_date[away]).days if away in last_date and pd.notna(dt) else np.nan

        # ELO (before update)
        home_elo_pre = elo[home]
        away_elo_pre = elo[away]

        # H2H
        h2h_key = (home, away)

        # Odds
        avg_h, avg_d, avg_a     = implied_probs(m["AvgH"],  m["AvgD"],  m["AvgA"])
        b365_h, b365_d, b365_a  = implied_probs(m["B365H"], m["B365D"], m["B365A"])

        row: dict[str, Any] = {
            # ── identifiers ────────────────────────────────────────────
            "match_datetime":           dt,
            "home_team":                home,
            "away_team":                away,
            "season":                   season,
            "is_current_season":        int(m["is_current_season"]),
            "is_future_match":          int(m["is_future_match"]),
            # ── table strength ─────────────────────────────────────────
            "home_matches_played_pre":  hs.matches,
            "away_matches_played_pre":  as_.matches,
            "home_points_pre":          hs.points,
            "away_points_pre":          as_.points,
            "home_ppm_pre":             home_ppm,
            "away_ppm_pre":             away_ppm,
            "diff_ppm_pre":             home_ppm - away_ppm if pd.notna(home_ppm) and pd.notna(away_ppm) else np.nan,
            "home_gd_pre":              hs.gf  - hs.ga,
            "away_gd_pre":              as_.gf - as_.ga,
            "home_gdpm_pre":            home_gdpm,
            "away_gdpm_pre":            away_gdpm,
            "diff_gdpm_pre":            home_gdpm - away_gdpm if pd.notna(home_gdpm) and pd.notna(away_gdpm) else np.nan,
            "home_home_ppm_pre":        h_home_ppm,
            "away_away_ppm_pre":        a_away_ppm,
            # ── ELO (cross-season, opponent-adjusted strength) ──────────
            "home_elo_pre":             home_elo_pre,
            "away_elo_pre":             away_elo_pre,
            "diff_elo_pre":             home_elo_pre - away_elo_pre,
            # ── overall rolling form ───────────────────────────────────
            "home_form3_pts_pre":       rmean(t_pts3[home]),
            "away_form3_pts_pre":       rmean(t_pts3[away]),
            "home_form5_pts_pre":       rmean(t_pts5[home]),
            "away_form5_pts_pre":       rmean(t_pts5[away]),
            "home_form10_pts_pre":      rmean(t_pts10[home]),
            "away_form10_pts_pre":      rmean(t_pts10[away]),
            "home_form5_gf_pre":        rmean(t_gf5[home]),
            "away_form5_gf_pre":        rmean(t_gf5[away]),
            "home_form5_ga_pre":        rmean(t_ga5[home]),
            "away_form5_ga_pre":        rmean(t_ga5[away]),
            # ── shots on target (overall last 5) ───────────────────────
            "home_form5_sot_for_pre":   rmean(t_sot_for5[home]),
            "away_form5_sot_for_pre":   rmean(t_sot_for5[away]),
            "home_form5_sot_ag_pre":    rmean(t_sot_ag5[home]),
            "away_form5_sot_ag_pre":    rmean(t_sot_ag5[away]),
            "diff_sot_pre":             rmean(t_sot_for5[home]) - rmean(t_sot_for5[away])
                                        if t_sot_for5[home] and t_sot_for5[away] else np.nan,
            # ── VENUE-SPECIFIC form ────────────────────────────────────
            "home_venue_pts5_pre":      rmean(t_home_pts5[home]),
            "away_venue_pts5_pre":      rmean(t_away_pts5[away]),
            "home_venue_gf5_pre":       rmean(t_home_gf5[home]),
            "away_venue_gf5_pre":       rmean(t_away_gf5[away]),
            "home_venue_ga5_pre":       rmean(t_home_ga5[home]),
            "away_venue_ga5_pre":       rmean(t_away_ga5[away]),
            "home_venue_sot5_pre":      rmean(t_home_sot5[home]),
            "away_venue_sot5_pre":      rmean(t_away_sot5[away]),
            "home_venue_sotag5_pre":    rmean(t_home_sotag5[home]),
            "away_venue_sotag5_pre":    rmean(t_away_sotag5[away]),
            # ── rest / fatigue ─────────────────────────────────────────
            "home_rest_days_pre":       home_rest,
            "away_rest_days_pre":       away_rest,
            "diff_rest_days_pre":       home_rest - away_rest if pd.notna(home_rest) and pd.notna(away_rest) else np.nan,
            # ── head-to-head (cross-season) ────────────────────────────
            "h2h_pts_pre":              rmean(h2h_pts[h2h_key]),
            "h2h_gf_pre":               rmean(h2h_gf[h2h_key]),
            "h2h_ga_pre":               rmean(h2h_ga[h2h_key]),
            # ── bookmaker odds ─────────────────────────────────────────
            "avg_home_prob":            avg_h,
            "avg_draw_prob":            avg_d,
            "avg_away_prob":            avg_a,
            "b365_home_prob":           b365_h,
            "b365_draw_prob":           b365_d,
            "b365_away_prob":           b365_a,
            # ── targets ───────────────────────────────────────────────
            "target_ftr":               m["FTR"],
            "target_home_goals":        m["FTHG"],
            "target_away_goals":        m["FTAG"],
            "target_home_win":  1 if m["FTR"] == "H" else (0 if pd.notna(m["FTR"]) else np.nan),
            "target_draw":      1 if m["FTR"] == "D" else (0 if pd.notna(m["FTR"]) else np.nan),
            "target_away_win":  1 if m["FTR"] == "A" else (0 if pd.notna(m["FTR"]) else np.nan),
        }
        rows.append(row)

        # ── Update state (AFTER snapping features) ─────────────────────────
        ftr = m["FTR"]
        if pd.notna(ftr) and pd.notna(m["FTHG"]) and pd.notna(m["FTAG"]):
            hg  = int(m["FTHG"])
            ag  = int(m["FTAG"])
            hp, ap = (3, 0) if ftr == "H" else ((1, 1) if ftr == "D" else (0, 3))
            hst = int(m["HST"]) if pd.notna(m["HST"]) else None
            ast = int(m["AST"]) if pd.notna(m["AST"]) else None

            # table
            hs.matches += 1;        hs.points    += hp; hs.gf    += hg; hs.ga    += ag
            hs.home_matches += 1;   hs.home_points += hp; hs.home_gf += hg; hs.home_ga += ag
            as_.matches += 1;       as_.points   += ap; as_.gf   += ag; as_.ga   += hg
            as_.away_matches += 1;  as_.away_points += ap; as_.away_gf += ag; as_.away_ga += hg

            # overall rolling
            for q3, q5, q10, q_gf, q_ga, team, pts, gfor, gagainst in [
                (t_pts3[home], t_pts5[home], t_pts10[home], t_gf5[home], t_ga5[home], home, hp, hg, ag),
                (t_pts3[away], t_pts5[away], t_pts10[away], t_gf5[away], t_ga5[away], away, ap, ag, hg),
            ]:
                q3.append(pts); q5.append(pts); q10.append(pts)
                q_gf.append(gfor); q_ga.append(gagainst)

            # shots on target (overall)
            if hst is not None:
                t_sot_for5[home].append(hst)
                t_sot_ag5[away].append(hst)
            if ast is not None:
                t_sot_for5[away].append(ast)
                t_sot_ag5[home].append(ast)

            # venue-specific (home team playing at home)
            t_home_pts5[home].append(hp);  t_home_gf5[home].append(hg); t_home_ga5[home].append(ag)
            if hst is not None: t_home_sot5[home].append(hst)
            if ast is not None: t_home_sotag5[home].append(ast)

            # venue-specific (away team playing away)
            t_away_pts5[away].append(ap);  t_away_gf5[away].append(ag); t_away_ga5[away].append(hg)
            if ast is not None: t_away_sot5[away].append(ast)
            if hst is not None: t_away_sotag5[away].append(hst)

            # ELO (persists)
            new_h, new_a = elo_update(elo[home], elo[away], ftr)
            elo[home] = new_h;  elo[away] = new_a

            # H2H (persists)
            h2h_pts[(home, away)].append(hp); h2h_gf[(home, away)].append(hg); h2h_ga[(home, away)].append(ag)
            h2h_pts[(away, home)].append(ap); h2h_gf[(away, home)].append(ag); h2h_ga[(away, home)].append(hg)

            if pd.notna(dt):
                last_date[home] = dt;  last_date[away] = dt

    out = (
        pd.DataFrame(rows)
        .sort_values(["match_datetime", "home_team", "away_team"])
        .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    ds = build_dataset()
    played   = ds[ds["is_future_match"] == 0]
    future   = ds[ds["is_future_match"] == 1]
    current  = ds[ds["is_current_season"] == 1]
    h2h_null = played["h2h_pts_pre"].isna().mean()
    sot_null = played["home_form5_sot_for_pre"].isna().mean()
    elo_range = played["diff_elo_pre"].describe()[["min","mean","max"]]
    print(f"Total rows       : {len(ds)}")
    print(f"  Played (all)   : {len(played)}")
    print(f"  Current season : {len(current)}")
    print(f"  Future fixtures: {len(future)}")
    print(f"H2H null rate    : {h2h_null:.1%}")
    print(f"SoT null rate    : {sot_null:.1%}")
    print(f"ELO diff (min/mean/max): {elo_range['min']:.0f} / {elo_range['mean']:.0f} / {elo_range['max']:.0f}")
    print(f"Feature columns  : {len([c for c in ds.columns if c.endswith('_pre') or 'prob' in c or 'elo' in c])}")
    ds.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_FILE}")
