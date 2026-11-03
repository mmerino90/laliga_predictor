"""
Pre-match feature engineering.

Matches are processed in kick-off order. Every feature for a match is read
from the running state *before* that match's result is applied, so no row can
see its own result (or any later one). tests/test_pipeline.py checks this.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
HOME_ADV_ELO = 60.0   # home advantage in ELO points


@dataclass
class TeamState:
    """Season-scoped league-table stats (reset each season)."""
    matches: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0
    home_matches: int = 0
    home_points: int = 0
    away_matches: int = 0
    away_points: int = 0


def implied_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Decimal odds -> probabilities with the bookmaker margin removed."""
    if any(pd.isna(v) or v <= 0 for v in (h, d, a)):
        return np.nan, np.nan, np.nan
    inv = np.array([1 / h, 1 / d, 1 / a])
    return tuple(inv / inv.sum())


def elo_expected(home_elo: float, away_elo: float) -> float:
    """Expected home score (win=1, draw=0.5) implied by the ELO gap + home advantage."""
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - HOME_ADV_ELO) / 400.0))


def elo_update(home_elo: float, away_elo: float, ftr: str) -> tuple[float, float]:
    """Return updated (home_elo, away_elo) after result ftr in {H, D, A}. Zero-sum."""
    actual = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
    delta = ELO_K * (actual - elo_expected(home_elo, away_elo))
    return home_elo + delta, away_elo - delta


def _mean(q: deque) -> float:
    return float(np.mean(q)) if q else np.nan


def _rolling(n: int) -> defaultdict:
    return defaultdict(lambda: deque(maxlen=n))


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match with pre-match features and the targets."""
    matches = matches.sort_values(["match_datetime", "home_team"]).reset_index(drop=True)

    # Persist across seasons
    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    h2h = {k: _rolling(5) for k in ("pts", "gf", "ga")}
    cur_season = None
    rows: list[dict[str, Any]] = []

    for m in matches.itertuples(index=False):
        home, away, dt = m.home_team, m.away_team, m.match_datetime

        if m.season != cur_season:  # reset season-scoped state
            cur_season = m.season
            table: dict[str, TeamState] = defaultdict(TeamState)
            # roll[name][team] -> deque of last-n values
            roll = {f"pts{n}": _rolling(n) for n in (3, 5, 10)}
            roll |= {k: _rolling(5) for k in ("gf", "ga", "sot_for", "sot_ag")}
            venue = {k: _rolling(5) for k in (
                "home_pts", "home_gf", "home_ga", "home_sot", "home_sotag",
                "away_pts", "away_gf", "away_ga", "away_sot", "away_sotag")}
            last_date: dict[str, pd.Timestamp] = {}

        hs, as_ = table[home], table[away]

        def per_match(num, den):
            return num / den if den else np.nan

        avg_h, avg_d, avg_a = implied_probs(m.AvgH, m.AvgD, m.AvgA)
        b365_h, b365_d, b365_a = implied_probs(m.B365H, m.B365D, m.B365A)
        home_rest = (dt - last_date[home]).days if home in last_date else np.nan
        away_rest = (dt - last_date[away]).days if away in last_date else np.nan

        row = {
            "match_datetime": dt, "home_team": home, "away_team": away,
            "season": m.season, "is_future_match": m.is_future_match,
            # table strength
            "home_matches_played_pre": hs.matches,
            "away_matches_played_pre": as_.matches,
            "home_ppm_pre": per_match(hs.points, hs.matches),
            "away_ppm_pre": per_match(as_.points, as_.matches),
            "home_gdpm_pre": per_match(hs.gf - hs.ga, hs.matches),
            "away_gdpm_pre": per_match(as_.gf - as_.ga, as_.matches),
            "home_home_ppm_pre": per_match(hs.home_points, hs.home_matches),
            "away_away_ppm_pre": per_match(as_.away_points, as_.away_matches),
            # ELO (cross-season, opponent-adjusted)
            "home_elo_pre": elo[home],
            "away_elo_pre": elo[away],
            # rest
            "home_rest_days_pre": home_rest,
            "away_rest_days_pre": away_rest,
            # head-to-head (cross-season)
            "h2h_pts_pre": _mean(h2h["pts"][(home, away)]),
            "h2h_gf_pre": _mean(h2h["gf"][(home, away)]),
            "h2h_ga_pre": _mean(h2h["ga"][(home, away)]),
            # bookmaker odds (margin removed)
            "avg_home_prob": avg_h, "avg_draw_prob": avg_d, "avg_away_prob": avg_a,
            "b365_home_prob": b365_h, "b365_draw_prob": b365_d, "b365_away_prob": b365_a,
            # targets
            "target_ftr": m.FTR,
            "target_home_goals": m.FTHG,
            "target_away_goals": m.FTAG,
        }
        for side, team in (("home", home), ("away", away)):
            for n in (3, 5, 10):
                row[f"{side}_form{n}_pts_pre"] = _mean(roll[f"pts{n}"][team])
            row[f"{side}_form5_gf_pre"] = _mean(roll["gf"][team])
            row[f"{side}_form5_ga_pre"] = _mean(roll["ga"][team])
            row[f"{side}_form5_sot_for_pre"] = _mean(roll["sot_for"][team])
            row[f"{side}_form5_sot_ag_pre"] = _mean(roll["sot_ag"][team])
            # venue form: home team's home games, away team's away games
            for stat in ("pts", "gf", "ga"):
                row[f"{side}_venue_{stat}5_pre"] = _mean(venue[f"{side}_{stat}"][team])
            row[f"{side}_venue_sot5_pre"] = _mean(venue[f"{side}_sot"][team])
            row[f"{side}_venue_sotag5_pre"] = _mean(venue[f"{side}_sotag"][team])
        rows.append(row)

        # ── apply the result AFTER the row has been recorded ──────────────────
        if pd.isna(m.FTR) or pd.isna(m.FTHG) or pd.isna(m.FTAG):
            continue
        hg, ag = int(m.FTHG), int(m.FTAG)
        hp, ap = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[m.FTR]
        hst = None if pd.isna(m.HST) else int(m.HST)
        ast = None if pd.isna(m.AST) else int(m.AST)

        hs.matches += 1; hs.points += hp; hs.gf += hg; hs.ga += ag
        hs.home_matches += 1; hs.home_points += hp
        as_.matches += 1; as_.points += ap; as_.gf += ag; as_.ga += hg
        as_.away_matches += 1; as_.away_points += ap

        for team, pts, gf, ga, sot_for, sot_ag in (
            (home, hp, hg, ag, hst, ast), (away, ap, ag, hg, ast, hst)
        ):
            for n in (3, 5, 10):
                roll[f"pts{n}"][team].append(pts)
            roll["gf"][team].append(gf)
            roll["ga"][team].append(ga)
            if sot_for is not None:
                roll["sot_for"][team].append(sot_for)
            if sot_ag is not None:
                roll["sot_ag"][team].append(sot_ag)

        for side, team, pts, gf, ga, sot, sotag in (
            ("home", home, hp, hg, ag, hst, ast), ("away", away, ap, ag, hg, ast, hst)
        ):
            venue[f"{side}_pts"][team].append(pts)
            venue[f"{side}_gf"][team].append(gf)
            venue[f"{side}_ga"][team].append(ga)
            if sot is not None:
                venue[f"{side}_sot"][team].append(sot)
            if sotag is not None:
                venue[f"{side}_sotag"][team].append(sotag)

        elo[home], elo[away] = elo_update(elo[home], elo[away], m.FTR)
        h2h["pts"][(home, away)].append(hp); h2h["pts"][(away, home)].append(ap)
        h2h["gf"][(home, away)].append(hg); h2h["gf"][(away, home)].append(ag)
        h2h["ga"][(home, away)].append(ag); h2h["ga"][(away, home)].append(hg)
        if pd.notna(dt):
            last_date[home] = last_date[away] = dt

    return add_derived(pd.DataFrame(rows))


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away gaps, consensus odds and season progress."""
    d = d.copy()
    for a, b, out in [
        ("home_ppm_pre", "away_ppm_pre", "diff_ppm_pre"),
        ("home_gdpm_pre", "away_gdpm_pre", "diff_gdpm_pre"),
        ("home_elo_pre", "away_elo_pre", "diff_elo_pre"),
        ("home_form3_pts_pre", "away_form3_pts_pre", "diff_form3_pts"),
        ("home_form5_pts_pre", "away_form5_pts_pre", "diff_form5_pts"),
        ("home_form10_pts_pre", "away_form10_pts_pre", "diff_form10_pts"),
        ("home_form5_gf_pre", "away_form5_gf_pre", "diff_form5_gf"),
        ("home_form5_ga_pre", "away_form5_ga_pre", "diff_form5_ga"),
        ("home_form5_sot_for_pre", "away_form5_sot_for_pre", "diff_sot_pre"),
        ("home_form5_sot_ag_pre", "away_form5_sot_ag_pre", "diff_sot_ag"),
        ("home_venue_pts5_pre", "away_venue_pts5_pre", "diff_venue_pts5"),
        ("home_venue_gf5_pre", "away_venue_gf5_pre", "diff_venue_gf5"),
        ("home_venue_ga5_pre", "away_venue_ga5_pre", "diff_venue_ga5"),
        ("home_rest_days_pre", "away_rest_days_pre", "diff_rest_days_pre"),
    ]:
        d[out] = d[a] - d[b]
    # Average the two bookmakers (either one alone if the other is missing)
    for o in ("home", "draw", "away"):
        d[f"consensus_{o}_prob"] = d[[f"avg_{o}_prob", f"b365_{o}_prob"]].mean(axis=1)
    # Matchday / 38, from games already played: known before kick-off, same scale every season
    d["season_progress"] = (d["home_matches_played_pre"] + d["away_matches_played_pre"]) / 76
    return d
