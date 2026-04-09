# La Liga 2025/26 Match Predictor

> **Walkthrough notebook:** [`laliga_predictor_walkthrough.ipynb`](laliga_predictor_walkthrough.ipynb)  
> **Diagnostic plots:** [`plots/`](plots/)

A machine learning pipeline that predicts La Liga match results and scorelines using 11 seasons of historical data (2015–26).

---

## How it works

### Architecture

```
data/                          ← raw match data
  SP1 2015.csv … SP1 2025 updated.csv   ← historical results (11 seasons)
  fixtures_25_26.csv           ← remaining fixtures for current season

data_engineering/
  build_model_dataset.py       ← Step 1: build feature dataset

data/
  model_ready_matches_25_26.csv  ← output of Step 1

prediction/
  train_predict.py             ← Step 2: train models + generate predictions

data/
  predictions_future_matches.csv  ← final predictions (one row per fixture)
  evaluation_report.txt           ← model performance report
```

### Step 1 — Feature Engineering (`build_model_dataset.py`)

Processes all 11 seasons chronologically and builds a row for every match with **pre-match only** features (no data leakage). Features computed per match:

| Feature group | Description |
|---|---|
| **ELO ratings** | Opponent-adjusted team strength, updated after every match, persists across seasons |
| **Season table stats** | Points per match, goal difference per match, home/away splits |
| **Rolling form** | Last 3 / 5 / 10 matches — points, goals scored, goals conceded |
| **Shots on target** | Rolling 5-match average shots on target for/against (luck-adjusted quality) |
| **Venue-specific form** | Home team's last 5 home games; away team's last 5 away games |
| **Head-to-head** | Last 5 H2H meetings between this exact pair (persists across seasons) |
| **Rest days** | Days since each team's previous match |
| **Bookmaker odds** | Consensus (Bet365 + market average) implied probabilities |
| **Season progress** | Fractional progress within the season (0 → 1) |

### Step 2 — Training & Prediction (`train_predict.py`)

Trains three models on all played matches, then predicts remaining fixtures:

| Model | Purpose | Algorithm |
|---|---|---|
| **Result classifier** | Predict H / D / A with probabilities | XGBoost multi-class softmax |
| **Home goals** | Expected goals for home team | XGBoost Poisson regressor |
| **Away goals** | Expected goals for away team | XGBoost Poisson regressor |

Key design choices:
- **Time-based split** — test set is always the most recent matches (no random shuffle)
- **Recency weighting** — recent matches weighted more (half-life = 1 full season)
- **OOF isotonic calibration** — probabilities corrected using out-of-fold cross-validation to avoid overconfidence
- **Walk-forward CV** — evaluates across 7 historical seasons for a reliable accuracy estimate

**Score distributions** are generated using the Poisson model: given expected goals λ_home and λ_away, the probability of every scoreline (0–0 through 5–5) is P(i–j) = Poisson(i, λ_h) × Poisson(j, λ_a).

### Model performance

| Metric | Value |
|---|---|
| Walk-forward CV accuracy (7 seasons, 798 matches) | **58.1%** |
| Current-season holdout accuracy (51 matches) | **52.9%** vs 51.0% odds baseline |
| RPS (lower = better) | **0.198** vs 0.201 odds baseline |

> The theoretical ceiling for football prediction is ~55–57%. Draws (≈26% of matches) have near-zero signal and are the main source of unpredictability.

---

## Output files

### `predictions_future_matches.csv`

One row per remaining fixture. Key columns:

| Column | Description |
|---|---|
| `match_datetime` | Kick-off date and time |
| `home_team` / `away_team` | Teams |
| `predicted_result` | H (home win) / D (draw) / A (away win) |
| `prob_home_win` / `prob_draw` / `prob_away_win` | Model probabilities (sum to 1) |
| `most_likely_score` | Single most probable scoreline |
| `most_likely_score_prob` | That scoreline's probability (typically 10–15%) |
| `score_probabilities` | Top 5 scorelines with individual probabilities |
| `confidence` | Max probability across H/D/A — how certain the model is |
| `confidence_tier` | Low / Medium / High / Very High |
| `poisson_prob_h/d/a` | H/D/A probabilities from the Poisson goal model (cross-check) |
| `odds_implied_result` | What the bookmaker odds predict |
| `model_vs_odds` | Whether model agrees or disagrees with the market |

### `evaluation_report.txt`

Full performance report including walk-forward CV results, confusion matrix, goals model MAE/RMSE, and top feature importances.

---

## How to update for each gameweek

After a gameweek is played:

### 1. Add new results to `data/SP1 2025 updated.csv`

Add rows in the same format as the existing file:

```
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,...
12/04/2026,Ath Bilbao,Villarreal,2,1,H,14,8,6,3,...
```

Required columns: `Date`, `HomeTeam`, `AwayTeam`, `FTHG` (home goals), `FTAG` (away goals), `FTR` (H/D/A), `HS`, `AS`, `HST`, `AST` (shots).

Odds columns (`AvgH`, `AvgD`, `AvgA`, `B365H`, `B365D`, `B365A`) are optional but improve accuracy.

### 2. Remove played fixtures from `data/fixtures_25_26.csv`

Delete the rows for matches that have now been played. The file should only contain genuinely future fixtures.

### 3. Run the pipeline

```bash
python data_engineering/build_model_dataset.py
python prediction/train_predict.py
```

Fresh predictions will be written to `data/predictions_future_matches.csv`.

---

## Why the model agrees with bookmaker odds on most fixtures

The model predicts the same result as the bookmakers on the majority of fixtures. This is expected, not a bug.

All the features we engineer — ELO, form, shots on target, head-to-head — are a subset of what professional oddsmakers already price in. The `consensus_home_prob` and `consensus_away_prob` features let the model inherit all of that market knowledge. Since our features don't add information the market doesn't have, the model rarely overrides the odds.

**What would generate genuine disagreements:**

| Signal | How to use it |
|---|---|
| **Live odds movement** | If a line moves from 2.0 → 2.4 in the 48 hours before kickoff, sharp money spotted something (injury, rotation). Scraping live odds 24 hours before each match would surface these discrepancies. |
| **Injury & suspension data** | A team missing its top striker materially changes the goals model. APIs like SofaScore or Transfermarkt provide this for free. Add it as a binary `home_striker_out` feature. |
| **Cup/European rotation** | Teams in Champions League often rest starters before big European games. This is publicly known but not captured in our dataset. |

**How to use current predictions confidently:**  
Filter `predictions_future_matches.csv` for rows where `confidence_tier = "High"` or `"Very High"`. These are the fixtures where the model has the clearest signal regardless of market alignment.

---

## Visualisations

Run `python visualise.py` to generate all diagnostic plots into `plots/`:

| Plot | Description |
|---|---|
| `feature_importance.png` | Top 20 features by XGBoost importance, coloured by group |
| `calibration_curves.png` | Are predicted probabilities reliable? Model vs perfect calibration |
| `confusion_matrix.png` | How often each actual result was predicted correctly |
| `elo_trajectory.png` | ELO rating over the season for the top 6 teams |
| `score_heatmap_example.png` | Full scoreline probability matrix for a sample fixture |

---

## Installation

```bash
pip install xgboost scikit-learn pandas numpy
```

Python 3.10+ required.

---

## Data sources

- Historical results (2015–2025): [football-data.co.uk](https://www.football-data.co.uk/spainm.php) — free CSV downloads
- Current season results: same source, updated weekly
- Fixtures: exported from a fixtures calendar and kept in `fixtures_25_26.csv`
