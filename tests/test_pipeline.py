import numpy as np
import pandas as pd
import pytest

from laliga import evaluate as E, model as M
from laliga.data import load_matches
from laliga.features import build_features, elo_update


@pytest.fixture(scope="module")
def matches():
    return load_matches()


@pytest.fixture(scope="module")
def features(matches):
    return build_features(matches)


def test_walk_forward_trains_only_on_the_past(features):
    """The bug behind the old 58% figure: test rows were also training rows."""
    for season, train, test in E.season_folds(features):
        assert set(train.index).isdisjoint(test.index)
        assert train["match_datetime"].max() < test["match_datetime"].min()
        assert (test["season"] == season).all()


def test_features_do_not_see_the_result(matches):
    """Changing a match's result must not change that match's own features,
    only the features of later matches."""
    played = matches[(matches["season"] == "2024-25") & (matches["is_future_match"] == 0)]
    played = played.reset_index(drop=True)
    i = 100
    flipped = played.copy()
    flipped.loc[i, ["FTHG", "FTAG", "FTR"]] = (9, 0, "H")
    a, b = build_features(played), build_features(flipped)
    feats = [c for c in M.FEATURES if c in a]
    pd.testing.assert_frame_equal(a.loc[:i, feats], b.loc[:i, feats])
    assert not a.loc[i + 1:, feats].equals(b.loc[i + 1:, feats])


def test_elo_is_zero_sum():
    for ftr in "HDA":
        h, a = elo_update(1600, 1450, ftr)
        assert h + a == pytest.approx(3050)


def test_rps_known_values():
    y = np.array([0, 2])
    perfect = np.array([[1, 0, 0], [0, 0, 1.0]])
    assert E.rps(y, perfect).mean() == 0
    # predicting H with certainty when A happened is the worst case: RPS 1
    assert E.rps(np.array([2]), np.array([[1, 0, 0.0]]))[0] == 1
    assert E.rps(np.array([1]), np.full((1, 3), 1 / 3))[0] == pytest.approx(1 / 9)


def test_score_grid_is_a_distribution():
    g = M.score_grid(1.6, 1.1)
    assert g.sum() == pytest.approx(1)
    assert np.unravel_index(g.argmax(), g.shape) in {(1, 0), (1, 1)}


def test_bookmaker_probs_have_no_margin(features):
    p = M.bookmakers(features[features["is_future_match"] == 0])
    assert not np.isnan(p).any()
    assert np.allclose(p.sum(axis=1), 1)
