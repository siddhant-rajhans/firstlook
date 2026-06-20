import numpy as np
import pandas as pd

from firstlook import leaderboard, play, fit_baseline


def _reg_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "x2": rng.normal(size=n),
                         "y": 3 * x + rng.normal(scale=0.1, size=n)})


def test_ranks_sorted_and_contiguous():
    lb = leaderboard(_reg_df(), "y")
    assert len(lb.entries) == 4
    scores = [e.score for e in lb.entries]
    assert scores == sorted(scores, reverse=True)
    assert [e.rank for e in lb.entries] == [1, 2, 3, 4]


def test_regression_best_is_strong():
    lb = leaderboard(_reg_df(), "y")
    assert lb.task == "regression" and lb.metric == "R2"
    assert lb.best.score > 0.9


def test_imbalanced_uses_balanced_accuracy():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"f": rng.normal(size=n), "y": rng.choice([0, 1], size=n, p=[0.9, 0.1])})
    lb = leaderboard(df, "y")
    assert lb.metric == "balanced accuracy"


def test_handles_categoricals_and_missing():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "num": rng.normal(size=150),
        "cat": rng.choice(["a", "b", "c"], size=150),
        "y": rng.integers(0, 2, size=150),
    })
    df.loc[3, "num"] = np.nan
    lb = leaderboard(df, "y")
    assert lb.entries
    assert all(np.isfinite(e.score) for e in lb.entries)


def test_no_target_is_noted():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    lb = leaderboard(df, None)
    assert lb.entries == [] and lb.note


def test_play_fit_all_attaches_leaderboard_and_best_baseline():
    rep = play(_reg_df(), target="y", fit="all", show=False)
    assert rep.leaderboard is not None and rep.leaderboard.entries
    assert rep.baseline is not None
    assert rep.baseline.model == rep.leaderboard.best.model
    assert abs(rep.baseline.score - rep.leaderboard.best.score) < 1e-9


def test_play_fit_true_is_single_baseline():
    rep = play(_reg_df(), target="y", fit=True, show=False)
    assert rep.baseline is not None
    assert rep.leaderboard is None


def test_linear_parity_baseline_vs_leaderboard():
    # the single LinearRegression baseline must equal its leaderboard entry
    df = _reg_df()
    b = fit_baseline(df, "y", "regression")
    lin = next(e for e in leaderboard(df, "y").entries if e.model == "LinearRegression")
    assert abs(b.score - lin.score) < 1e-9
