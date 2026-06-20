import numpy as np
import pandas as pd

from firstlook import fit_baseline, play


def test_regression_baseline_scores():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(scale=0.1, size=200)})
    b = fit_baseline(df, "y", "regression")
    assert b.model == "LinearRegression"
    assert b.metric == "R2"
    assert b.score > 0.9  # near-perfect linear relationship


def test_classification_baseline_scores():
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"f": rng.normal(0, 1, 100), "y": 0})
    b = pd.DataFrame({"f": rng.normal(5, 1, 100), "y": 1})
    df = pd.concat([a, b], ignore_index=True)
    res = fit_baseline(df, "y", "classification")
    assert res.metric == "accuracy"
    assert res.score > 0.9  # well-separated classes


def test_handles_categoricals_and_missing():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "num": rng.normal(size=120),
        "cat": rng.choice(["a", "b", "c"], size=120),
        "y": rng.integers(0, 2, size=120),
    })
    df.loc[3, "num"] = np.nan  # the pipeline should impute, not crash
    res = fit_baseline(df, "y", "classification")
    assert res.score is not None


def test_tiny_data_is_skipped():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [2.0, 4, 6, 8]})
    res = fit_baseline(df, "y", "regression")
    assert res.score is None and "few rows" in res.note


def test_play_with_fit_attaches_baseline():
    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    df = pd.DataFrame({"x": x, "y": 2 * x + 1 + rng.normal(scale=0.1, size=150)})
    report = play(df, target="y", fit=True, show=False)
    assert report.baseline is not None
    assert report.baseline.score > 0.9
