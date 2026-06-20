"""Regression tests for the data-science review fixes."""
import numpy as np
import pandas as pd

from firstlook import detect_task, fit_baseline
from firstlook.visualize import _rank_features


def test_binary_float_target_is_classification():
    # nunique == 2 is binary classification even as floats in a small frame
    df = pd.DataFrame({"x": np.arange(30), "y": [0.0, 1.0] * 15})
    assert detect_task(df, "y") == "classification"


def test_imbalanced_classification_uses_balanced_accuracy():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"f": rng.normal(size=n), "y": rng.choice([0, 1], size=n, p=[0.9, 0.1])})
    res = fit_baseline(df, "y", "classification")
    assert res.metric == "balanced accuracy"  # not plain accuracy on imbalanced data


def test_balanced_classification_uses_plain_accuracy():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"f": rng.normal(size=200), "y": rng.choice([0, 1], size=200)})
    res = fit_baseline(df, "y", "classification")
    assert res.metric == "accuracy"


def test_classification_ranks_separating_feature_first():
    # 'sep' separates the classes; 'noise' doesn't. Correlation-ratio ranking
    # must surface 'sep' (Pearson-on-label-codes would not, reliably).
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "noise": rng.normal(0, 1, 200),
        "sep": np.concatenate([rng.normal(0, 1, 100), rng.normal(6, 1, 100)]),
        "y": np.array([0] * 100 + [1] * 100),
    })
    ranked = _rank_features(df, "y", ["noise", "sep"], "classification")
    assert ranked[0] == "sep"
