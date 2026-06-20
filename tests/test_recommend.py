import numpy as np
import pandas as pd

from firstlook import recommend, visualize


def test_regression_starts_linear():
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0 + 1})
    rec = recommend(df, "y")
    assert rec.task == "regression"
    assert rec.start == "LinearRegression"
    assert rec.models


def test_classification_flags_imbalance_and_categorical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "amt": rng.exponential(size=300),
        "region": rng.choice(["NA", "EU"], size=300),
        "fraud": rng.choice([0, 1], size=300, p=[0.92, 0.08]),
    })
    rec = recommend(df, "fraud")
    assert rec.task == "classification"
    joined = " ".join(rec.notes).lower()
    assert "imbalanced" in joined
    assert "categorical" in joined


def test_missing_values_flagged():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0] * 20, "y": np.arange(80) * 1.0})
    rec = recommend(df, "y")
    assert any("missing" in n.lower() for n in rec.notes)


def test_visualize_returns_figure():
    df = pd.DataFrame({"a": np.arange(60) * 1.0, "b": np.arange(60) * 1.5, "y": ([0, 1] * 30)})
    fig = visualize(df, "y")
    assert fig.data  # at least one trace
