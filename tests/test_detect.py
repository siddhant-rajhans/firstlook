import numpy as np
import pandas as pd
import pytest

from firstlook import detect_task


def test_no_target_is_clustering():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert detect_task(df, None) == "clustering"


def test_string_target_is_classification():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})
    assert detect_task(df, "y") == "classification"


def test_few_int_classes_is_classification():
    df = pd.DataFrame({"x": np.arange(150), "y": ([0, 1, 2] * 50)})
    assert detect_task(df, "y") == "classification"


def test_continuous_target_is_regression():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    assert detect_task(df, "y") == "regression"


def test_small_regression_not_mistaken_for_classification():
    # 7 distinct values in 7 rows — the trap. Ratio check must keep it regression.
    df = pd.DataFrame({"weight": [2.4, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4],
                       "mpg": [24, 22, 20, 19, 17, 15, 14]})
    assert detect_task(df, "mpg") == "regression"


def test_bool_target_is_classification():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [True, False, True, False]})
    assert detect_task(df, "y") == "classification"


def test_missing_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        detect_task(df, "nope")
