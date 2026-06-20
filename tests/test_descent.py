import numpy as np
import pandas as pd

import firstlook
from firstlook.descent import Descent


def _reg_df(n=150):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 2.5 * x + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})


def _clf_df(n=150):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})


def test_regression_descent_converges():
    d = firstlook.descent(_reg_df(), target="y", show=False)
    assert isinstance(d, Descent)
    assert d.task == "regression"
    assert d.model == "linear regression" and d.metric == "MSE"
    assert d.figure is not None
    assert len(d.loss_history) == d.steps + 1
    assert d.final_loss < d.loss_history[0]   # loss actually decreased
    assert d.final_loss < 0.2                 # strong feature -> low MSE


def test_classification_descent_binary():
    d = firstlook.descent(_clf_df(), target="y", show=False)
    assert d.task == "classification"
    assert d.model == "logistic regression" and d.metric == "log-loss"
    assert d.final_loss < d.loss_history[0]
    assert d.figure is not None


def test_picks_most_associated_feature():
    d = firstlook.descent(_reg_df(), target="y", show=False)
    assert d.feature == "x"            # the signal, not 'noise'


def test_feature_override():
    d = firstlook.descent(_reg_df(), target="y", feature="noise", show=False)
    assert d.feature == "noise"


def test_figure_has_surface_path_and_curve():
    d = firstlook.descent(_reg_df(), target="y", show=False)
    kinds = [type(t).__name__ for t in d.figure.data]
    assert "Surface" in kinds                 # the loss surface
    assert kinds.count("Scatter3d") >= 1      # the descent path (+ start/min markers)
    assert "Scatter" in kinds                 # the loss-per-step curve


def test_no_target_is_graceful():
    d = firstlook.descent(_reg_df(), target=None, show=False)
    assert d.figure is None and d.note


def test_multiclass_is_graceful():
    df = pd.DataFrame({"x": range(30), "y": ["a", "b", "c"] * 10})
    d = firstlook.descent(df, target="y", show=False)
    assert d.figure is None and "multiclass" in d.note


def test_constant_feature_is_graceful():
    df = pd.DataFrame({"x": [1.0] * 30, "y": np.arange(30.0)})
    d = firstlook.descent(df, target="y", show=False)
    assert d.figure is None and d.note


def test_play_descent_attaches_and_renders():
    rep = firstlook.play(_reg_df(), target="y", show=False, descent=True)
    assert rep.descent is not None and rep.descent.figure is not None
    assert "gradient descent" in rep._repr_html_()
