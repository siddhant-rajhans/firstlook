"""See gradient descent: the 3-D loss surface, the path down it, and the
loss-per-step curve. Pure numpy — no scikit-learn — so it stays in the light core.

It fits a deliberately tiny two-parameter model — a bias ``theta0`` plus one
feature's weight ``theta1`` — so the loss genuinely *is* a surface you can look
at: linear regression (MSE) for a regression target, logistic regression
(log-loss) for a binary one. That honesty is the point — the surface is the real
loss of the real fit, which is why it's a clean convex bowl rather than the bumpy
landscape a deep network would draw. Same idea as the classic gradient-descent
picture, run on your data.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import theme
from .detect import detect_task
from .visualize import _numeric_feats, _rank_features

INK, CYAN, GOLD, PINK, GREEN = theme.INK, theme.CYAN, theme.GOLD, theme.PINK, theme.GREEN
PANEL, MUTED, GRID, PURPLE = theme.PANEL, theme.MUTED, theme.GRID, theme.PURPLE

_SURFACE_SCALE = [[0.0, "#0a0a18"], [0.35, PURPLE], [0.7, PINK], [1.0, GOLD]]


@dataclass
class Descent:
    """Result of :func:`descent` — the fit, its trajectory, and the figure."""
    task: Optional[str] = None
    feature: Optional[str] = None
    target: Optional[str] = None
    model: Optional[str] = None
    metric: Optional[str] = None
    lr: Optional[float] = None
    steps: Optional[int] = None
    final_loss: Optional[float] = None
    loss_history: List[float] = field(default_factory=list)
    path: List[Tuple[float, float]] = field(default_factory=list)
    figure: object = None
    note: Optional[str] = None

    def __str__(self):
        if self.figure is None:
            return f"descent: {self.note}"
        return (f"descent: {self.model} on '{self.feature}' -> {self.target}  "
                f"{self.metric} {self.loss_history[0]:.4f} -> {self.final_loss:.4f} "
                f"in {self.steps} steps (lr={self.lr})")

    def show(self):
        if self.figure is not None:
            self.figure.show()
        else:
            print(self)
        return self

    def to_html(self, path):
        if self.figure is None:
            raise ValueError(f"nothing to write: {self.note}")
        self.figure.write_html(path, include_plotlyjs="cdn")
        return path

    def _repr_html_(self):
        if self.figure is None:
            return f"<p style='color:{MUTED};font-family:Inter,system-ui,sans-serif'>descent — {self.note}</p>"
        cap = (f"<div style='font-family:Inter,system-ui,sans-serif;color:{MUTED};"
               f"font-size:13px;margin:4px 0 6px'>gradient descent · "
               f"<b style='color:{INK}'>{self.model}</b> on feature "
               f"<b style='color:{CYAN}'>{self.feature}</b> · {self.metric} "
               f"{self.loss_history[0]:.4f} &rarr; <b style='color:{GREEN}'>"
               f"{self.final_loss:.4f}</b> in {self.steps} steps</div>")
        return cap + self.figure.to_html(full_html=False, include_plotlyjs="cdn")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def descent(df, target=None, feature=None, *, lr=None, steps=60, show=True, title=None):
    """Visualise gradient descent on ``df``: loss surface + path + loss curve.

    Fits a 2-parameter model (bias + one feature) so the loss is a real surface.
    Regression -> linear / MSE; binary classification -> logistic / log-loss.
    The feature defaults to the one most associated with the target (override with
    ``feature=``). Returns a :class:`Descent`; pass ``show=False`` for just the object.
    """
    steps = max(1, int(steps))
    task = detect_task(df, target)
    d = Descent(task=task, target=target, steps=steps)

    if target is None or task == "clustering":
        d.note = "needs a target column (regression or binary classification)"
        return _finish(d, show)
    if task == "classification" and df[target].dropna().nunique() != 2:
        d.note = "the descent view supports regression and binary classification, not multiclass"
        return _finish(d, show)

    num = _numeric_feats(df, target)
    if not num:
        d.note = "needs at least one numeric feature to descend on"
        return _finish(d, show)
    feat = feature or _rank_features(df, target, num, task)[0]
    d.feature = feat

    data = df[[feat, target]].dropna()
    if len(data) < 5:
        d.note = "not enough rows after dropping missing values"
        return _finish(d, show)

    x = data[feat].to_numpy(dtype=float)
    if x.std() == 0:
        d.note = f"feature '{feat}' is constant — nothing to descend on"
        return _finish(d, show)
    x = (x - x.mean()) / x.std()  # standardise so one learning rate fits any column

    if task == "regression":
        y = data[target].to_numpy(dtype=float)
        if y.std() == 0:
            d.note = "target is constant — no surface to draw"
            return _finish(d, show)
        y = (y - y.mean()) / y.std()
        d.model, d.metric = "linear regression", "MSE"

        def loss_fn(w0, w1):
            return float(0.5 * np.mean((w0 + w1 * x - y) ** 2))

        def grads(w0, w1):
            err = (w0 + w1 * x) - y
            return float(err.mean()), float((err * x).mean())
    else:
        classes = sorted(data[target].dropna().unique().tolist(), key=str)
        y = (data[target].to_numpy() == classes[1]).astype(float)
        d.model, d.metric = "logistic regression", "log-loss"

        def loss_fn(w0, w1):
            p = np.clip(_sigmoid(w0 + w1 * x), 1e-9, 1 - 1e-9)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        def grads(w0, w1):
            err = _sigmoid(w0 + w1 * x) - y
            return float(err.mean()), float((err * x).mean())

    lr = 0.3 if lr is None else float(lr)
    d.lr = lr
    w0 = w1 = 0.0
    path, losses = [], []
    for _ in range(steps + 1):
        path.append((w0, w1))
        losses.append(loss_fn(w0, w1))
        g0, g1 = grads(w0, w1)
        w0 -= lr * g0
        w1 -= lr * g1
    d.path, d.loss_history, d.final_loss = path, losses, losses[-1]
    d.figure = _figure(x, y, task, path, losses, feat, d.metric, title)
    return _finish(d, show)


def _finish(d, show):
    if show and d.figure is not None:
        d.figure.show()
    return d


def _figure(x, y, task, path, losses, feat, metric, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    w0s = np.array([p[0] for p in path])
    w1s = np.array([p[1] for p in path])

    def span(vals):
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.5, 0.45 * (hi - lo))
        return lo - pad, hi + pad

    lo0, hi0 = span(w0s)
    lo1, hi1 = span(w1s)
    g0 = np.linspace(lo0, hi0, 48)
    g1 = np.linspace(lo1, hi1, 48)

    xs, ys = x, y
    if xs.shape[0] > 1500:  # keep the grid cheap on big frames
        idx = np.random.default_rng(0).choice(xs.shape[0], 1500, replace=False)
        xs, ys = xs[idx], ys[idx]

    W0 = g0[None, :, None]
    W1 = g1[:, None, None]
    X = xs[None, None, :]
    if task == "regression":
        Z = 0.5 * np.mean((W0 + W1 * X - ys[None, None, :]) ** 2, axis=2)
    else:
        P = np.clip(_sigmoid(W0 + W1 * X), 1e-9, 1 - 1e-9)
        Yb = ys[None, None, :]
        Z = -np.mean(Yb * np.log(P) + (1 - Yb) * np.log(1 - P), axis=2)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=("loss surface and the path down", "loss per step"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(go.Surface(
        x=g0, y=g1, z=Z, colorscale=_SURFACE_SCALE, showscale=False, opacity=0.9,
        contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
        name="loss"), 1, 1)
    fig.add_trace(go.Scatter3d(
        x=w0s, y=w1s, z=losses, mode="lines+markers",
        line=dict(color=INK, width=5), marker=dict(size=3, color=INK),
        name="path"), 1, 1)
    fig.add_trace(go.Scatter3d(
        x=[w0s[0]], y=[w1s[0]], z=[losses[0]], mode="markers",
        marker=dict(size=6, color=PINK), name="start"), 1, 1)
    fig.add_trace(go.Scatter3d(
        x=[w0s[-1]], y=[w1s[-1]], z=[losses[-1]], mode="markers",
        marker=dict(size=6, color=GREEN), name="minimum"), 1, 1)
    fig.add_trace(go.Scatter(
        x=list(range(len(losses))), y=losses, mode="lines+markers",
        line=dict(color=CYAN, width=3), marker=dict(size=4, color=CYAN),
        showlegend=False), 1, 2)

    fig.update_layout(
        template="firstlook", height=480, showlegend=False,
        title=title or f"gradient descent — {metric} on '{feat}'",
        margin=dict(l=0, r=12, t=58, b=0),
        scene=dict(
            xaxis_title="θ₀ (bias)", yaxis_title="θ₁ (weight)", zaxis_title="loss J(θ)",
            bgcolor=PANEL,
            xaxis=dict(color=MUTED, gridcolor=GRID),
            yaxis=dict(color=MUTED, gridcolor=GRID),
            zaxis=dict(color=MUTED, gridcolor=GRID),
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.1)),
        ),
    )
    fig.update_xaxes(title_text="step", row=1, col=2)
    fig.update_yaxes(title_text=f"loss ({metric})", row=1, col=2)
    return fig
