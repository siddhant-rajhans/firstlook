"""Auto-pick the right charts for a dataframe and return one Plotly figure."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .detect import detect_task

PALETTE = theme.PALETTE
CYAN, GOLD = theme.CYAN, theme.GOLD


def _numeric_feats(df, target):
    return [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]


def _corr_ratio(cats, values):
    """Correlation ratio (eta-squared): association between a categorical
    target and a numeric feature — between-group variance over total variance."""
    d = pd.DataFrame({"c": cats.values, "v": pd.to_numeric(values, errors="coerce").values}).dropna()
    if d.empty:
        return 0.0
    grand = d["v"].mean()
    ss_total = ((d["v"] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * (g["v"].mean() - grand) ** 2 for _, g in d.groupby("c"))
    return ss_between / ss_total


def _rank_features(df, target, num, task):
    """Order numeric features by association with the target.

    Classification uses the correlation ratio (eta-squared) — the right measure
    for a categorical target. Regression uses |Pearson r|. Ranking features by
    correlation with integer-encoded class labels would impose a fake ordering
    on the classes, so we don't.
    """
    if target is None or not num:
        return num
    y = df[target]
    if task == "classification":
        score = {c: _corr_ratio(y, df[c]) for c in num}
    else:
        def pear(c):
            try:
                return abs(np.corrcoef(df[c].astype(float), y.astype(float))[0, 1])
            except Exception:
                return 0.0
        score = {c: pear(c) for c in num}
    return sorted(num, key=lambda c: np.nan_to_num(score[c]), reverse=True)


def visualize(df, target=None, task=None):
    """Return a 2x2 Plotly dashboard chosen to fit the data and task."""
    task = task or detect_task(df, target)
    num = _numeric_feats(df, target)
    ranked = _rank_features(df, target, num, task)

    titles = [f"target: {target}" if target else "first feature",
              "the key relationship", "feature correlations", "a closer look"]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=titles,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.16, horizontal_spacing=0.12,
    )

    # P1 - target (or first feature) distribution
    if task == "classification" and target is not None:
        vc = df[target].value_counts()
        fig.add_trace(go.Bar(x=[str(i) for i in vc.index], y=vc.values,
                             marker_color=PALETTE[: len(vc)], showlegend=False), 1, 1)
    else:
        col = target if (target and pd.api.types.is_numeric_dtype(df[target])) else (ranked[0] if ranked else None)
        if col is not None:
            fig.add_trace(go.Histogram(x=df[col], marker_color=CYAN, showlegend=False), 1, 1)

    # P2 - the key relationship
    if task == "classification" and target is not None and len(ranked) >= 2:
        f1, f2 = ranked[0], ranked[1]
        for i, (cls, g) in enumerate(df.groupby(target)):
            fig.add_trace(go.Scatter(x=g[f1], y=g[f2], mode="markers", name=str(cls),
                                     marker=dict(size=8, color=PALETTE[i % len(PALETTE)],
                                                 line=dict(color="#04101a", width=0.5))), 1, 2)
        fig.update_xaxes(title_text=f1, row=1, col=2)
        fig.update_yaxes(title_text=f2, row=1, col=2)
    elif target is not None and ranked:
        f1 = ranked[0]
        fig.add_trace(go.Scatter(x=df[f1], y=df[target], mode="markers", showlegend=False,
                                 marker=dict(size=9, color=CYAN, line=dict(color="#04101a", width=0.5))), 1, 2)
        fig.update_xaxes(title_text=f1, row=1, col=2)
        fig.update_yaxes(title_text=str(target), row=1, col=2)
    elif len(ranked) >= 2:  # clustering / no target
        fig.add_trace(go.Scatter(x=df[ranked[0]], y=df[ranked[1]], mode="markers", showlegend=False,
                                 marker=dict(size=8, color=theme.PURPLE)), 1, 2)
        fig.update_xaxes(title_text=ranked[0], row=1, col=2)
        fig.update_yaxes(title_text=ranked[1], row=1, col=2)

    # P3 - correlation heatmap
    cols = num + ([target] if (target and pd.api.types.is_numeric_dtype(df[target])) else [])
    if len(cols) >= 2:
        corr = df[cols].corr()
        fig.add_trace(go.Heatmap(z=corr.values, x=cols, y=cols, zmin=-1, zmax=1,
                                 colorscale=[[0, theme.PINK], [0.5, "#1a1a2e"], [1, CYAN]],
                                 showscale=False), 2, 1)

    # P4 - a closer look
    if task == "classification" and target is not None and ranked:
        f1 = ranked[0]
        for i, (cls, g) in enumerate(df.groupby(target)):
            fig.add_trace(go.Box(y=g[f1], name=str(cls), marker_color=PALETTE[i % len(PALETTE)],
                                 showlegend=False), 2, 2)
        fig.update_yaxes(title_text=f1, row=2, col=2)
    elif ranked:
        f = ranked[1] if len(ranked) >= 2 else ranked[0]
        fig.add_trace(go.Histogram(x=df[f], marker_color=GOLD, showlegend=False), 2, 2)
        fig.update_xaxes(title_text=f, row=2, col=2)

    fig.update_layout(template="firstlook", height=620, margin=dict(l=55, r=30, t=50, b=45),
                     legend=dict(font=dict(size=11)))
    for a in fig.layout.annotations:
        a.font.color = theme.INK
        a.font.size = 13
    return fig
