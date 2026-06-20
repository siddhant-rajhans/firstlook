"""Fit and rank a small panel of models — the advisor becomes a doer.

Every model runs through the *same* preprocessing pipeline and is scored with the
*same* honest metric as the single baseline (see ``_sklearn``), so the ranking is
apples-to-apples and imbalance-aware.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from . import _sklearn, theme


@dataclass
class Entry:
    model: str
    score: float
    std: float
    rank: int = 0


@dataclass
class Leaderboard:
    task: str
    metric: Optional[str] = None
    cv: Optional[int] = None
    entries: List[Entry] = field(default_factory=list)
    note: Optional[str] = None

    @property
    def best(self):
        return self.entries[0] if self.entries else None

    def __str__(self):
        if not self.entries:
            return f"leaderboard: {self.note}"
        lines = [f"leaderboard ({self.metric}, {self.cv}-fold CV):"]
        for e in self.entries:
            lines.append(f"  {e.rank}. {e.model:26} {e.score:.3f} +/- {e.std:.3f}")
        return "\n".join(lines)


def _panel(task, y):
    """Return ``[(name, estimator), ...]`` for the task (mirrors recommend.py)."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    )
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    if task == "regression":
        return [
            ("LinearRegression", LinearRegression()),
            ("RandomForest", RandomForestRegressor(random_state=0)),
            ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=0)),
            ("KNeighbors", KNeighborsRegressor()),
        ]
    cw = "balanced" if _sklearn.needs_class_weight(y) else None
    return [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight=cw)),
        ("RandomForest", RandomForestClassifier(random_state=0, class_weight=cw)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=0)),
        ("KNeighbors", KNeighborsClassifier()),
    ]


def leaderboard(df, target, task=None, models=None):
    """Fit & cross-validate a model panel; return a ranked :class:`Leaderboard`."""
    from .detect import detect_task
    task = task or detect_task(df, target)
    if task == "clustering" or target is None:
        return Leaderboard(task=task, note="no target, nothing to fit")
    _sklearn._require_sklearn()

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    data = df.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])
    guard = _sklearn.fit_guard(X, y, task)
    if guard:
        return Leaderboard(task=task, note=guard)

    pre, _, _ = _sklearn.build_preprocessor(X)
    metric, scoring = _sklearn.scoring_for(task, y)
    cv = _sklearn.cv_for(task, y)

    panel = _panel(task, y)
    if models:
        wanted = set(models)
        panel = [(n, e) for (n, e) in panel if n in wanted]

    entries = []
    for name, est in panel:
        try:
            pipe = Pipeline([("pre", pre), ("model", est)])
            s = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            entries.append(Entry(model=name, score=float(s.mean()), std=float(s.std())))
        except Exception:
            continue  # a model that can't fit this data simply doesn't appear
    entries.sort(key=lambda e: e.score, reverse=True)
    for i, e in enumerate(entries, 1):
        e.rank = i
    return Leaderboard(task=task, metric=metric, cv=cv, entries=entries)


def to_figure(lb):
    """A horizontal bar of the leaderboard in the firstlook theme (or None)."""
    if not lb.entries:
        return None
    import plotly.graph_objects as go
    ordered = list(reversed(lb.entries))  # best at top
    fig = go.Figure(go.Bar(
        x=[e.score for e in ordered],
        y=[e.model for e in ordered],
        orientation="h",
        error_x=dict(type="data", array=[e.std for e in ordered],
                     color="rgba(255,255,255,0.4)"),
        marker_color=theme.CYAN,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="firstlook", height=90 + 46 * len(ordered),
        margin=dict(l=10, r=20, t=46, b=34),
        title=f"model leaderboard — {lb.metric} ({lb.cv}-fold CV)",
        xaxis_title=lb.metric,
    )
    return fig
