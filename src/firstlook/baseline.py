"""Train the recommended baseline model and report an honest score.

Needs scikit-learn (the ``fit`` / ``dev`` extra). Preprocessing, the honest
metric rule, and CV folds all come from ``_sklearn`` so the single baseline and
the leaderboard stay in lock-step.
"""
from dataclasses import dataclass
from typing import Optional

from . import _sklearn


@dataclass
class Baseline:
    model: Optional[str]
    metric: Optional[str]
    score: Optional[float]
    std: Optional[float] = None
    cv: Optional[int] = None
    note: Optional[str] = None

    def __str__(self):
        if self.score is None:
            return f"baseline: {self.note}"
        return (f"baseline: {self.model}  {self.metric} {self.score:.3f} "
                f"+/- {self.std:.3f}  ({self.cv}-fold CV)")


def fit_baseline(df, target, task):
    """Cross-validate the recommended start model; return a :class:`Baseline`."""
    if task == "clustering" or target is None:
        return Baseline(None, None, None, note="no target, nothing to fit")
    _sklearn._require_sklearn()

    data = df.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])
    guard = _sklearn.fit_guard(X, y, task)
    if guard:
        return Baseline(None, None, None, note=guard)

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    pre, _, _ = _sklearn.build_preprocessor(X)
    metric, scoring = _sklearn.scoring_for(task, y)
    cv = _sklearn.cv_for(task, y)

    if task == "regression":
        model, name = LinearRegression(), "LinearRegression"
    else:
        name = "LogisticRegression"
        cw = "balanced" if _sklearn.needs_class_weight(y) else None
        model = LogisticRegression(max_iter=1000, class_weight=cw)

    pipe = Pipeline([("pre", pre), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    return Baseline(model=name, metric=metric, score=float(scores.mean()),
                    std=float(scores.std()), cv=cv)
