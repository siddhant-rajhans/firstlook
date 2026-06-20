"""Train the recommended baseline model and report an honest score.

Needs scikit-learn (the ``fit`` / ``dev`` extra). Preprocessing is built into
the pipeline (median-impute + scale numerics, most-frequent-impute + one-hot
categoricals), so it fits straight on the messy data the recommender only warns
about. Scoring is cross-validated, so a tiny test split can't flatter or wreck it.
"""
from dataclasses import dataclass
from typing import Optional


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
    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
    except ImportError as e:
        raise ImportError(
            "baseline fitting needs scikit-learn; install it with "
            "`pip install 'firstlook[fit]'`"
        ) from e

    data = df.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target])
    n = len(data)
    if n < 10 or X.shape[1] == 0:
        return Baseline(None, None, None, note="too few rows to score reliably")

    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )

    if task == "regression":
        model, name, metric, scoring = LinearRegression(), "LinearRegression", "R2", "r2"
        cv = min(5, max(2, n // 4))
    else:
        vc = y.value_counts()
        per_class = int(vc.min())
        if per_class < 2:
            return Baseline("LogisticRegression", "accuracy", None,
                            note="a class has <2 samples; can't cross-validate")
        model = LogisticRegression(max_iter=1000)
        name = "LogisticRegression"
        cv = min(5, max(2, per_class))
        # plain accuracy flatters imbalanced data (a majority-only guesser scores
        # high), so switch to balanced accuracy exactly when imbalance is flagged.
        if (vc.max() / len(y)) > 0.7:
            metric, scoring = "balanced accuracy", "balanced_accuracy"
        else:
            metric, scoring = "accuracy", "accuracy"

    pipe = Pipeline([("pre", pre), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    return Baseline(model=name, metric=metric, score=float(scores.mean()),
                    std=float(scores.std()), cv=cv)
