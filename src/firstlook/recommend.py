"""Recommend models to try, with reasons and data-aware warnings.

The model lists follow scikit-learn's algorithm cheat-sheet; the warnings come
from looking at the actual data (imbalance, categoricals, missing values, tiny n).
"""
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .detect import detect_task

_REGRESSION = [
    ("LinearRegression", "interpretable baseline, the right first try"),
    ("Ridge / Lasso", "linear + regularization; Lasso also drops weak features"),
    ("RandomForestRegressor", "non-linear, handles interactions, little tuning"),
    ("GradientBoosting / XGBoost", "usually the top scorer on tabular data"),
]

_CLASSIFICATION = [
    ("LogisticRegression", "interpretable baseline, gives probabilities"),
    ("KNeighborsClassifier", "simple, non-linear, nothing to train"),
    ("RandomForestClassifier", "strong default, handles mixed features"),
    ("GradientBoosting / XGBoost", "usually best on tabular data"),
]

_CLUSTERING = [
    ("KMeans", "the default when you roughly know how many groups"),
    ("DBSCAN", "arbitrary shapes, no k, flags outliers"),
    ("PCA -> cluster", "reduce dimensions first if many features"),
]


@dataclass
class Recommendation:
    task: str
    start: str
    models: List[Tuple[str, str]]
    notes: List[str]

    def __str__(self):  # plain-text rendering for scripts / print()
        lines = [f"task: {self.task}    start with: {self.start}", "", "models to try:"]
        lines += [f"  - {m:28} {r}" for m, r in self.models]
        if self.notes:
            lines += ["", "watch out for:"] + [f"  -> {n}" for n in self.notes]
        return "\n".join(lines)


def recommend(df, target=None, task=None):
    """Return a :class:`Recommendation` for ``df`` predicting ``target``."""
    task = task or detect_task(df, target)
    n = len(df)
    feats = [c for c in df.columns if c != target]
    n_cat = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in feats)
    notes = []

    if n < 50:
        notes.append(f"only {n} rows - any model will be shaky; more data is the biggest win")

    if task == "regression":
        start, models = "LinearRegression", _REGRESSION
    elif task == "classification":
        start, models = "LogisticRegression", _CLASSIFICATION
        if target is not None:
            y = df[target]
            notes.append(f"{y.nunique()} classes")
            vc = y.value_counts(normalize=True)
            if len(vc) and vc.max() > 0.7:
                notes.append(
                    f"imbalanced - {vc.idxmax()!r} is {vc.max()*100:.0f}% of rows; "
                    "watch recall, try class_weight"
                )
    else:
        start, models = "KMeans", _CLUSTERING

    if n_cat:
        notes.append(f"{n_cat} categorical feature(s) - encode (one-hot/ordinal) before fitting")
    if df.isna().any().any():
        notes.append("missing values present - impute or drop first")

    return Recommendation(task=task, start=start, models=models, notes=notes)
