"""Shared scikit-learn preprocessing + scoring logic.

The single place sklearn is touched for fitting/scoring, so the one-model
baseline (``baseline.py``) and the model panel (``leaderboard.py``) share the
exact same preprocessing, honest-metric rule, and CV-fold logic — no drift.
"""

_IMPORT_HINT = ("this needs scikit-learn; install it with "
                "`pip install 'firstlook[fit]'`")


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(_IMPORT_HINT) from e


def build_preprocessor(X):
    """Return ``(ColumnTransformer, num_cols, cat_cols)``.

    Numerics: median-impute then standard-scale. Categoricals: most-frequent-
    impute then one-hot (unknown-safe). Everything else dropped.
    """
    _require_sklearn()
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

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
    return pre, num, cat


def fit_guard(X, y, task):
    """Return a human-readable note if the data can't be scored reliably, else None."""
    if len(X) < 10 or X.shape[1] == 0:
        return "too few rows to score reliably"
    if task != "regression" and int(y.value_counts().min()) < 2:
        return "a class has <2 samples; can't cross-validate"
    return None


def needs_class_weight(y):
    """True when the majority class exceeds 70% of rows (imbalanced)."""
    vc = y.value_counts()
    return len(vc) > 0 and (vc.max() / len(y)) > 0.7


def scoring_for(task, y):
    """Return ``(metric_label, sklearn_scoring_key)`` honoring the imbalance rule.

    Plain accuracy flatters imbalanced data (a majority-only guesser scores
    high), so on imbalance we score with balanced accuracy instead.
    """
    if task == "regression":
        return "R2", "r2"
    if needs_class_weight(y):
        return "balanced accuracy", "balanced_accuracy"
    return "accuracy", "accuracy"


def cv_for(task, y):
    """Cross-validation fold count, capped by row/class counts."""
    if task == "regression":
        return min(5, max(2, len(y) // 4))
    return min(5, max(2, int(y.value_counts().min())))
