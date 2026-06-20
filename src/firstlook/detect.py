"""Figure out what kind of problem a dataframe poses."""
import pandas as pd

TASKS = ("regression", "classification", "clustering")


def detect_task(df, target=None):
    """Return ``"regression"``, ``"classification"``, or ``"clustering"``.

    No target -> clustering. A non-numeric target -> classification. A numeric
    target is classification only when it has few distinct values relative to
    the number of rows (encoded labels), otherwise regression. The ratio check
    is what stops a small regression set (e.g. 7 distinct MPG values in 7 rows)
    from being mistaken for 7-class classification.
    """
    if target is None:
        return "clustering"
    if target not in df.columns:
        raise KeyError(f"target {target!r} is not a column. Columns: {list(df.columns)}")

    y = df[target].dropna()
    if y.empty:
        return "clustering"

    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n, nun = len(y), y.nunique()
    if nun == 2:
        return "classification"  # binary is classification regardless of dtype/size
    if pd.api.types.is_float_dtype(y):
        return "classification" if (nun <= 10 and nun / n < 0.05) else "regression"
    # integer target
    return "classification" if (nun <= 20 and nun / n < 0.2) else "regression"
