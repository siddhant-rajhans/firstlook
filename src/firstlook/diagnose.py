"""Data doctor — flag the problems that quietly wreck models.

Pure pandas/numpy (no sklearn). Turns the recommender's shallow notes into a
real diagnostic pass: target leakage, ID-like / constant columns, near-duplicate
rows, heavy skew/outliers, plus light feature-engineering suggestions.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .detect import detect_task

_SEV_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Finding:
    kind: str          # leakage | id_like | constant | duplicates | skew | outliers | fe
    column: Optional[str]
    severity: str      # high | medium | low
    message: str
    suggestion: Optional[str] = None


@dataclass
class Diagnosis:
    findings: List[Finding] = field(default_factory=list)

    def by_kind(self, kind):
        return [f for f in self.findings if f.kind == kind]

    @property
    def high(self):
        return [f for f in self.findings if f.severity == "high"]

    def __bool__(self):
        return bool(self.findings)

    def __str__(self):
        if not self.findings:
            return "data doctor: nothing alarming."
        out = ["data doctor:"]
        for f in self.findings:
            line = f"  [{f.severity}] {f.message}"
            if f.suggestion:
                line += f"  -> {f.suggestion}"
            out.append(line)
        return "\n".join(out)


def diagnose(df, target=None, task=None, max_rows=50000):
    """Inspect ``df`` for modeling hazards; return a :class:`Diagnosis`.

    Structural checks (constant/ID/duplicate columns) run on the full frame;
    the heavier numeric stats (skew, outliers, leakage) run on a sample when the
    frame is large.
    """
    task = task or detect_task(df, target)
    d = df.sample(max_rows, random_state=0) if len(df) > max_rows else df
    feats = [c for c in df.columns if c != target]
    findings: List[Finding] = []

    # --- row-level: near-duplicates (full frame) ---
    dup = int(df.duplicated().sum())
    if dup:
        pct = 100 * dup / len(df)
        findings.append(Finding("duplicates", None, "medium" if pct > 1 else "low",
                                f"{dup} duplicate rows ({pct:.0f}%)",
                                "drop exact duplicates before modeling"))

    # --- per-column ---
    for c in feats:
        full = df[c]
        nun = int(full.nunique(dropna=True))

        if nun <= 1:
            findings.append(Finding("constant", c, "high", f"'{c}' is constant",
                                    "drop it - it carries no signal"))
            continue

        top_share = full.value_counts(normalize=True, dropna=True).iloc[0]
        if top_share > 0.99:
            findings.append(Finding("constant", c, "medium",
                                    f"'{c}' is almost constant ({top_share*100:.0f}% one value)",
                                    "likely low signal; consider dropping"))

        if not pd.api.types.is_float_dtype(full) and nun == len(df) and len(df) > 10:
            findings.append(Finding("id_like", c, "medium",
                                    f"'{c}' looks like an ID (every value unique)",
                                    "drop it - identifiers don't generalize"))

        if pd.api.types.is_numeric_dtype(full) and not pd.api.types.is_bool_dtype(full):
            v = pd.to_numeric(d[c], errors="coerce").dropna().astype(float)
            if len(v) > 5:
                sk = float(v.skew())
                if abs(sk) > 2:
                    findings.append(Finding("skew", c, "low",
                                            f"'{c}' is heavily skewed (skew={sk:.1f})",
                                            "consider a log / log1p transform"))
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    share = float(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean())
                    if share > 0.02:
                        findings.append(Finding("outliers", c, "low",
                                                f"'{c}' has {share*100:.0f}% extreme outliers",
                                                "inspect or clip before scaling"))

        if pd.api.types.is_datetime64_any_dtype(full):
            findings.append(Finding("fe", c, "low", f"'{c}' is a datetime column",
                                    "extract parts (year/month/day-of-week) or cyclical-encode"))
        elif full.dtype == object and 50 < nun < len(df):
            findings.append(Finding("fe", c, "low",
                                    f"'{c}' is high-cardinality categorical ({nun} levels)",
                                    "target/ordinal-encode rather than one-hot"))

    # --- leakage (needs a target) ---
    if target is not None and task != "clustering":
        from .visualize import _corr_ratio
        y = d[target]
        y_num = pd.to_numeric(y, errors="coerce") if task == "regression" else None
        for c in feats:
            if not pd.api.types.is_numeric_dtype(d[c]):
                continue
            try:
                if task == "classification":
                    assoc = _corr_ratio(y, d[c])
                else:
                    assoc = abs(np.corrcoef(pd.to_numeric(d[c], errors="coerce").astype(float),
                                            y_num.astype(float))[0, 1])
                if np.isfinite(assoc) and assoc > 0.98:
                    findings.append(Finding("leakage", c, "high",
                                            f"'{c}' almost perfectly predicts the target (assoc={assoc:.2f})",
                                            "possible leakage - is this value known at prediction time?"))
            except Exception:
                continue

    findings.sort(key=lambda f: _SEV_RANK.get(f.severity, 3))
    return Diagnosis(findings=findings)
