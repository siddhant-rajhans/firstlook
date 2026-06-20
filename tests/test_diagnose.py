import numpy as np
import pandas as pd

from firstlook import diagnose, play


def test_detects_leakage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    df = pd.DataFrame({
        "real": rng.normal(size=200),
        "leak": y + rng.normal(scale=1e-6, size=200),  # ~perfectly predicts y
        "y": y,
    })
    diag = diagnose(df, target="y")
    leaks = diag.by_kind("leakage")
    assert any(f.column == "leak" for f in leaks)
    assert all(f.severity == "high" for f in leaks)


def test_detects_id_and_constant():
    n = 120
    df = pd.DataFrame({
        "id": [f"u{i}" for i in range(n)],     # all-unique -> id-like
        "const": ["x"] * n,                     # constant
        "real": np.random.default_rng(1).normal(size=n),
        "y": np.random.default_rng(2).integers(0, 2, size=n),
    })
    diag = diagnose(df, target="y")
    assert any(f.kind == "id_like" and f.column == "id" for f in diag.findings)
    assert any(f.kind == "constant" and f.column == "const" for f in diag.findings)


def test_detects_duplicates():
    base = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [5.0, 6, 7, 8], "y": [0, 1, 0, 1]})
    df = pd.concat([base, base.iloc[[0, 1]]], ignore_index=True)  # 2 dup rows
    diag = diagnose(df, target="y")
    dups = diag.by_kind("duplicates")
    assert dups and "2 duplicate rows" in dups[0].message


def test_detects_skew_with_log_suggestion():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "skewed": rng.exponential(scale=10, size=300) ** 2,  # heavy right skew
        "y": rng.normal(size=300),
    })
    diag = diagnose(df, target="y")
    skews = diag.by_kind("skew")
    assert any(f.column == "skewed" for f in skews)
    assert any("log" in (f.suggestion or "") for f in skews)


def test_clean_frame_has_no_high_findings():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "x2": rng.normal(size=300), "y": (x > 0).astype(int)})
    diag = diagnose(df, target="y")
    assert not diag.high  # nothing severe on clean, balanced data


def test_play_diagnose_attaches():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=80), "const": [1] * 80,
                       "y": rng.integers(0, 2, size=80)})
    r = play(df, target="y", diagnose=True, show=False)
    assert r.diagnosis is not None
    assert any(f.kind == "constant" for f in r.diagnosis.findings)
