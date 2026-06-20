import sys

import numpy as np
import pandas as pd
import pytest

from firstlook import explain, play
from firstlook.explain import build_prompt, resolve_completer, Narrative


def _report():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f1": rng.normal(size=80),
        "f2": rng.normal(size=80),
        "cat": ["SECRET_CELL_VALUE"] * 80,   # raw values must NOT leak into the prompt
        "y": rng.integers(0, 2, size=80),
    })
    return play(df, target="y", show=False)


def test_build_prompt_is_grounded_and_leaks_no_raw_values():
    p = build_prompt(_report())
    assert "classification" in p          # task
    assert "y" in p                       # target
    assert "LogisticRegression" in p      # recommended model
    assert "SECRET_CELL_VALUE" not in p   # no raw cell values in the prompt


def test_stub_completer_works():
    n = explain(_report(), using=lambda prompt: "a short honest read.")
    assert isinstance(n, Narrative)
    assert n.text == "a short honest read."
    assert n.model == "custom"


def test_play_explain_attaches_narrative():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.integers(0, 2, size=60)})
    r = play(df, target="y", explain=lambda p: "stub narrative", show=False)
    assert r.narrative is not None and r.narrative.text == "stub narrative"


def test_provider_string_resolves_without_network():
    fn, label = resolve_completer("openai:gpt-4o-mini")
    assert callable(fn) and label == "openai:gpt-4o-mini"
    fn2, label2 = resolve_completer("ollama:llama3")
    assert callable(fn2) and label2 == "ollama:llama3"
    fn3, label3 = resolve_completer("anthropic:claude-opus-4-8")
    assert callable(fn3) and label3 == "anthropic:claude-opus-4-8"


def test_explain_failure_is_graceful():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.integers(0, 2, size=60)})

    def boom(prompt):
        raise RuntimeError("no llm reachable")

    with pytest.warns(UserWarning):
        r = play(df, target="y", explain=boom, show=False)
    assert r.narrative is None  # degraded to None, did not crash at()


def test_core_imports_no_llm_sdk():
    import firstlook  # noqa: F401
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules


def test_auto_detect_errors_when_unconfigured(monkeypatch):
    monkeypatch.delenv("FIRSTLOOK_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import firstlook._llm_adapters as A
    monkeypatch.setattr(A, "ollama_models", lambda *a, **k: [])
    with pytest.raises(RuntimeError):
        resolve_completer(None)
