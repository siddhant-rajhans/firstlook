"""Turn a Report into an honest plain-English narrative, using any LLM.

Provider-agnostic and SDK-free: ``explain`` takes a ``Completer`` callable
(``prompt -> text``) — bring your own, or let firstlook resolve a built-in
adapter from a ``"provider:model"`` string, a config dict, or auto-detection
(env vars / a local Ollama). The prompt is built from the *structured* Report
only — task, ranked features, recommendation notes, baseline/leaderboard — never
the raw rows.
"""
import os
from dataclasses import dataclass
from typing import Callable, Optional

Completer = Callable[[str], str]  # prompt in, text out


@dataclass
class Narrative:
    text: str
    model: Optional[str] = None

    def __str__(self):
        return self.text

    def _repr_markdown_(self):  # Jupyter renders this as markdown
        return self.text


def build_prompt(report, max_features=5):
    """A grounded, honesty-first prompt from the structured report (no raw data)."""
    r = report.recommendation
    L = [
        "You are a careful data scientist writing for someone about to model a dataset.",
        "Using ONLY the facts below, write a short, honest read of the data: 4-6 "
        "sentences, plain English, no markdown headings. Do not invent numbers, "
        "columns, or models that are not listed. If a score is modest, or the data is "
        "small or imbalanced, say so plainly instead of overselling. End with one "
        "concrete next step.",
        "",
        "FACTS:",
        f"- Problem type: {report.task}",
        f"- Shape: {report.shape[0]} rows, {report.shape[1]} columns",
        f"- Target column: {report.target}",
    ]
    if report.key_features:
        feats = ", ".join(map(str, report.key_features[:max_features]))
        L.append(f"- Most informative features: {feats}")
    L.append(f"- Recommended starting model: {r.start}")
    L.append(f"- Candidate models: {', '.join(m for m, _ in r.models)}")
    if r.notes:
        L.append(f"- Data-quality warnings: {'; '.join(r.notes)}")
    b = report.baseline
    if b is not None and getattr(b, "score", None) is not None:
        L.append(f"- Cross-validated baseline: {b.model} scored {b.metric} "
                 f"{b.score:.3f} (+/- {b.std:.3f})")
    lb = report.leaderboard
    if lb is not None and getattr(lb, "entries", None):
        tops = "; ".join(f"{e.model} {e.score:.3f}" for e in lb.entries[:4])
        L.append(f"- Model leaderboard ({lb.metric}): {tops}")
    return "\n".join(L)


def resolve_completer(using=None):
    """Return ``(completer, label)`` from a callable, ``"provider:model"`` string,
    config dict, or auto-detection. Never makes a network call here except the
    cheap Ollama probe used during auto-detection."""
    from . import _llm_adapters as A

    if callable(using):
        return using, "custom"

    if isinstance(using, dict):
        provider = (using.get("provider") or "openai").lower()
        model = using.get("model", "")
        if provider in ("openai", "openai-compatible", "lmstudio", "vllm"):
            return A.openai_completer(model, api_key=using.get("api_key"),
                                      base_url=using.get("base_url", "https://api.openai.com/v1")), f"{provider}:{model}"
        if provider == "ollama":
            return A.ollama_completer(model, host=using.get("base_url", "http://localhost:11434")), f"ollama:{model}"
        if provider == "anthropic":
            return A.anthropic_completer(model, api_key=using.get("api_key")), f"anthropic:{model}"
        raise ValueError(f"unknown LLM provider: {provider!r}")

    if isinstance(using, str):
        provider, _, model = using.partition(":")
        provider = provider.lower()
        if provider == "ollama":
            return A.ollama_completer(model or "llama3"), f"ollama:{model or 'llama3'}"
        if provider == "anthropic":
            return A.anthropic_completer(model or "claude-opus-4-8"), f"anthropic:{model or 'claude-opus-4-8'}"
        # default: treat as an OpenAI-compatible model name
        m = model or provider  # allow a bare model name too
        return A.openai_completer(m), f"openai:{m}"

    # using is None -> auto-detect (local-first)
    spec = os.environ.get("FIRSTLOOK_LLM")
    if spec:
        return resolve_completer(spec)
    if os.environ.get("OPENAI_API_KEY"):
        return A.openai_completer("gpt-4o-mini"), "openai:gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return A.anthropic_completer("claude-opus-4-8"), "anthropic:claude-opus-4-8"
    models = A.ollama_models()
    if models:
        return A.ollama_completer(models[0]), f"ollama:{models[0]}"
    raise RuntimeError(
        "no LLM configured for explain=. Pass a callable, a 'provider:model' string "
        "(e.g. 'ollama:llama3', 'openai:gpt-4o-mini', 'anthropic:claude-opus-4-8'), "
        "set FIRSTLOOK_LLM / OPENAI_API_KEY / ANTHROPIC_API_KEY, or run a local Ollama."
    )


def explain(report, using=None, max_features=5):
    """Narrate a :class:`~firstlook.report.Report` with any LLM. Returns a :class:`Narrative`."""
    completer, label = resolve_completer(using)
    text = completer(build_prompt(report, max_features=max_features))
    return Narrative(text=text.strip(), model=label)
