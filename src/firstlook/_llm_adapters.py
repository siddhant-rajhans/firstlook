"""Tiny, dependency-free LLM adapters — each returns a Completer ``(prompt) -> text``.

Plain stdlib ``urllib`` so firstlook's core needs no SDK. Covers any local or
cloud model: OpenAI-compatible servers (OpenAI, LM Studio, vLLM, many clouds),
Ollama (local), and Anthropic. API keys are read from the environment and never
logged.
"""
import json
import os
import urllib.error
import urllib.request


def _post_json(url, payload, headers, timeout=90):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"LLM request failed (HTTP {e.code}): {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach the LLM endpoint: {e.reason}") from None


def openai_completer(model, api_key=None, base_url="https://api.openai.com/v1"):
    """OpenAI-compatible /chat/completions — OpenAI, LM Studio, vLLM, many clouds."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = _post_json(url, {"model": model,
                                "messages": [{"role": "user", "content": prompt}]}, headers)
        return data["choices"][0]["message"]["content"].strip()
    return complete


def ollama_completer(model, host="http://localhost:11434"):
    """Local Ollama via /api/generate (no key needed)."""
    url = host.rstrip("/") + "/api/generate"

    def complete(prompt):
        data = _post_json(url, {"model": model, "prompt": prompt, "stream": False}, {})
        return (data.get("response") or "").strip()
    return complete


def anthropic_completer(model, api_key=None, max_tokens=1024):
    """Anthropic /v1/messages (raw HTTP — no SDK)."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    url = "https://api.anthropic.com/v1/messages"

    def complete(prompt):
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        data = _post_json(url, {"model": model, "max_tokens": max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}, headers)
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return "".join(parts).strip()
    return complete


def ollama_models(host="http://localhost:11434", timeout=2):
    """Return installed Ollama model names, or [] if Ollama isn't reachable."""
    try:
        req = urllib.request.Request(host.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
