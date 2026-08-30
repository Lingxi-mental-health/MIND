"""Pluggable LLM client used for offline PRB construction and judge calls.

The client deliberately avoids any heavy SDK dependency: it accepts a
``provider`` name and a callable that, given ``(system, user, **kwargs)``,
returns a string.  Concrete adapters are registered in ``ADAPTERS`` and can be
extended by users via :func:`register_adapter`.

Default adapters
----------------
``echo``      Deterministic mock used by unit tests / static checks.
``openrouter``HTTP adapter for OpenRouter-style endpoints (Kimi-K2, GLM-4.7).
``deepseek``  HTTP adapter for DeepSeek API.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


ChatFn = Callable[[str, str, Dict[str, Any]], str]


@dataclass
class LLMClient:
    """A thin wrapper around a chat function.

    Parameters
    ----------
    provider:
        Adapter name, must be present in :data:`ADAPTERS` (or registered via
        :func:`register_adapter`).
    model:
        Model identifier (e.g. ``moonshotai/kimi-k2``).
    api_key:
        Optional API key.  If ``None``, falls back to environment variables.
    """

    provider: str = "echo"
    model: str = "stub"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.extra = self.extra or {}

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        adapter = ADAPTERS.get(self.provider)
        if adapter is None:
            raise KeyError(
                f"No LLM adapter registered for provider={self.provider!r}. "
                f"Known adapters: {sorted(ADAPTERS)}"
            )
        return adapter(self, system, user, kwargs)


# ---------------------------------------------------------------------------
# Adapter registry.
# ---------------------------------------------------------------------------

ADAPTERS: Dict[str, Callable[[LLMClient, str, str, Dict[str, Any]], str]] = {}


def register_adapter(
    name: str,
) -> Callable[[Callable[[LLMClient, str, str, Dict[str, Any]], str]],
              Callable[[LLMClient, str, str, Dict[str, Any]], str]]:
    def deco(fn):
        ADAPTERS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Built-in adapters.
# ---------------------------------------------------------------------------

@register_adapter("echo")
def _echo(self_: LLMClient, system: str, user: str, kw: Dict[str, Any]) -> str:
    """Deterministic mock that simply echoes a small JSON-like structure.

    Useful for unit tests and dry-runs of the pipeline without any network.
    """

    return (
        "(Criterion/check) Major depressive episode duration\n"
        "(Observed evidence) Patient reports low mood for 3 weeks\n"
        "(Missing or unclear check) Sleep / functional impairment / safety risk\n"
        "(Exclusion/safety cue) Denies suicidal ideation\n"
        "(Next-inquiry rationale) Asking about sleep clarifies neurovegetative criteria\n"
        "(Next question) How has your sleep been over the past two weeks?\n"
    )


@register_adapter("openrouter")
def _openrouter(self_: LLMClient, system: str, user: str, kw: Dict[str, Any]) -> str:
    """Minimal OpenRouter adapter (used for Kimi-K2 / GLM-4.7 in §F)."""

    import urllib.request

    api_key = self_.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    url = self_.base_url or "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": self_.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": kw.get("temperature", 0.2),
        "max_tokens": kw.get("max_tokens", 1024),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
                return obj["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - network path
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after retries: {last_err!r}")


@register_adapter("deepseek")
def _deepseek(self_: LLMClient, system: str, user: str, kw: Dict[str, Any]) -> str:
    """Minimal DeepSeek API adapter."""

    import urllib.request

    api_key = self_.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    url = self_.base_url or "https://api.deepseek.com/chat/completions"
    payload = {
        "model": self_.model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": kw.get("temperature", 0.0),
        "max_tokens": kw.get("max_tokens", 1024),
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # pragma: no cover
        obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]


__all__ = ["LLMClient", "register_adapter", "ADAPTERS"]
