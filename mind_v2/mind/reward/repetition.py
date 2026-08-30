"""Semantic duplicate detection (§3.2.4, repetition trigger).

For static-alignment purposes we use a token-overlap proxy.  At runtime, the
``ragen`` env has a stronger LLM-based duplicate detector — :class:`MindEnv`
will prefer that detector when an ``env_llm_worker`` is available.
"""
from __future__ import annotations

from typing import Iterable


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def is_semantic_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    return jaccard(a, b) >= threshold


def has_recent_duplicate(
    candidate: str, history: Iterable[str], window: int = 4, threshold: float = 0.6
) -> bool:
    history_list = list(history)[-window:]
    return any(is_semantic_duplicate(candidate, h, threshold) for h in history_list)


__all__ = ["jaccard", "is_semantic_duplicate", "has_recent_duplicate"]
