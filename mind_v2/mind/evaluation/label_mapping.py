"""Canonical four-way ICD-10-style label mapping (§4 main paper, §G mapping).

The released MDD-5k diagnoses are mapped as follows:

* depressive diagnoses                       → Depression
* anxiety diagnoses                          → Anxiety
* anxiety--depression / co-occurring         → Mixed
* everything else                            → Other
"""
from __future__ import annotations

import re
from typing import Iterable

CATEGORIES = ("Depression", "Anxiety", "Mixed", "Other")


_DEPRESSION_KEYWORDS = (
    "depression", "depressive", "mdd", "dysthymia", "抑郁",
)
_ANXIETY_KEYWORDS = (
    "anxiety", "gad", "panic", "phobia", "焦虑",
)
_MIXED_KEYWORDS = (
    "mixed", "anxiety-depression", "anxiety and depression", "co-occurring",
    "comorbid", "焦虑抑郁", "混合",
)


def canonical_label(text: str) -> str:
    """Map any free-form diagnosis text into one of :data:`CATEGORIES`."""

    if not text:
        return "Other"
    low = text.lower()
    if _matches(low, _MIXED_KEYWORDS):
        return "Mixed"
    if _matches(low, _DEPRESSION_KEYWORDS) and _matches(low, _ANXIETY_KEYWORDS):
        return "Mixed"
    if _matches(low, _DEPRESSION_KEYWORDS):
        return "Depression"
    if _matches(low, _ANXIETY_KEYWORDS):
        return "Anxiety"
    return "Other"


def _matches(text: str, keywords: Iterable[str]) -> bool:
    for k in keywords:
        if k in text:
            return True
    return False


__all__ = ["CATEGORIES", "canonical_label"]
