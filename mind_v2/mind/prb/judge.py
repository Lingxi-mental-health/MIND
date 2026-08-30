"""Reliability assessment (§3.1.3) for PRB primitives.

The judge assigns a 1--5 score and three hard-issue flags ``H1/H2/H3``:

* ``H1`` — Dialogue inconsistency.
* ``H2`` — Reference irrelevance.
* ``H3`` — Unsupported clinical content.

Only primitives with ``score >= τ_rel`` *and* no hard-issue flag enter
:math:`\\mathcal{S}_t,\\mathcal{R}_t`.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

from .entry import PRBEntry
from ..utils.llm_client import LLMClient
from ..utils.prompts import RELIABILITY_JUDGE_PROMPT


def judge_entry(entry: PRBEntry, *, llm: LLMClient, max_tokens: int = 256) -> PRBEntry:
    """Score a single entry in place and return it."""

    user = (
        f"CONTEXT q_i: {entry.q}\n\n"
        f"PRIMITIVE:\n"
        f"(Criterion) {entry.criterion}\n"
        f"(Observed) {entry.observed_evidence}\n"
        f"(Missing) {entry.missing_check}\n"
        f"(Exclusion) {entry.exclusion_cue}\n"
        f"(Rationale) {entry.next_inquiry_rationale}\n"
    )
    raw = llm.chat(
        system=RELIABILITY_JUDGE_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    score, hard_issues = _parse_judge_response(raw)
    entry.reliability_score = score
    entry.hard_issues.update(hard_issues)
    return entry


def judge_all(entries: Iterable[PRBEntry], *, llm: LLMClient) -> List[PRBEntry]:
    return [judge_entry(e, llm=llm) for e in entries]


# ---------------------------------------------------------------------------
# Response parsing.
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"\(?score\)?\s*[:：]?\s*(\d(?:\.\d+)?)\s*/\s*5", re.IGNORECASE)
_HARD_RE = re.compile(r"H([123])\s*[:：]?\s*(YES|NO)", re.IGNORECASE)


def _parse_judge_response(raw: str) -> tuple[float, Dict[str, bool]]:
    score_match = _SCORE_RE.search(raw)
    score = float(score_match.group(1)) if score_match else 0.0

    hard = {"H1": False, "H2": False, "H3": False}
    for m in _HARD_RE.finditer(raw):
        hard[f"H{m.group(1)}"] = (m.group(2).upper() == "YES")
    return score, hard


__all__ = ["judge_entry", "judge_all"]
