"""Offline construction of the Psychiatric Reasoning Bank (PRB).

This module realises the three offline steps in §3.1:

1. ``build_clinical_state`` — distills :math:`q_i` from a training-split
   dialogue (§3.1.1).
2. ``synthesise_support`` — produces the criterion-indexed primitive
   :math:`(c_i,o_i,g_i,x_i,u_i)` (§3.1.2).
3. :mod:`mind.prb.judge` — assigns the reliability metadata :math:`\\rho_i`
   (§3.1.3, separate file).

All three steps use a plug-in :class:`mind.utils.llm_client.LLMClient` so that
the construction LLM (``Kimi-K2`` / ``DeepSeek-V3.2`` / ``GLM-4.7``) can be
swapped without touching downstream code.  This is the design tested in the
PRB-construction-LLM sensitivity study (Appendix F, Table 8).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from .entry import PRBEntry
from ..utils.llm_client import LLMClient
from ..utils.prompts import (
    CLINICAL_STATE_PROMPT,
    SUPPORT_SYNTHESIS_PROMPT,
)


# ---------------------------------------------------------------------------
# Step 1 — clinical retrieval state q_i.
# ---------------------------------------------------------------------------

def build_clinical_state(
    dialogue: Iterable[Dict[str, str]],
    *,
    llm: LLMClient,
    max_tokens: int = 600,
) -> str:
    """Produce :math:`q_i` from a single training-split dialogue.

    Parameters
    ----------
    dialogue:
        An iterable of ``{"role": "doctor"/"patient", "content": ...}`` dicts.
    llm:
        Any :class:`LLMClient` (Kimi-K2 by default in the paper).

    Returns
    -------
    A fact-only paragraph that follows the canonical schema documented in
    :mod:`mind.evidence_state.schema`.
    """

    history = "\n".join(f"{turn['role']}: {turn['content']}" for turn in dialogue)
    return llm.chat(
        system=CLINICAL_STATE_PROMPT,
        user=f"DIALOGUE:\n{history}",
        max_tokens=max_tokens,
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# Step 2 — criterion-indexed support primitive.
# ---------------------------------------------------------------------------

def synthesise_support(
    q_i: str,
    *,
    llm: LLMClient,
    references: str = "",
    next_question_hint: str = "",
    max_tokens: int = 800,
) -> Dict[str, str]:
    """Return a dict with keys ``criterion / observed / missing / exclusion /
    rationale / next_question`` (§3.1.2)."""

    user = (
        f"CLINICAL_RETRIEVAL_STATE:\n{q_i}\n\n"
        f"CLINICAL_REFERENCES:\n{references or '(no extra references)'}\n\n"
        f"REFERENCE_NEXT_QUESTION (optional, may be empty):\n{next_question_hint}"
    )
    raw = llm.chat(
        system=SUPPORT_SYNTHESIS_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return _parse_support_response(raw, next_question_hint)


def _parse_support_response(raw: str, next_question_hint: str) -> Dict[str, str]:
    """Tolerant parser for the ``(Criterion) ...; (Observed) ...; ...`` format."""

    fields = {
        "criterion":      "",
        "observed":       "",
        "missing":        "",
        "exclusion":      "",
        "rationale":      "",
        "next_question":  next_question_hint,
    }
    for line in raw.splitlines():
        low = line.strip()
        for key, label in (
            ("criterion", "criterion"),
            ("observed", "observed"),
            ("missing", "missing"),
            ("exclusion", "exclusion"),
            ("rationale", "next-inquiry"),
            ("next_question", "next question"),
        ):
            if label in low.lower():
                # Strip ``(label) ...`` prefix.
                value = low.split(")", 1)[-1].strip(": ").strip()
                if not value:
                    continue
                fields[key] = value
                break
    return fields


# ---------------------------------------------------------------------------
# JSONL helpers.
# ---------------------------------------------------------------------------

def write_entries(entries: Iterable[PRBEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")


def read_entries(path: str) -> List[PRBEntry]:
    with open(path, "r", encoding="utf-8") as f:
        return [PRBEntry.from_dict(json.loads(line)) for line in f if line.strip()]


__all__ = [
    "build_clinical_state",
    "synthesise_support",
    "write_entries",
    "read_entries",
]
