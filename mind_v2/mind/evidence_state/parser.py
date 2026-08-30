"""Parse a clinical retrieval state :math:`q_t` into typed
:math:`\\mathcal{O}_t,\\mathcal{M}_t,\\mathcal{D}_t`.

The parser is intentionally rule-based.  It works on the canonical paragraph
produced by the PRB construction prompt (§3.1.1), e.g.::

    Patient reports low mood for 3 weeks; duration: 3 weeks;
    impairment: not mentioned; risk: denies SI; psychosis/mania cues: unclear;
    stressors: recent breakup; substances: not mentioned.

The output is consumed by :mod:`mind.prb.state_operator` to populate
:math:`\\mathcal{S}_t,\\mathcal{R}_t`.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .schema import MISSING_SENTINELS, RETRIEVAL_STATE_FIELDS
from .state import (
    DifferentialHypotheses,
    EvidenceState,
    MissingChecks,
    ObservedEvidence,
    ReliabilityMetadata,
    SupportSet,
)


# ---------------------------------------------------------------------------
# Field-level parsing helpers.
# ---------------------------------------------------------------------------

# Loose synonym map.  We accept English / Chinese / mixed phrasing so that
# the constructor LLM does not have to memorise the exact column name.
_FIELD_PATTERNS: Dict[str, List[str]] = {
    "chief_complaint":       ["chief complaint", "主诉", "complaint"],
    "symptoms_and_duration": ["symptoms", "duration", "病程", "症状"],
    "severity":              ["severity", "intensity", "严重程度"],
    "sleep":                 ["sleep", "睡眠"],
    "impairment":            ["impairment", "function", "功能"],
    "safety_risk":           ["risk", "self-harm", "suicidal", "safety", "安全风险"],
    "psychosis_mania_cues":  ["psychosis", "mania", "psychotic", "幻觉", "躁狂"],
    "stressors":             ["stressor", "trigger", "诱因", "压力"],
    "substance_use":         ["substance", "alcohol", "drug", "物质", "饮酒"],
}


def _is_missing(value: str) -> bool:
    if value is None:
        return True
    v = value.strip().lower()
    if not v:
        return True
    return any(s in v for s in MISSING_SENTINELS)


def _split_into_clauses(text: str) -> List[str]:
    # Accept ``;`` ``。`` ``；`` as primary clause separators.
    return [c.strip(" .,；。") for c in re.split(r"[;；。]", text) if c.strip()]


def parse_retrieval_state(q_t: str) -> Tuple[ObservedEvidence, MissingChecks]:
    """Parse :math:`q_t` into :math:`(\\mathcal{O}_t,\\mathcal{M}_t)`.

    Returns
    -------
    observed:
        Confirmed and explicitly negated facts.
    missing:
        List of fields whose value is one of :data:`MISSING_SENTINELS`.
    """

    confirmed: Dict[str, str] = {}
    negated: Dict[str, str] = {}
    missing_fields: List[str] = []

    seen: Dict[str, bool] = {f: False for f in RETRIEVAL_STATE_FIELDS}

    for clause in _split_into_clauses(q_t):
        # Identify which canonical field this clause speaks about.
        target = _match_field(clause)
        if target is None:
            continue
        seen[target] = True
        value = _strip_field_label(clause, target)

        if _is_missing(value):
            missing_fields.append(target)
            continue

        # Heuristic for explicit negation.
        if re.search(r"\b(no|denies|denied|否认|没有|无)\b", value, flags=re.IGNORECASE):
            negated[target] = value
        else:
            confirmed[target] = value

    # Fields that the LLM did not even mention are treated as missing.
    for f, was_seen in seen.items():
        if not was_seen and f not in missing_fields:
            missing_fields.append(f)

    return ObservedEvidence(confirmed=confirmed, negated=negated), MissingChecks(
        fields=missing_fields
    )


def _match_field(clause: str) -> str | None:
    low = clause.lower()
    for field_name, patterns in _FIELD_PATTERNS.items():
        for p in patterns:
            if p in low:
                return field_name
    return None


def _strip_field_label(clause: str, field_name: str) -> str:
    # Strip ``label:`` prefix if present so that we can analyse the value alone.
    return re.sub(rf"^[^:：]*[:：]\s*", "", clause).strip()


# ---------------------------------------------------------------------------
# Differential inference.
# ---------------------------------------------------------------------------

# Lightweight cue → category mapping aligned with the four-way ICD-10
# screening label space used in the paper.
_DIFFERENTIAL_CUES: Dict[str, List[str]] = {
    "Depression":    ["low mood", "anhedonia", "worthlessness", "抑郁", "情绪低落"],
    "Anxiety":       ["worry", "panic", "tension", "焦虑", "紧张"],
    "Mixed":         ["mixed", "comorbid", "混合", "焦虑抑郁"],
    "Other":         ["psychosis", "mania", "substance", "trauma", "ptsd"],
}


def infer_active_differentials(observed: ObservedEvidence) -> DifferentialHypotheses:
    """Heuristic differential proposal :math:`\\mathcal{D}_t`.

    The result is overwritten by the policy when the LLM emits an explicit
    differential trace; here we only initialise the field with a coarse cue
    match so that :math:`\\mathcal{D}_t\\ne\\emptyset` even before retrieval.
    """

    text = " ".join(list(observed.confirmed.values()) + list(observed.negated.values())).lower()
    active: List[str] = []
    for category, cues in _DIFFERENTIAL_CUES.items():
        if any(c in text for c in cues):
            active.append(category)
    if not active:
        # If we observed nothing yet, keep the four canonical categories on the
        # table so the policy is not forced to commit prematurely.
        active = list(_DIFFERENTIAL_CUES.keys())
    return DifferentialHypotheses(active=active, eliminated=[])


def build_initial_state(q_t: str) -> EvidenceState:
    """Convenience constructor used by :class:`mind.env.MindEnv`."""

    observed, missing = parse_retrieval_state(q_t)
    return EvidenceState(
        O=observed,
        M=missing,
        D=infer_active_differentials(observed),
        S=SupportSet(),
        R=ReliabilityMetadata(),
    )


__all__ = [
    "parse_retrieval_state",
    "infer_active_differentials",
    "build_initial_state",
]
