"""Public surface of :mod:`mind.evidence_state`.

>>> from mind_v2.mind.evidence_state import EvidenceState, build_initial_state
>>> state = build_initial_state("Patient reports low mood for 3 weeks; sleep: unclear")
"""
from .schema import (
    FIELD_DESCRIPTIONS,
    HARD_ISSUE_FLAGS,
    MISSING_SENTINELS,
    RETRIEVAL_STATE_FIELDS,
)
from .state import (
    DifferentialHypotheses,
    EvidenceState,
    MissingChecks,
    ObservedEvidence,
    ReliabilityMetadata,
    SupportPrimitive,
    SupportSet,
)
from .parser import (
    build_initial_state,
    infer_active_differentials,
    parse_retrieval_state,
)

__all__ = [
    "EvidenceState",
    "ObservedEvidence",
    "MissingChecks",
    "DifferentialHypotheses",
    "SupportSet",
    "SupportPrimitive",
    "ReliabilityMetadata",
    "RETRIEVAL_STATE_FIELDS",
    "FIELD_DESCRIPTIONS",
    "HARD_ISSUE_FLAGS",
    "MISSING_SENTINELS",
    "parse_retrieval_state",
    "infer_active_differentials",
    "build_initial_state",
]
