"""Schema for the clinical retrieval state :math:`q_t`.

The fields exactly follow §3.1.1 (``Clinical Retrieval State``) of the paper:
chief complaint, symptoms and duration, severity, sleep, impairment, safety
risk, psychosis/mania cues, stressors, and substance use.  Unknown fields
must be marked as ``not mentioned`` or ``unclear`` so that
:class:`mind.evidence_state.parser` can extract :math:`\\mathcal{M}_t`.
"""
from __future__ import annotations

# Canonical ordering used everywhere in MIND-v2.
RETRIEVAL_STATE_FIELDS = (
    "chief_complaint",
    "symptoms_and_duration",
    "severity",
    "sleep",
    "impairment",
    "safety_risk",
    "psychosis_mania_cues",
    "stressors",
    "substance_use",
)

# Human-readable description for each field; consumed by the prompts in
# ``mind_v2/prompts/`` and rendered to the policy when verbalising
# :math:`\\mathcal{E}_t`.
FIELD_DESCRIPTIONS = {
    "chief_complaint":        "patient's main reason for the visit",
    "symptoms_and_duration":  "current psychiatric symptoms and how long they have lasted",
    "severity":               "subjective severity / frequency / intensity",
    "sleep":                  "sleep quality, onset / maintenance / early-awakening",
    "impairment":             "functional impairment in work / school / social life",
    "safety_risk":            "self-harm, suicidal ideation, harm-to-others, agitation",
    "psychosis_mania_cues":   "psychotic symptoms, manic / hypomanic episodes",
    "stressors":              "recent psychosocial stressors / triggers",
    "substance_use":          "substance / medication / alcohol use",
}

# Sentinels that mark a field as a missing-check candidate.  See §3.1.1 and
# Appendix D (Automatic MCR computation).
MISSING_SENTINELS = (
    "not mentioned",
    "unclear",
    "未提及",
    "不清楚",
    "n/a",
    "unknown",
    "to be assessed",
)

# Hard-issue flag identifiers used by the reliability judge (see §3.1.3).
HARD_ISSUE_FLAGS = ("H1", "H2", "H3")
