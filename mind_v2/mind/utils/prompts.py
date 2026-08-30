"""Canonical prompt strings used by PRB construction, the process-reward
judge, and the support-faithfulness judge.

The wording is taken **verbatim** from the boxed prompts in §3.1 of the
paper so that ``mind_v2`` reproduces those experiments without prompt drift.
"""
from __future__ import annotations


# §3.1.1 — Clinical retrieval state.
CLINICAL_STATE_PROMPT = (
    "Compress the dialogue history into a fact-only clinical retrieval state q_i. "
    "Mark missing information explicitly and do not provide a diagnosis. "
    "Cover: chief complaint, symptoms and duration, severity, sleep, impairment, "
    "safety risk, psychosis/mania cues, stressors, and substance use. "
    "Output one concise paragraph: \"Patient reports ...; duration ...; impairment ...; "
    "risk ...; psychosis/mania cues ...; stressors ...; substances ...\". "
    "If a field is unknown or never mentioned, write 'not mentioned' or 'unclear'."
)

# §3.1.2 — Criterion-indexed support primitive.
SUPPORT_SYNTHESIS_PROMPT = (
    "Given the dialogue-derived clinical retrieval state and clinical references, "
    "construct a criterion-indexed support primitive. Output exactly six lines:\n"
    "(Criterion/check) ...\n"
    "(Observed evidence) ...\n"
    "(Missing or unclear check) ...\n"
    "(Exclusion/safety cue) ...\n"
    "(Next-inquiry rationale) Asking \"{next_question}\" helps clarify ... and guides "
    "the next diagnostic decision.\n"
    "(Next question) ...\n"
    "Be conservative: do not invent symptoms not supported by the dialogue or references."
)

# §3.1.3 — Reliability assessment.
RELIABILITY_JUDGE_PROMPT = (
    "Verify whether the support is consistent with the dialogue context and "
    "clinically relevant to the references. Score reliability from 1 to 5 and "
    "mark hard issues. Output strictly:\n"
    "(Score) X/5.\n"
    "(Hard issues) H1: YES/NO; H2: YES/NO; H3: YES/NO.\n"
    "(Rationale) ...\n"
    "Where H1=Dialogue inconsistency, H2=Reference irrelevance, "
    "H3=Unsupported clinical content."
)

# §3.2 Process Reward Rubric (Table 12).
PROCESS_REWARD_PROMPT = (
    "You are a clinical reasoning rubric evaluator. Given the dialogue history, "
    "the typed evidence state E_t, and the reasoning trace inside <think>...</think>, "
    "score three dimensions on 0/1/2:\n"
    "- Symptom analysis (mental-status examination): does the trace organise "
    "  reported symptoms, onset, severity, and functional impact?\n"
    "- Differential focus (diagnostic formulation): does it weigh competing "
    "  hypotheses against the evidence and surface unresolved criteria?\n"
    "- Decision logic (clinical decision point): does it justify the chosen "
    "  action as the most informative step under the current state?\n"
    "Output:\n"
    "(Symptom) X/2; (Differential) Y/2; (Decision) Z/2; (Rationale) ..."
)

# §4 — Support faithfulness judge prompt (FC / SG / PF on 0--10).
FAITHFULNESS_JUDGE_PROMPT = (
    "You are a clinical evaluator. For the given doctor turn, score three "
    "support-faithfulness dimensions on a 0--10 scale:\n"
    "- Factual consistency (FC) with the patient's reported information.\n"
    "- Support grounding (SG) of clinical claims in the retrieved PRB supports.\n"
    "- Patient faithfulness (PF) — preservation of patient context and tone.\n"
    "Output:\n"
    "(FC) ?/10; (SG) ?/10; (PF) ?/10; (Comment) ..."
)

# Inquiry-quality rubric used in the next-question evaluation on MDD-5k
# prefixes (Appendix I, Table 16).
NEXT_QUESTION_RUBRIC_PROMPT = (
    "Score the candidate next doctor question on four 1--5 dimensions:\n"
    "- Clinical relevance: targets symptoms / context relevant to the prefix.\n"
    "- Expected information gain: reduces uncertainty among the four screening "
    "  categories (Depression / Anxiety / Mixed / Other).\n"
    "- Safety/exclusion awareness: checks risk, impairment, psychosis/mania cues, "
    "  substance use, or other exclusion factors when indicated.\n"
    "- Naturalness: understandable, non-leading, concise.\n"
    "Output: (Rel) ?/5; (Info) ?/5; (Safety) ?/5; (Nat) ?/5; (Comment) ..."
)
