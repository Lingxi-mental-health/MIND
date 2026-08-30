"""Rule-based triggers for state-conditioned trajectory rectification.

Each trigger is a pure function of the evidence-state fields plus the raw
output (Algorithm 1 + Table 13).  Triggers do *not* mutate state — that is the
job of :class:`mind.rectification.rectifier.Rectifier`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

from ..evidence_state.state import EvidenceState
from ..reward.repetition import has_recent_duplicate
from ..utils.format import parse_stage_ii


class TriggerKind(str, Enum):
    MALFORMED          = "malformed"
    REPEATED_INQUIRY   = "repeated_inquiry"
    GENERIC_LOW_YIELD  = "generic_low_yield"
    LOW_RELIABILITY    = "low_reliability"
    BUDGET_VIOLATION   = "budget_violation"
    UNSAFE_OFF_TRACK   = "unsafe_off_track"


@dataclass
class TriggerHit:
    kind: TriggerKind
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return True


# ---------------------------------------------------------------------------
# Individual triggers.
# ---------------------------------------------------------------------------

def _generic_low_yield(question: str) -> bool:
    """Heuristic: question does not target any decisive criterion category."""

    if not question:
        return False
    low = question.lower()
    targets = (
        "symptom", "duration", "sleep", "impair", "function",
        "risk", "self-harm", "suic", "psycho", "mania",
        "stress", "trigger", "alcohol", "substance",
        "症状", "持续", "睡眠", "功能", "风险", "自杀", "压力", "幻觉", "躁狂",
    )
    return not any(t in low for t in targets)


def detect_triggers(
    *,
    raw_response: str,
    state: EvidenceState,
    history_questions: Iterable[str],
    last_turn: bool,
    made_diagnosis: bool,
    off_track_keywords: Optional[List[str]] = None,
) -> List[TriggerHit]:
    """Return *all* triggers that fired for the current turn."""

    hits: List[TriggerHit] = []

    # 1. Malformed output.
    parsed = parse_stage_ii(raw_response)
    if not parsed.is_valid:
        hits.append(TriggerHit(TriggerKind.MALFORMED, "Stage-II tags missing"))
    if parsed.is_valid and not parsed.is_diagnosis():
        # 2. Repeated inquiry.
        if has_recent_duplicate(parsed.answer, history_questions):
            hits.append(TriggerHit(TriggerKind.REPEATED_INQUIRY, parsed.answer[:80]))
        # 3. Generic low-yield.
        if _generic_low_yield(parsed.answer):
            hits.append(TriggerHit(TriggerKind.GENERIC_LOW_YIELD, parsed.answer[:80]))

    # 4. Low retrieval reliability.
    if state.R.gated_in and not any(state.R.gated_in):
        hits.append(TriggerHit(TriggerKind.LOW_RELIABILITY, "all PRB candidates rejected"))

    # 5. Budget violation.
    if last_turn and not made_diagnosis:
        hits.append(TriggerHit(
            TriggerKind.BUDGET_VIOLATION, "no diagnosis at last turn"
        ))

    # 6. Unsafe / off-track.
    keywords = off_track_keywords or [
        "i love you", "as an ai", "as a language model", "haha",
    ]
    if parsed.answer and any(k in parsed.answer.lower() for k in keywords):
        hits.append(TriggerHit(TriggerKind.UNSAFE_OFF_TRACK, parsed.answer[:80]))

    return hits


__all__ = ["TriggerKind", "TriggerHit", "detect_triggers"]
