"""State-conditioned trajectory rectifier (§3.2.4, Algorithm 1).

Behaviour:

1. On any trigger, request a *constrained self-retry* (at most once per turn).
2. If the retry still violates the same trigger and the per-episode fallback
   cap is not reached, replace the action with a structured reference inquiry
   built from the nearest reliable PRB entry.

The rectifier never silently passes through invalid output; if both the retry
and the fallback fail, it raises :class:`RectificationFailure` so that the
training loop can apply the corresponding penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from .triggers import TriggerHit, TriggerKind, detect_triggers
from ..evidence_state.state import EvidenceState, SupportPrimitive
from ..utils.format import parse_stage_ii, render_stage_ii


class RectificationFailure(RuntimeError):
    """Raised when both retry and fallback fail to produce a valid action."""


@dataclass
class RectificationCaps:
    self_retry_per_turn: int = 1     # see Table 13
    fallback_per_episode: int = 2
    total_per_episode: int = 3


@dataclass
class RectificationDecision:
    final_response: str
    used_self_retry: bool
    used_fallback: bool
    triggers: List[TriggerHit] = field(default_factory=list)
    fallback_primitive: Optional[SupportPrimitive] = None


@dataclass
class Rectifier:
    """High-level state-conditioned recovery operator.

    Parameters
    ----------
    retry_fn:
        Callable that receives the original response and the list of triggered
        failure modes; should return a fresh response string.  Typically wraps
        an LLM call with a stricter system prompt.
    """

    caps: RectificationCaps = field(default_factory=RectificationCaps)
    retry_fn: Optional[Callable[[str, List[TriggerHit], EvidenceState], str]] = None

    # Episode-level counters (reset by ``new_episode``).
    _fallback_used: int = 0
    _total_used: int = 0

    # ------------------------------------------------------------------
    def new_episode(self) -> None:
        self._fallback_used = 0
        self._total_used = 0

    # ------------------------------------------------------------------
    def __call__(
        self,
        *,
        raw_response: str,
        state: EvidenceState,
        history_questions: Iterable[str],
        last_turn: bool,
        made_diagnosis: bool,
    ) -> RectificationDecision:
        triggers = detect_triggers(
            raw_response=raw_response,
            state=state,
            history_questions=history_questions,
            last_turn=last_turn,
            made_diagnosis=made_diagnosis,
        )

        decision = RectificationDecision(
            final_response=raw_response,
            used_self_retry=False,
            used_fallback=False,
            triggers=list(triggers),
        )
        if not triggers:
            return decision
        if self._total_used >= self.caps.total_per_episode:
            return decision

        # 1. Constrained self-retry.
        if self.retry_fn is not None and self.caps.self_retry_per_turn > 0:
            retried = self.retry_fn(raw_response, triggers, state)
            self._total_used += 1
            decision.used_self_retry = True
            decision.final_response = retried
            triggers_after = detect_triggers(
                raw_response=retried,
                state=state,
                history_questions=history_questions,
                last_turn=last_turn,
                made_diagnosis=made_diagnosis,
            )
            if not triggers_after:
                decision.triggers = list(triggers_after)
                return decision
            triggers = triggers_after

        # 2. PRB-guided fallback.
        if self._fallback_used < self.caps.fallback_per_episode and state.S.primitives:
            primitive = self._nearest_reliable(state.S.primitives)
            if primitive is not None:
                question = primitive.next_inquiry_rationale or "Could you tell me more about your symptoms?"
                # Strip the boilerplate ``Asking "..." helps clarify ...`` if any.
                fallback_q = self._extract_question(question, primitive)
                think = (
                    "Original output triggered: "
                    + ", ".join(t.kind.value for t in triggers)
                    + ". Falling back to PRB-guided structured inquiry."
                )
                decision.final_response = render_stage_ii(think, fallback_q)
                decision.used_fallback = True
                decision.fallback_primitive = primitive
                self._fallback_used += 1
                self._total_used += 1
                return decision

        # 3. Both retry and fallback exhausted.
        raise RectificationFailure(
            "Self-retry and PRB fallback both unavailable; "
            f"triggers={[t.kind.value for t in triggers]}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_reliable(primitives: List[SupportPrimitive]) -> Optional[SupportPrimitive]:
        ranked = sorted(
            (p for p in primitives if p.passes_gate(threshold=0.0)),
            key=lambda p: p.similarity, reverse=True,
        )
        return ranked[0] if ranked else None

    @staticmethod
    def _extract_question(rationale: str, primitive: SupportPrimitive) -> str:
        # Try to find a quoted question inside the rationale.
        import re

        m = re.search(r'"([^"]+\?)"', rationale)
        if m:
            return m.group(1)
        if primitive.missing_check:
            return f"Could you tell me about {primitive.missing_check.strip()}?"
        return "Could you tell me more about your current symptoms?"


__all__ = [
    "Rectifier",
    "RectificationCaps",
    "RectificationDecision",
    "RectificationFailure",
]
