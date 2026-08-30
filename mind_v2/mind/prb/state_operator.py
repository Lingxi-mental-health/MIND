"""State-construction operator :math:`g_{\\rm PRB}(q_t)\\to(\\mathcal{S}_t,\\mathcal{R}_t)`.

This is the operator that distinguishes MIND from Standard RAG (§3 ``Why a
typed state rather than retrieved text``):

* It receives a *retrieval state* :math:`q_t`, not raw history.
* It populates the **typed** fields :math:`\\mathcal{S}_t,\\mathcal{R}_t` of
  the evidence-state interface, not free-form text.
* It applies reliability gating, propagating only primitives with
  ``score >= τ_rel`` and no hard-issue flags.
* The same primitives are reused across action selection, reward computation,
  and fallback (see §3 ``Operational semantics``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .retriever import PRBRetriever
from ..evidence_state.state import (
    EvidenceState,
    ReliabilityMetadata,
    SupportPrimitive,
    SupportSet,
)


@dataclass
class PRBStateOperator:
    """Bind a retriever and a reliability threshold."""

    retriever: PRBRetriever
    tau_rel: float = 0.2  # see Table 13 (default 0.2)
    score_min: float = 1.0  # reliability judge scale: 1--5
    score_max: float = 5.0
    top_k: int = 4

    def _normalise_score(self, score: float) -> float:
        """Min--max normalise a judge score to ``[0, 1]``.

        Values outside the configured judge range are clipped so that a
        malformed or over-range score cannot bypass reliability gating.
        """

        span = self.score_max - self.score_min
        if span <= 0:
            raise ValueError("score_max must be greater than score_min")
        return max(0.0, min(1.0, (score - self.score_min) / span))

    # ------------------------------------------------------------------
    # Core operator.
    # ------------------------------------------------------------------
    def __call__(self, q_t: str) -> Tuple[SupportSet, ReliabilityMetadata]:
        candidates: List[SupportPrimitive] = self.retriever.retrieve(q_t, top_k=self.top_k)

        gated_in: List[bool] = []
        scores: List[float] = []
        hard_issues_list = []
        admitted: List[SupportPrimitive] = []

        for prim in candidates:
            normalised = self._normalise_score(prim.reliability_score)
            scores.append(normalised)
            hard_issues_list.append(dict(prim.hard_issues))
            ok = (
                normalised >= self.tau_rel
                and not any(prim.hard_issues.get(k, False) for k in ("H1", "H2", "H3"))
            )
            gated_in.append(ok)
            if ok:
                admitted.append(prim)

        return (
            SupportSet(primitives=admitted),
            ReliabilityMetadata(
                scores=scores,
                hard_issues=hard_issues_list,
                threshold=self.tau_rel,
                gated_in=gated_in,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience: mutate an existing :class:`EvidenceState` in place.
    # ------------------------------------------------------------------
    def populate(self, state: EvidenceState, q_t: str) -> EvidenceState:
        S, R = self(q_t)
        state.S = S
        state.R = R
        return state


__all__ = ["PRBStateOperator"]
