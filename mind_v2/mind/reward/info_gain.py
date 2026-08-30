"""Information-gain reward (§3 ``Operational semantics``, §3.2.3).

The reward credits a turn that reduces :math:`|\\mathcal{M}_t|`, i.e. that
moves at least one missing-check field from ``unclear``/``not mentioned`` to
``observed`` or ``explicitly negated``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..evidence_state.state import EvidenceState


@dataclass
class InfoGainReward:
    """Stateless info-gain reward computed across two consecutive states.

    Returns the *number* of resolved fields multiplied by ``lambda_gain``.
    """

    lambda_gain: float = 0.005  # see Table 11

    def __call__(self, prev: EvidenceState, curr: EvidenceState) -> float:
        prev_missing = set(prev.M.fields)
        curr_missing = set(curr.M.fields)
        resolved = prev_missing - curr_missing
        return self.lambda_gain * float(len(resolved))


def missing_check_resolved(prev: EvidenceState, curr: EvidenceState) -> int:
    """Boolean indicator used by the MCR metric (Appendix D)."""

    prev_missing = set(prev.M.fields)
    curr_missing = set(curr.M.fields)
    return int(len(prev_missing - curr_missing) > 0)


__all__ = ["InfoGainReward", "missing_check_resolved"]
