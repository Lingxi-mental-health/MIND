"""Aggregator that combines all reward signals defined in §3.2.3.

.. math::

    r_t = r_t^{\\rm proc} + r_t^{\\rm gain} + r_t^{\\rm fmt} + r_t^{\\rm pen}, \\\\
    R   = \\sum_t \\alpha_t r_t + \\beta r^{\\rm term}.

The default weights match Table 11 of the paper:

* terminal screening: 5.0
* information gain:   0.005
* clinical reasoning: 0.01
* format compliance:  0.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .format_compliance import FormatPenalty, FormatPenaltyConfig
from .info_gain import InfoGainReward
from .process_reward import ProcessRewardJudge, ProcessRewardScores
from .terminal_reward import TerminalReward
from ..evidence_state.state import EvidenceState


@dataclass
class RewardWeights:
    terminal: float = 5.0
    info_gain: float = 0.005
    process: float = 0.01
    format: float = 0.1
    retrieval: float = 0.0  # optional retrieval-shaping weight; off by default


@dataclass
class TurnRewardResult:
    total: float
    process: float
    info_gain: float
    format: float
    retrieval: float
    breakdown: Dict[str, float]


@dataclass
class MindRewardManager:
    """End-to-end reward bookkeeping for a single MIND episode."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    info_gain_reward: InfoGainReward = field(default_factory=InfoGainReward)
    format_penalty: FormatPenalty = field(default_factory=FormatPenalty)
    terminal_reward: TerminalReward = field(default_factory=TerminalReward)
    process_judge: Optional[ProcessRewardJudge] = None  # may be None for tests

    # ------------------------------------------------------------------
    # Per-turn reward.
    # ------------------------------------------------------------------
    def turn_reward(
        self,
        *,
        prev_state: EvidenceState,
        curr_state: EvidenceState,
        process_scores: Optional[ProcessRewardScores],
        format_signals: Dict[str, float],
        retrieval_score: float = 0.0,
    ) -> TurnRewardResult:
        process_value = (
            self.weights.process * process_scores.normalised_mean()
            if process_scores is not None
            else 0.0
        )
        info_gain_value = self.info_gain_reward(prev_state, curr_state)
        format_value = self.weights.format * sum(format_signals.values())
        retrieval_value = self.weights.retrieval * retrieval_score

        total = process_value + info_gain_value + format_value + retrieval_value
        return TurnRewardResult(
            total=total,
            process=process_value,
            info_gain=info_gain_value,
            format=format_value,
            retrieval=retrieval_value,
            breakdown={
                "process": process_value,
                "info_gain": info_gain_value,
                "format": format_value,
                "retrieval": retrieval_value,
                **{f"format::{k}": v for k, v in format_signals.items()},
            },
        )

    # ------------------------------------------------------------------
    # Terminal reward.
    # ------------------------------------------------------------------
    def terminal(
        self,
        *,
        predicted_diagnosis: str,
        gold_diagnosis: str,
        predicted_recommendation: str = "",
        gold_recommendation: str = "",
        recommendation_score_fn=None,
    ) -> float:
        return self.terminal_reward(
            predicted_diagnosis,
            gold_diagnosis,
            predicted_recommendation=predicted_recommendation,
            gold_recommendation=gold_recommendation,
            recommendation_score_fn=recommendation_score_fn,
        )


__all__ = [
    "RewardWeights",
    "TurnRewardResult",
    "MindRewardManager",
]
