"""Terminal screening reward :math:`r^{\\rm term}` (§3.2.3, Table 11).

For the four-way ICD-10-style screening label space (Depression / Anxiety /
Mixed / Other), the reward is a hard ``1[ d_p == d* ]`` multiplied by 5.0.
A small partial-credit bonus is provided for free-form recommendation text
that overlaps with the gold recommendation; this matches the original
``MIND`` codebase but is *not* counted against the headline metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..evaluation.label_mapping import canonical_label


@dataclass
class TerminalReward:
    weight: float = 5.0
    recommendation_weight: float = 0.0  # paper sets 0; MIND legacy uses 5.0

    def __call__(
        self,
        predicted_diagnosis: str,
        gold_diagnosis: str,
        *,
        predicted_recommendation: str = "",
        gold_recommendation: str = "",
        recommendation_score_fn=None,
    ) -> float:
        pred = canonical_label(predicted_diagnosis)
        gold = canonical_label(gold_diagnosis)
        r = self.weight * float(pred == gold)
        if (
            self.recommendation_weight > 0.0
            and recommendation_score_fn is not None
            and gold_recommendation
        ):
            r += self.recommendation_weight * float(
                recommendation_score_fn(predicted_recommendation, gold_recommendation)
            )
        return r


__all__ = ["TerminalReward"]
