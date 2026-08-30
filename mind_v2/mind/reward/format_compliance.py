"""Format-compliance and operational penalties (§3.2.3, §3.2.4).

The penalties below are *signed* — they are added to the per-turn reward as
negative numbers when triggered.  Triggers are aligned with Table 13.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from ..utils.format import parse_stage_i, parse_stage_ii


@dataclass
class FormatPenaltyConfig:
    weight_format: float = 0.1          # malformed output / extra content
    weight_repetition: float = 0.1      # semantic duplicate inquiry
    weight_budget: float = 0.1          # ignoring remaining budget
    weight_unsafe: float = 0.1          # unsafe / off-track response


@dataclass
class FormatPenalty:
    cfg: FormatPenaltyConfig = field(default_factory=FormatPenaltyConfig)

    # ------------------------------------------------------------------
    def stage_i_compliance(self, raw: str) -> float:
        out = parse_stage_i(raw)
        if not out.is_valid:
            return -self.cfg.weight_format
        # No content after </rag_query>.
        if re.search(r"</rag_query>(.+)$", raw, re.DOTALL):
            tail = re.search(r"</rag_query>(.+)$", raw, re.DOTALL).group(1).strip()
            if tail:
                return -self.cfg.weight_format
        return 0.0

    def stage_ii_compliance(self, raw: str) -> float:
        out = parse_stage_ii(raw)
        if not out.is_valid:
            return -self.cfg.weight_format
        # Strict tail check: nothing after </answer>.
        if re.search(r"</answer>(.+)$", raw, re.DOTALL):
            tail = re.search(r"</answer>(.+)$", raw, re.DOTALL).group(1).strip()
            if tail:
                return -self.cfg.weight_format
        return 0.0

    # ------------------------------------------------------------------
    def repetition_penalty(self, current_question: str, history: Iterable[str]) -> float:
        from .repetition import is_semantic_duplicate

        for past in history:
            if is_semantic_duplicate(current_question, past):
                return -self.cfg.weight_repetition
        return 0.0

    def budget_penalty(self, *, last_turn: bool, made_diagnosis: bool) -> float:
        if last_turn and not made_diagnosis:
            return -self.cfg.weight_budget
        return 0.0

    def unsafe_penalty(self, response: str, *, off_track_keywords: List[str]) -> float:
        low = response.lower()
        if any(k in low for k in off_track_keywords):
            return -self.cfg.weight_unsafe
        return 0.0


__all__ = ["FormatPenalty", "FormatPenaltyConfig"]
