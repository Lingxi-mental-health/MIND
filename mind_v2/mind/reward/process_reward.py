"""Process reward (§3.2.3, Table 12).

Three rubric dimensions on a 0/1/2 scale, normalised to [0, 1] before
aggregation::

    r_t^proc = (1/|C|) Σ_{c∈C} S_t^c / S_max,  C = {sym, diff, dec}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..evidence_state.state import EvidenceState
from ..utils.llm_client import LLMClient
from ..utils.prompts import PROCESS_REWARD_PROMPT


_S_MAX = 2.0
_DIMS = ("sym", "diff", "dec")


@dataclass
class ProcessRewardScores:
    sym: float
    diff: float
    dec: float

    def normalised_mean(self) -> float:
        return (self.sym + self.diff + self.dec) / (3.0 * _S_MAX)

    def as_dict(self) -> Dict[str, float]:
        return {"sym": self.sym, "diff": self.diff, "dec": self.dec}


class ProcessRewardJudge:
    """LLM-based rubric scorer."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def score(
        self, *, dialogue: str, evidence_state: EvidenceState, reasoning_trace: str,
    ) -> ProcessRewardScores:
        user = (
            f"DIALOGUE_HISTORY:\n{dialogue}\n\n"
            f"{evidence_state.render()}\n\n"
            f"REASONING_TRACE:\n{reasoning_trace}\n"
        )
        raw = self.llm.chat(
            system=PROCESS_REWARD_PROMPT, user=user, max_tokens=256, temperature=0.0
        )
        return _parse_rubric(raw)


_SYM_RE = re.compile(r"\(?symptom\)?\s*\(?\s*(\d)\s*/\s*2", re.IGNORECASE)
_DIFF_RE = re.compile(r"\(?differential\)?\s*\(?\s*(\d)\s*/\s*2", re.IGNORECASE)
_DEC_RE = re.compile(r"\(?decision\)?\s*\(?\s*(\d)\s*/\s*2", re.IGNORECASE)


def _parse_rubric(raw: str) -> ProcessRewardScores:
    def _grab(p: re.Pattern[str]) -> float:
        m = p.search(raw)
        return float(m.group(1)) if m else 0.0
    return ProcessRewardScores(
        sym=_grab(_SYM_RE), diff=_grab(_DIFF_RE), dec=_grab(_DEC_RE)
    )


__all__ = ["ProcessRewardJudge", "ProcessRewardScores"]
