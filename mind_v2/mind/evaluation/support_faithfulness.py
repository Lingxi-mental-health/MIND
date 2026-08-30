"""Support faithfulness evaluation (FC / SG / PF, Table 4 + Appendix E).

Uses a configurable LLM judge.  Default: DeepSeek-V3.2; Appendix E repeats
with GPT-4o.  The judge prompt is :data:`mind.utils.prompts.FAITHFULNESS_JUDGE_PROMPT`.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..utils.llm_client import LLMClient
from ..utils.prompts import FAITHFULNESS_JUDGE_PROMPT


@dataclass
class FaithfulnessScores:
    fc: float
    sg: float
    pf: float

    @property
    def avg(self) -> float:
        return (self.fc + self.sg + self.pf) / 3.0


_FC_RE = re.compile(r"\(?fc\)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)
_SG_RE = re.compile(r"\(?sg\)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)
_PF_RE = re.compile(r"\(?pf\)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)


def judge_turn(
    *,
    llm: LLMClient,
    dialogue: str,
    retrieved_supports: str,
    doctor_turn: str,
) -> FaithfulnessScores:
    user = (
        f"DIALOGUE:\n{dialogue}\n\n"
        f"RETRIEVED_SUPPORTS:\n{retrieved_supports}\n\n"
        f"DOCTOR_TURN:\n{doctor_turn}\n"
    )
    raw = llm.chat(
        system=FAITHFULNESS_JUDGE_PROMPT, user=user, max_tokens=256, temperature=0.0
    )
    return _parse(raw)


def _parse(raw: str) -> FaithfulnessScores:
    def _grab(p: re.Pattern[str]) -> float:
        m = p.search(raw)
        return float(m.group(1)) if m else 0.0
    return FaithfulnessScores(fc=_grab(_FC_RE), sg=_grab(_SG_RE), pf=_grab(_PF_RE))


def aggregate(scores: Iterable[FaithfulnessScores]) -> FaithfulnessScores:
    fcs, sgs, pfs = zip(*[(s.fc, s.sg, s.pf) for s in scores]) if scores else ([], [], [])
    if not fcs:
        return FaithfulnessScores(0.0, 0.0, 0.0)
    return FaithfulnessScores(
        fc=statistics.fmean(fcs),
        sg=statistics.fmean(sgs),
        pf=statistics.fmean(pfs),
    )


__all__ = ["FaithfulnessScores", "judge_turn", "aggregate"]
