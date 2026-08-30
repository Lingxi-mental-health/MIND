"""PRB entry schema (§3.1, Eq.\\,(5)).

Each entry is a typed primitive::

    e_i = (q_i, c_i, o_i, g_i, x_i, u_i, ρ_i)

The constructor LLM is *only* an offline preprocessor: at inference time the
policy does not query it.  All fields are stored as plain text so that the
PRB can be serialised to JSONL without any model dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class PRBEntry:
    """A single PRB primitive."""

    entry_id: str
    q: str                    # clinical retrieval state
    criterion: str            # c_i
    observed_evidence: str    # o_i
    missing_check: str        # g_i
    exclusion_cue: str        # x_i
    next_inquiry_rationale: str  # u_i
    next_question: str = ""   # canonical reference inquiry text
    reliability_score: float = 0.0  # ρ_i (1..5)
    hard_issues: Dict[str, bool] = field(
        default_factory=lambda: {"H1": False, "H2": False, "H3": False}
    )
    # Bookkeeping required for leakage prevention (Appendix A).
    source_split: str = "train"
    source_emr_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PRBEntry":
        flags = data.get("hard_issues") or {"H1": False, "H2": False, "H3": False}
        return cls(
            entry_id=str(data["entry_id"]),
            q=data.get("q", ""),
            criterion=data.get("criterion", ""),
            observed_evidence=data.get("observed_evidence", ""),
            missing_check=data.get("missing_check", ""),
            exclusion_cue=data.get("exclusion_cue", ""),
            next_inquiry_rationale=data.get("next_inquiry_rationale", ""),
            next_question=data.get("next_question", ""),
            reliability_score=float(data.get("reliability_score", 0.0)),
            hard_issues={k: bool(flags.get(k, False)) for k in ("H1", "H2", "H3")},
            source_split=data.get("source_split", "train"),
            source_emr_id=str(data.get("source_emr_id", "")),
        )


__all__ = ["PRBEntry"]
