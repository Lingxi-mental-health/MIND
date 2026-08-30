"""Typed evidence-state interface :math:`\\mathcal{E}_t`.

Mirrors the formalisation in §3 of the paper:

.. math::

    \\mathcal{E}_t = (\\mathcal{O}_t, \\mathcal{M}_t, \\mathcal{D}_t,
                     \\mathcal{S}_t, \\mathcal{R}_t)

where the five fields are typed decision variables shared across the policy,
the process-reward judge and the rectification operator.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import HARD_ISSUE_FLAGS, RETRIEVAL_STATE_FIELDS


@dataclass
class ObservedEvidence:
    """Field :math:`\\mathcal{O}_t`: patient-reported confirmed / negated facts."""

    confirmed: Dict[str, str] = field(default_factory=dict)
    negated: Dict[str, str] = field(default_factory=dict)

    def num_known_fields(self) -> int:
        return len(self.confirmed) + len(self.negated)

    def render(self) -> str:
        if not self.confirmed and not self.negated:
            return "(none)"
        lines = []
        for k, v in self.confirmed.items():
            lines.append(f"- [confirmed] {k}: {v}")
        for k, v in self.negated.items():
            lines.append(f"- [negated]   {k}: {v}")
        return "\n".join(lines)


@dataclass
class MissingChecks:
    """Field :math:`\\mathcal{M}_t`: criterion / exclusion checks left unknown."""

    fields: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fields)

    def remove(self, name: str) -> bool:
        if name in self.fields:
            self.fields.remove(name)
            return True
        return False

    def render(self) -> str:
        if not self.fields:
            return "(no missing checks)"
        return "; ".join(f"<{name}>" for name in self.fields)


@dataclass
class DifferentialHypotheses:
    """Field :math:`\\mathcal{D}_t`: active competing screening categories."""

    active: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)

    def render(self) -> str:
        active = ", ".join(self.active) if self.active else "(none)"
        eliminated = ", ".join(self.eliminated) if self.eliminated else "(none)"
        return f"active={active}; eliminated={eliminated}"


@dataclass
class SupportPrimitive:
    """A single PRB support primitive :math:`e_i=(q_i,c_i,o_i,g_i,x_i,u_i,\\rho_i)`.

    ``rho`` stores the reliability score (1--5) and hard-issue flags
    (``H1``/``H2``/``H3``).  The judge prompt is documented in §3.1.3.
    """

    q: str = ""
    criterion: str = ""
    observed_evidence: str = ""
    missing_check: str = ""
    exclusion_cue: str = ""
    next_inquiry_rationale: str = ""
    reliability_score: float = 0.0
    hard_issues: Dict[str, bool] = field(
        default_factory=lambda: {flag: False for flag in HARD_ISSUE_FLAGS}
    )
    similarity: float = 0.0

    def passes_gate(self, threshold: float) -> bool:
        if any(self.hard_issues.get(f, False) for f in HARD_ISSUE_FLAGS):
            return False
        return self.reliability_score >= threshold

    def render(self) -> str:
        return (
            f"(criterion) {self.criterion}\n"
            f"(observed) {self.observed_evidence}\n"
            f"(missing)  {self.missing_check}\n"
            f"(exclusion) {self.exclusion_cue}\n"
            f"(next-inquiry rationale) {self.next_inquiry_rationale}"
        )


@dataclass
class SupportSet:
    """Field :math:`\\mathcal{S}_t`: reliability-gated support primitives."""

    primitives: List[SupportPrimitive] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.primitives)

    def render(self, max_chars_each: int = 600) -> str:
        if not self.primitives:
            return "(null support)"
        rendered = []
        for i, p in enumerate(self.primitives, 1):
            block = p.render()
            if len(block) > max_chars_each:
                block = block[: max_chars_each] + "..."
            rendered.append(f"[support {i}] sim={p.similarity:.3f}\n{block}")
        return "\n\n".join(rendered)


@dataclass
class ReliabilityMetadata:
    """Field :math:`\\mathcal{R}_t`: scores + hard-issue flags per support."""

    scores: List[float] = field(default_factory=list)
    hard_issues: List[Dict[str, bool]] = field(default_factory=list)
    threshold: float = 0.0
    gated_in: List[bool] = field(default_factory=list)

    def render(self) -> str:
        if not self.scores:
            return "(no retrieval)"
        items = []
        for i, (s, gi, flags) in enumerate(
            zip(self.scores, self.gated_in, self.hard_issues), 1
        ):
            tag = "ADMIT" if gi else "REJECT"
            issue_str = ",".join(k for k, v in flags.items() if v) or "-"
            items.append(f"[r{i}] {tag} score={s:.2f} hard-issues={issue_str}")
        return "; ".join(items)


@dataclass
class EvidenceState:
    """:math:`\\mathcal{E}_t` shared by the policy, reward and fallback."""

    O: ObservedEvidence = field(default_factory=ObservedEvidence)
    M: MissingChecks = field(default_factory=MissingChecks)
    D: DifferentialHypotheses = field(default_factory=DifferentialHypotheses)
    S: SupportSet = field(default_factory=SupportSet)
    R: ReliabilityMetadata = field(default_factory=ReliabilityMetadata)

    # ------------------------------------------------------------------
    # Field-masking utilities (used by Appendix G).
    # ------------------------------------------------------------------
    def mask(self, *fields_to_mask: str) -> "EvidenceState":
        """Return a copy of the state with selected fields blanked.

        Used by ``mind.evaluation.field_masking`` and the controlled
        comparison in Table 5 to test the causal role of each field.
        """

        cp = EvidenceState(
            O=ObservedEvidence(dict(self.O.confirmed), dict(self.O.negated)),
            M=MissingChecks(list(self.M.fields)),
            D=DifferentialHypotheses(list(self.D.active), list(self.D.eliminated)),
            S=SupportSet([_clone_support(p) for p in self.S.primitives]),
            R=ReliabilityMetadata(
                list(self.R.scores),
                [dict(h) for h in self.R.hard_issues],
                self.R.threshold,
                list(self.R.gated_in),
            ),
        )
        for f in fields_to_mask:
            f = f.upper()
            if f == "O":
                cp.O = ObservedEvidence()
            elif f == "M":
                cp.M = MissingChecks()
            elif f == "D":
                cp.D = DifferentialHypotheses()
            elif f == "S":
                cp.S = SupportSet()
            elif f == "R":
                cp.R = ReliabilityMetadata()
            else:
                raise ValueError(f"Unknown evidence-state field: {f}")
        return cp

    # ------------------------------------------------------------------
    # Verbalisation used by the policy / process-reward judge / fallback.
    # ------------------------------------------------------------------
    def render(self) -> str:
        """Render :math:`\\mathcal{E}_t` as a structured prompt block."""

        return (
            "=== Evidence State E_t ===\n"
            "[O] Observed evidence:\n" + self.O.render() + "\n\n"
            "[M] Missing/unclear checks: " + self.M.render() + "\n\n"
            "[D] Active differentials: " + self.D.render() + "\n\n"
            "[S] PRB supports (reliability-gated):\n" + self.S.render() + "\n\n"
            "[R] Reliability metadata: " + self.R.render() + "\n"
            "=========================="
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clone_support(p: SupportPrimitive) -> SupportPrimitive:
    return SupportPrimitive(
        q=p.q,
        criterion=p.criterion,
        observed_evidence=p.observed_evidence,
        missing_check=p.missing_check,
        exclusion_cue=p.exclusion_cue,
        next_inquiry_rationale=p.next_inquiry_rationale,
        reliability_score=p.reliability_score,
        hard_issues=dict(p.hard_issues),
        similarity=p.similarity,
    )


__all__ = [
    "RETRIEVAL_STATE_FIELDS",
    "ObservedEvidence",
    "MissingChecks",
    "DifferentialHypotheses",
    "SupportPrimitive",
    "SupportSet",
    "ReliabilityMetadata",
    "EvidenceState",
]
