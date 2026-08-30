"""Manual inspection support for §3 ``Evidence-state interface`` (91.8\\%).

Implements the audit protocol used in Appendix H:

1. Sample 100 turns uniformly from the simulator log.
2. For each turn, an annotator rates each of the five fields on a 0/1/2
   scale: ``0 = wrong``, ``1 = partial``, ``2 = correct``.
3. Report partial-or-better ratio per field and overall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


_FIELDS = ("O", "M", "D", "S", "R")


@dataclass
class AuditResult:
    per_field_partial_or_better: Dict[str, float]
    overall_partial_or_better: float
    n_turns: int


def aggregate_audit(rows: Iterable[Dict[str, int]]) -> AuditResult:
    """Aggregate annotator scores into a per-field acceptance rate."""

    counts: Dict[str, int] = {f: 0 for f in _FIELDS}
    totals: Dict[str, int] = {f: 0 for f in _FIELDS}
    overall_total = 0
    overall_pass = 0
    for row in rows:
        for f in _FIELDS:
            v = int(row.get(f, 0))
            totals[f] += 1
            overall_total += 1
            if v >= 1:
                counts[f] += 1
                overall_pass += 1
    per_field = {
        f: (counts[f] / totals[f]) if totals[f] else 0.0 for f in _FIELDS
    }
    overall = (overall_pass / overall_total) if overall_total else 0.0
    return AuditResult(
        per_field_partial_or_better=per_field,
        overall_partial_or_better=overall,
        n_turns=totals[_FIELDS[0]],
    )


__all__ = ["AuditResult", "aggregate_audit"]
