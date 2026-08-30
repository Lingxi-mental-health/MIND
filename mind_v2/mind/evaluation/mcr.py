"""Missing-Check Resolution Rate (MCR), Appendix D.

For each inquiry turn, we extract missing checks from :math:`q_t`, count the
turn as resolved if at least one check moves from ``unclear/not mentioned``
to observed/explicitly negated in :math:`q_{t+1}`, and report:

.. math::

    {\\rm MCR} = \\frac{1}{N}\\sum_t \\mathbb{1}[|\\Delta_t|>0].

We also report a repetition rate via Jaccard semantic-duplicate detection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..reward.repetition import has_recent_duplicate


@dataclass
class MCRStats:
    mcr: float
    repetition: float
    inquiry_turns: int
    resolved_turns: int


def compute_mcr(trajectory: Sequence[dict]) -> MCRStats:
    """Compute MCR / repetition for a single trajectory.

    Each element of ``trajectory`` must expose:

    * ``state_before.missing_fields`` — list of strings.
    * ``state_after.missing_fields``  — list of strings.
    * ``question`` — string.
    * ``is_diagnosis`` — bool.
    """

    inquiry_turns = 0
    resolved_turns = 0
    repetitions = 0
    history_questions: list[str] = []

    for step in trajectory:
        if step.get("is_diagnosis", False):
            continue
        prev = set(step.get("state_before", {}).get("missing_fields", []))
        curr = set(step.get("state_after", {}).get("missing_fields", []))
        if not prev:
            continue
        inquiry_turns += 1
        if (prev - curr):
            resolved_turns += 1
        q = step.get("question", "")
        if has_recent_duplicate(q, history_questions):
            repetitions += 1
        history_questions.append(q)

    if inquiry_turns == 0:
        return MCRStats(mcr=0.0, repetition=0.0, inquiry_turns=0, resolved_turns=0)
    return MCRStats(
        mcr=resolved_turns / inquiry_turns,
        repetition=repetitions / inquiry_turns,
        inquiry_turns=inquiry_turns,
        resolved_turns=resolved_turns,
    )


def aggregate_mcr(trajectories: Iterable[Sequence[dict]]) -> MCRStats:
    n = 0
    sum_mcr = 0.0
    sum_rep = 0.0
    inq = 0
    res = 0
    for traj in trajectories:
        s = compute_mcr(traj)
        if s.inquiry_turns == 0:
            continue
        n += 1
        sum_mcr += s.mcr
        sum_rep += s.repetition
        inq += s.inquiry_turns
        res += s.resolved_turns
    if n == 0:
        return MCRStats(mcr=0.0, repetition=0.0, inquiry_turns=0, resolved_turns=0)
    return MCRStats(
        mcr=sum_mcr / n, repetition=sum_rep / n, inquiry_turns=inq, resolved_turns=res
    )


__all__ = ["MCRStats", "compute_mcr", "aggregate_mcr"]
