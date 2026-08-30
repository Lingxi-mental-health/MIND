"""Parameter sensitivity sweeps (Appendix J + Figure 6 + Figure 7 + Figure 8).

* ``sweep_top_k``      — retrieval depth ``k`` ∈ {2, 3, 4, 6, 8} (Figure 6).
* ``sweep_horizon``    — turn budget ``L`` ∈ {4, 6, 8, 10, 12} (Figure 7).
* ``sweep_threshold``  — reliability threshold ``τ_rel`` ∈
  {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40} (Figure 8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass
class SweepPoint:
    name: str
    value: float
    metrics: Dict[str, float]


def _sweep(
    name: str,
    values: Sequence[float],
    rollout_fn: Callable[[float], Dict[str, float]],
) -> List[SweepPoint]:
    return [SweepPoint(name=name, value=v, metrics=rollout_fn(v)) for v in values]


def sweep_top_k(rollout_fn) -> List[SweepPoint]:
    return _sweep("top_k", (2, 3, 4, 6, 8), rollout_fn)


def sweep_horizon(rollout_fn) -> List[SweepPoint]:
    return _sweep("L", (4, 6, 8, 10, 12), rollout_fn)


def sweep_reliability(rollout_fn) -> List[SweepPoint]:
    return _sweep(
        "tau_rel",
        (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
        rollout_fn,
    )


__all__ = ["SweepPoint", "sweep_top_k", "sweep_horizon", "sweep_reliability"]
