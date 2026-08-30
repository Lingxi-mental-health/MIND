"""Statistical helpers — bootstrap CI + paired bootstrap, used in §4.

These match the ``B = 1,000`` bootstrap protocol declared in Appendix B
``Statistical Testing``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float


def bootstrap_metric(
    samples: Sequence[float],
    metric_fn: Callable[[Sequence[float]], float] = None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    metric_fn = metric_fn or (lambda xs: sum(xs) / max(1, len(xs)))
    rng = random.Random(seed)
    n = len(samples)
    base = metric_fn(samples)
    boots: List[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(metric_fn([samples[i] for i in idx]))
    boots.sort()
    lo = boots[int((1 - ci) / 2 * len(boots))]
    hi = boots[int((1 + ci) / 2 * len(boots)) - 1]
    return BootstrapResult(point=base, ci_low=lo, ci_high=hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    metric_fn: Callable[[Sequence[float]], float] = None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return ``(point_difference_a_minus_b, one_sided_p_value)``."""

    if len(a) != len(b):
        raise ValueError("a and b must have the same length (paired bootstrap)")
    metric_fn = metric_fn or (lambda xs: sum(xs) / max(1, len(xs)))
    rng = random.Random(seed)
    n = len(a)
    base = metric_fn(a) - metric_fn(b)
    cnt_le_zero = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diff = metric_fn([a[i] for i in idx]) - metric_fn([b[i] for i in idx])
        if diff <= 0.0:
            cnt_le_zero += 1
    p_value = cnt_le_zero / n_resamples
    return base, p_value


__all__ = [
    "BootstrapResult",
    "bootstrap_metric",
    "paired_bootstrap_diff",
]
