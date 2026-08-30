"""Field-masking analysis (Appendix G).

For a trained policy and a test trajectory, mask each of
:math:`\\mathcal{O}/\\mathcal{M}/\\mathcal{D}/\\mathcal{S}/\\mathcal{R}` at
inference time and re-run the episode.  Report differentiated degradation
patterns:

* masking ``M`` should cause the largest MCR drop;
* masking ``O`` should cause the largest accuracy drop;
* masking ``S`` should cause the largest faithfulness drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

from ..evidence_state.state import EvidenceState


FIELDS = ("O", "M", "D", "S", "R")


@dataclass
class FieldMaskingResult:
    field: str
    accuracy: float
    macro_f1: float
    mcr: float
    support_faithfulness: float


def run_field_masking(
    *,
    rollout_fn: Callable[[List[str]], Dict[str, float]],
    fields: Iterable[str] = FIELDS,
) -> List[FieldMaskingResult]:
    """Run the policy under each field-mask configuration.

    Parameters
    ----------
    rollout_fn:
        Callable that receives the masked-fields list and returns a metrics
        dict with keys ``acc``, ``macro_f1``, ``mcr``, ``faithfulness``.
    """

    results: List[FieldMaskingResult] = []
    # Baseline: no masking.
    base = rollout_fn([])
    results.append(FieldMaskingResult(
        field="-", accuracy=base["acc"], macro_f1=base["macro_f1"],
        mcr=base["mcr"], support_faithfulness=base["faithfulness"],
    ))
    for f in fields:
        m = rollout_fn([f])
        results.append(FieldMaskingResult(
            field=f, accuracy=m["acc"], macro_f1=m["macro_f1"],
            mcr=m["mcr"], support_faithfulness=m["faithfulness"],
        ))
    return results


__all__ = ["FieldMaskingResult", "run_field_masking", "FIELDS"]
