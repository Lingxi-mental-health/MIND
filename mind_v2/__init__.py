"""MIND v2 — paper-aligned reimplementation of the MIND framework.

This package mirrors the methodology and experiments described in
``latex/acl_latex.tex``.  It is structured as:

* ``mind.evidence_state`` — typed evidence-state interface
  :math:`\\mathcal{E}_t=(\\mathcal{O}_t,\\mathcal{M}_t,\\mathcal{D}_t,\\mathcal{S}_t,\\mathcal{R}_t)`.
* ``mind.prb`` — Criteria-Grounded Psychiatric Reasoning Bank
  (offline build, retrieval, reliability gating, state-construction operator).
* ``mind.reward`` — process reward (sym/diff/dec rubric), info-gain reward,
  format compliance, terminal screening reward, and the RL aggregator.
* ``mind.rectification`` — rule-based triggers and the self-retry / PRB-guided
  fallback operator.
* ``mind.env`` — the multi-turn psychiatric consultation environment with
  Stage-I (``<rag_query>``) / Stage-II (``<think><answer>``) format and the
  PsySim-Std / PsySim-Adapt patient simulators.
* ``mind.training`` — SFT and GRPO runners that wrap ``ragen``/``verl``.
* ``mind.evaluation`` — MCR, support faithfulness, field masking, state
  construction quality, parameter analysis, and external MDD-5k evaluation.
"""

__all__ = [
    "evidence_state",
    "prb",
    "reward",
    "rectification",
    "env",
    "training",
    "evaluation",
    "utils",
]

__version__ = "2.0.0"
