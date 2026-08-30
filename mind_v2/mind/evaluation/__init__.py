"""Evaluation sub-package."""
from .label_mapping import CATEGORIES, canonical_label
from .mcr import MCRStats, aggregate_mcr, compute_mcr
from .support_faithfulness import (
    FaithfulnessScores, judge_turn, aggregate as aggregate_faithfulness,
)
from .field_masking import FIELDS, FieldMaskingResult, run_field_masking
from .state_construction_quality import AuditResult, aggregate_audit
from .parameter_analysis import (
    SweepPoint, sweep_horizon, sweep_reliability, sweep_top_k,
)
from .mdd5k import (
    DiagnosticResult,
    NextQuestionScore,
    macro_f1,
    run_next_question_eval,
    run_public_diagnostic,
)
from .statistics import BootstrapResult, bootstrap_metric, paired_bootstrap_diff

__all__ = [
    "CATEGORIES", "canonical_label",
    "MCRStats", "aggregate_mcr", "compute_mcr",
    "FaithfulnessScores", "judge_turn", "aggregate_faithfulness",
    "FIELDS", "FieldMaskingResult", "run_field_masking",
    "AuditResult", "aggregate_audit",
    "SweepPoint", "sweep_horizon", "sweep_reliability", "sweep_top_k",
    "DiagnosticResult", "NextQuestionScore",
    "macro_f1", "run_next_question_eval", "run_public_diagnostic",
    "BootstrapResult", "bootstrap_metric", "paired_bootstrap_diff",
]
