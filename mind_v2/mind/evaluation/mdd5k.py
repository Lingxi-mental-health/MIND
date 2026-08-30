"""External evaluation on the publicly released MDD-5k diagnostic dialogues
(§4 ``Public MDD-5k Diagnostic Evaluation``, Table 2).

Two protocols are exposed:

* :func:`run_public_diagnostic` — the policy receives the *full* released
  dialogue and predicts a four-way screening category.
* :func:`run_next_question_eval` — the policy receives a *prefix* (chief
  complaint or first :math:`K=10` exchanges) and emits one next question;
  scored by the rubric in :data:`NEXT_QUESTION_RUBRIC_PROMPT`.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from ..utils.llm_client import LLMClient
from ..utils.prompts import NEXT_QUESTION_RUBRIC_PROMPT
from .label_mapping import canonical_label, CATEGORIES


@dataclass
class DiagnosticResult:
    accuracy: float
    macro_f1: float
    confusion: Dict[str, Dict[str, int]]


def macro_f1(confusion: Dict[str, Dict[str, int]]) -> float:
    f1s = []
    for cls in CATEGORIES:
        tp = confusion.get(cls, {}).get(cls, 0)
        fp = sum(
            confusion.get(other, {}).get(cls, 0) for other in CATEGORIES if other != cls
        )
        fn = sum(
            confusion.get(cls, {}).get(other, 0) for other in CATEGORIES if other != cls
        )
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return statistics.fmean(f1s) if f1s else 0.0


def run_public_diagnostic(
    *,
    cases: Iterable[Dict[str, str]],
    predict_fn: Callable[[str], str],
) -> DiagnosticResult:
    """Diagnose each case from the full released dialogue."""

    confusion = {c: {c2: 0 for c2 in CATEGORIES} for c in CATEGORIES}
    total = 0
    correct = 0
    for case in cases:
        dialogue = case.get("dialogue", "")
        gold = canonical_label(case.get("gold_diagnosis", ""))
        pred = canonical_label(predict_fn(dialogue))
        confusion.setdefault(gold, {}).setdefault(pred, 0)
        confusion[gold][pred] = confusion[gold].get(pred, 0) + 1
        total += 1
        if pred == gold:
            correct += 1
    accuracy = correct / total if total else 0.0
    return DiagnosticResult(accuracy=accuracy, macro_f1=macro_f1(confusion), confusion=confusion)


@dataclass
class NextQuestionScore:
    relevance: float
    info: float
    safety: float
    naturalness: float


def run_next_question_eval(
    *,
    cases: Iterable[Dict[str, str]],
    next_question_fn: Callable[[str], str],
    judge: LLMClient,
) -> List[NextQuestionScore]:
    out: List[NextQuestionScore] = []
    for case in cases:
        prefix = case["prefix"]
        question = next_question_fn(prefix)
        raw = judge.chat(
            system=NEXT_QUESTION_RUBRIC_PROMPT,
            user=f"DIALOGUE_PREFIX:\n{prefix}\n\nCANDIDATE_QUESTION:\n{question}",
            temperature=0.0,
            max_tokens=256,
        )
        out.append(_parse_next_question(raw))
    return out


def _parse_next_question(raw: str) -> NextQuestionScore:
    import re

    def grab(p: str) -> float:
        m = re.search(p + r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*/\s*5", raw, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0
    return NextQuestionScore(
        relevance=grab(r"\(?rel\)?"),
        info=grab(r"\(?info\)?"),
        safety=grab(r"\(?safety\)?"),
        naturalness=grab(r"\(?nat\)?"),
    )


__all__ = [
    "DiagnosticResult",
    "macro_f1",
    "run_public_diagnostic",
    "NextQuestionScore",
    "run_next_question_eval",
]
