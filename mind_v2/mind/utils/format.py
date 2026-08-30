"""Common helpers for the two-stage turn format used by MIND (§3.2).

Stage I:  ``<think>...</think> <rag_query>...</rag_query>``
Stage II: ``<think>...</think> <answer>...</answer>``

The ``answer`` block carries either the next inquiry or a final diagnosis
formatted as ``Diagnosis: ...; Recommendation: ...``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RAG_QUERY_RE = re.compile(r"<rag_query>(.*?)</rag_query>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_DIAG_RE = re.compile(
    r"Diagnosis\s*[:：]\s*(?P<diag>.*?)(?=Recommendation\s*[:：]|$)",
    re.DOTALL | re.IGNORECASE,
)
_REC_RE = re.compile(
    r"Recommendation\s*[:：]\s*(?P<rec>.*?)$", re.DOTALL | re.IGNORECASE
)


@dataclass
class StageIOutput:
    think: str
    rag_query: str
    raw: str

    @property
    def is_valid(self) -> bool:
        return bool(self.rag_query.strip())


@dataclass
class StageIIOutput:
    think: str
    answer: str
    raw: str

    @property
    def is_valid(self) -> bool:
        return bool(self.think.strip()) and bool(self.answer.strip())

    def is_diagnosis(self) -> bool:
        return bool(_DIAG_RE.search(self.answer))

    def parse_diagnosis(self) -> Optional[tuple[str, str]]:
        d = _DIAG_RE.search(self.answer)
        r = _REC_RE.search(self.answer)
        if d is None:
            return None
        diag = d.group("diag").strip()
        rec = r.group("rec").strip() if r else ""
        return diag, rec


def parse_stage_i(text: str) -> StageIOutput:
    think = _THINK_RE.search(text)
    q = _RAG_QUERY_RE.search(text)
    return StageIOutput(
        think=think.group(1).strip() if think else "",
        rag_query=q.group(1).strip() if q else "",
        raw=text,
    )


def parse_stage_ii(text: str) -> StageIIOutput:
    think = _THINK_RE.search(text)
    ans = _ANSWER_RE.search(text)
    return StageIIOutput(
        think=think.group(1).strip() if think else "",
        answer=ans.group(1).strip() if ans else "",
        raw=text,
    )


def render_stage_i(think: str, rag_query: str) -> str:
    return f"<think>{think}</think>\n<rag_query>{rag_query}</rag_query>"


def render_stage_ii(think: str, answer: str) -> str:
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


__all__ = [
    "StageIOutput",
    "StageIIOutput",
    "parse_stage_i",
    "parse_stage_ii",
    "render_stage_i",
    "render_stage_ii",
]
