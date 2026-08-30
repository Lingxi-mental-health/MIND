"""Vector retrieval over a PRB index (BGE-large-zh-v1.5 by default).

We re-use the heavy lifting of ``MIND/ragen/env/med_dialogue/rag_retriever.py``
but expose a *typed* interface aligned with §3.1: every retrieval call returns
:class:`mind.evidence_state.SupportPrimitive` candidates that are ready to be
gated and inserted into :math:`\\mathcal{S}_t,\\mathcal{R}_t`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .entry import PRBEntry
from ..evidence_state.state import SupportPrimitive


@dataclass
class _IndexHandles:
    embeddings: np.ndarray
    entries: List[PRBEntry]


class PRBRetriever:
    """Top-:math:`k` semantic retrieval over PRB entries."""

    def __init__(
        self,
        index_dir: str,
        bge_model_path: str,
        device: str = "cpu",
        top_k: int = 4,
    ) -> None:
        self.index_dir = index_dir
        self.bge_model_path = bge_model_path
        self.device = device
        self.top_k = top_k
        self._handles: Optional[_IndexHandles] = None
        self._encoder = None  # SentenceTransformer

    # ------------------------------------------------------------------
    # Lazy loading.
    # ------------------------------------------------------------------
    def _load(self) -> _IndexHandles:
        if self._handles is not None:
            return self._handles
        emb_path = os.path.join(self.index_dir, "embeddings.npy")
        meta_path = os.path.join(self.index_dir, "entries.jsonl")
        legacy_path = os.path.join(self.index_dir, "entries.pkl")

        # allow_pickle=False：.npy 只承载纯数值数组，不应含 Python 对象。
        emb = np.load(emb_path, allow_pickle=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-8)

        if not os.path.exists(meta_path) and os.path.exists(legacy_path):
            raise RuntimeError(
                f"Found a legacy pickle index at {legacy_path}. Loading it would "
                "deserialise arbitrary Python objects, so it is refused. Convert it "
                "with:\n"
                "  python -m mind_v2.scripts.prb.convert_legacy_index "
                f"--index_dir {self.index_dir}\n"
                "or rebuild with mind_v2.scripts.prb.build_index."
            )

        entries: List[PRBEntry] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(PRBEntry.from_dict(json.loads(line)))

        if len(entries) != emb.shape[0]:
            raise ValueError(
                f"PRB index is inconsistent: {emb.shape[0]} embeddings vs "
                f"{len(entries)} entries in {self.index_dir}"
            )

        self._handles = _IndexHandles(embeddings=emb, entries=entries)
        return self._handles

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer(
                self.bge_model_path, device=self.device, local_files_only=True
            )

    # ------------------------------------------------------------------
    # Build (called from ``scripts/prb/build_index.py``).
    # ------------------------------------------------------------------
    @staticmethod
    def build_index(
        entries: Sequence[PRBEntry],
        *,
        out_dir: str,
        bge_model_path: str,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        os.makedirs(out_dir, exist_ok=True)
        encoder = SentenceTransformer(
            bge_model_path, device=device, local_files_only=True
        )
        texts = [e.q for e in entries]
        emb = encoder.encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
        )
        np.save(os.path.join(out_dir, "embeddings.npy"), np.asarray(emb))
        # JSONL 而非 pickle：PRBEntry 全部字段都是纯文本/数值，序列化无需
        # 承担反序列化即执行代码的风险（见 mind.prb.entry 的模块说明）。
        with open(os.path.join(out_dir, "entries.jsonl"), "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Retrieve.
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SupportPrimitive]:
        """Return the top-:math:`k` candidate :class:`SupportPrimitive`s.

        Reliability gating is *not* applied here — that is the job of
        :class:`mind.prb.state_operator.PRBStateOperator`.
        """

        if top_k is None:
            top_k = self.top_k
        h = self._load()
        self._ensure_encoder()

        q_emb = self._encoder.encode(query, normalize_embeddings=True)
        sims = h.embeddings @ q_emb.reshape(-1)
        if top_k < len(sims):
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        else:
            top_idx = np.argsort(sims)[::-1][:top_k]

        primitives: List[SupportPrimitive] = []
        for i in top_idx:
            e = h.entries[int(i)]
            primitives.append(
                SupportPrimitive(
                    q=e.q,
                    criterion=e.criterion,
                    observed_evidence=e.observed_evidence,
                    missing_check=e.missing_check,
                    exclusion_cue=e.exclusion_cue,
                    next_inquiry_rationale=e.next_inquiry_rationale,
                    reliability_score=e.reliability_score,
                    hard_issues=dict(e.hard_issues),
                    similarity=float(sims[int(i)]),
                )
            )
        return primitives


__all__ = ["PRBRetriever"]
