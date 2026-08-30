"""SFT runner thin wrapper around ``ragen`` / ``verl``.

This module *does not* re-implement training; it builds the SFT input format
that matches the two-stage protocol (Stage I + Stage II) and delegates to
``ragen.trainer.fsdp_sft_trainer`` (LoRA, see Table 11).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass
class SFTConfig:
    base_model: str
    train_data: str           # parquet path (Stage I + Stage II concatenated)
    output_dir: str
    lora_rank: int = 64
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    effective_batch_size: int = 128
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    epochs: int = 3
    max_seq_length: int = 4096
    grad_clip: float = 1.0
    project_name: str = "mind_v2_sft"

    def to_overrides(self) -> Sequence[str]:
        """Convert into Hydra-style overrides for ``ragen.trainer.fsdp_sft_trainer``."""

        return [
            f"sft.training.base_model={self.base_model}",
            f"sft.training.lora_rank={self.lora_rank}",
            f"sft.training.lora_alpha={self.lora_alpha}",
            f"sft.training.learning_rate={self.learning_rate}",
            f"sft.training.train_batch_size={self.effective_batch_size}",
            f"sft.training.micro_batch_size={self.per_device_batch_size}",
            f"sft.training.epochs={self.epochs}",
            f"sft.training.max_length={self.max_seq_length}",
            f"sft.output_dir={self.output_dir}",
            f"sft.training.project_name={self.project_name}",
            f"sft.data_generation.data_dir={os.path.dirname(self.train_data)}",
        ]


def write_sft_jsonl(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Serialise SFT rows; each row must contain ``prompt`` / ``response``."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            assert "prompt" in row and "response" in row, row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def launch_sft(cfg: SFTConfig) -> int:
    """Invoke the ``ragen`` SFT entry-point."""

    cmd = [
        "python", "-m", "ragen.trainer.fsdp_sft_trainer",
        *cfg.to_overrides(),
    ]
    print("[mind_v2.sft] launching:", " ".join(cmd))
    return subprocess.call(cmd)


__all__ = ["SFTConfig", "write_sft_jsonl", "launch_sft"]
