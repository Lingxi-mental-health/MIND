"""GRPO runner — wraps ``ragen.trainer.main_ppo`` with MIND-aware overrides.

This is intentionally a thin shell.  All MIND-specific behaviour lives in:

* :class:`mind.env.MindEnv` (state factorisation, two-stage format).
* :class:`mind.reward.MindRewardManager` (process / info-gain / format /
  retrieval / terminal aggregation).
* :class:`mind.rectification.Rectifier` (state-conditioned recovery).

The runner exports them under ``ragen``-compatible attribute names so that
``ragen.trainer.main_ppo`` can pick them up via Hydra config overrides.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class GRPOConfig:
    base_model: str
    sft_ckpt: str = ""
    train_parquet: str = ""
    val_parquet: str = ""
    project_name: str = "mind_v2_grpo"
    experiment_name: str = "mind_v2_grpo_run"

    actor_lr: float = 5e-6
    kl_coef: float = 0.02
    clip_low: float = 0.10
    clip_high: float = 0.18
    entropy_coef: float = 1e-4
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    generations_per_prompt: int = 4
    policy_updates: int = 1500
    episodes_per_update: int = 128
    minibatch_size: int = 64
    micro_batch_per_gpu: int = 1
    ppo_epochs_per_update: int = 1
    grad_clip: float = 1.0
    max_prompt_length: int = 4096
    max_response_length: int = 1024
    max_turns: int = 10

    # Reward weights (Table 11).
    weight_terminal: float = 5.0
    weight_info_gain: float = 0.005
    weight_process: float = 0.01
    weight_format: float = 0.1

    # Reliability gating.
    reliability_threshold: float = 0.2
    top_k: int = 4

    extra: List[str] = field(default_factory=list)

    def to_overrides(self) -> Sequence[str]:
        """Hydra-style overrides for ``ragen.trainer.main_ppo``."""

        return [
            f"model.base_model={self.base_model}",
            f"model.experiment_name={self.experiment_name}",
            f"trainer.project_name={self.project_name}",
            f"training.max_turns={self.max_turns}",
            f"training.max_response_length={self.max_response_length}",
            f"training.max_start_length={self.max_prompt_length}",
            f"optimization.actor_lr={self.actor_lr}",
            f"optimization.kl_coef={self.kl_coef}",
            f"+mind.reward.weight_terminal={self.weight_terminal}",
            f"+mind.reward.weight_info_gain={self.weight_info_gain}",
            f"+mind.reward.weight_process={self.weight_process}",
            f"+mind.reward.weight_format={self.weight_format}",
            f"+mind.prb.reliability_threshold={self.reliability_threshold}",
            f"+mind.prb.top_k={self.top_k}",
            f"training.train_data_path={self.train_parquet}",
            f"training.val_data_path={self.val_parquet}",
            *self.extra,
        ]


def launch_grpo(cfg: GRPOConfig) -> int:
    """Invoke ``python -m ragen.trainer.main_ppo`` with MIND-v2 overrides."""

    cmd = [
        "python", "-m", "ragen.trainer.main_ppo",
        *cfg.to_overrides(),
    ]
    print("[mind_v2.grpo] launching:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MIND_V2", "1")
    return subprocess.call(cmd, env=env)


__all__ = ["GRPOConfig", "launch_grpo"]
