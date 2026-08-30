#!/usr/bin/env bash
# GRPO RL training (Table 12). Wraps `mind_v2.mind.training.grpo_runner`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIND_V2_ROOT="$(cd "${HERE}/../.." && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SFT_CKPT="${SFT_CKPT:-${MIND_V2_ROOT}/outputs/sft/best}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${MIND_V2_ROOT}/data/rl/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${MIND_V2_ROOT}/data/rl/val.parquet}"

cd "${MIND_V2_ROOT}/.."
exec python -c "
from mind_v2.mind.training import GRPOConfig, launch_grpo
cfg = GRPOConfig(
    base_model='${BASE_MODEL}',
    sft_ckpt='${SFT_CKPT}',
    train_parquet='${TRAIN_PARQUET}',
    val_parquet='${VAL_PARQUET}',
)
raise SystemExit(launch_grpo(cfg))
"
