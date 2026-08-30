#!/usr/bin/env bash
# SFT warm-start (Table 11). Wraps `mind_v2.mind.training.sft_runner`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIND_V2_ROOT="$(cd "${HERE}/../.." && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-${MIND_V2_ROOT}/data/sft/train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIND_V2_ROOT}/outputs/sft}"

cd "${MIND_V2_ROOT}/.."
exec python -c "
from mind_v2.mind.training import SFTConfig, launch_sft
cfg = SFTConfig(
    base_model='${BASE_MODEL}',
    train_data='${TRAIN_DATA}',
    output_dir='${OUTPUT_DIR}',
)
raise SystemExit(launch_sft(cfg))
"
