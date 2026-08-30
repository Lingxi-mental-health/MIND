#!/bin/bash
# 转换 RAGEN FSDP checkpoint 到 Hugging Face 格式

export CUDA_VISIBLE_DEVICES=6
MIND_CONDA_ENV="${MIND_CONDA_ENV:-mind}"
if [ -n "${MIND_CONDA_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "${MIND_CONDA_ACTIVATE}" "${MIND_CONDA_ENV}"
fi

CHECKPOINT_DIR="${MIND_PROJECT_ROOT:-$(pwd)}/checkpoints/RAGEN/exp-medical-consultation-patientllm-random_t-wo-sft/20251207_172728/best_checkpoint"
LOCAL_DIR="$CHECKPOINT_DIR/actor"
HF_CONFIG_PATH="$CHECKPOINT_DIR/actor/huggingface"
TARGET_DIR="$CHECKPOINT_DIR/hf_format"

echo "Converting FSDP checkpoint to Hugging Face format..."
echo "Source: $LOCAL_DIR"
echo "Target: $TARGET_DIR"

python "${MIND_PROJECT_ROOT:-$(pwd)}/verl/scripts/model_merger.py" \
    --backend fsdp \
    --hf_model_path "$HF_CONFIG_PATH" \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Conversion successful! HF model saved to: $TARGET_DIR"
else
    echo "❌ Conversion failed!"
    exit 1
fi

