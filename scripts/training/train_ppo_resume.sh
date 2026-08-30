#!/bin/bash

# ---------------------------------------------------------------------------
# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进脚本
#   MIND_PROJECT_ROOT  实验产出根目录（日志/checkpoint），默认当前目录
#   MIND_DATA_ROOT     数据集目录，默认 ./data
#   MIND_MODEL_ROOT    本地模型目录，默认 ./models
#   MIND_CONDA_SH      conda 初始化脚本；未设置则跳过 conda 激活
#   MIND_CONDA_ENV     conda 环境名，默认 mind
# ---------------------------------------------------------------------------
MIND_PROJECT_ROOT="${MIND_PROJECT_ROOT:-$(pwd)}"
MIND_DATA_ROOT="${MIND_DATA_ROOT:-${MIND_PROJECT_ROOT}/data}"
MIND_MODEL_ROOT="${MIND_MODEL_ROOT:-${MIND_PROJECT_ROOT}/models}"
MIND_CONDA_ENV="${MIND_CONDA_ENV:-mind}"

if [ -n "${MIND_CONDA_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "${MIND_CONDA_SH}"
  conda activate "${MIND_CONDA_ENV}"
fi

# ---------------------------------------------------------------------------
# 代理与凭据：一律从环境变量读取，禁止硬编码写入仓库
#   export HTTP_PROXY_URL="http://<user>:<pass>@<proxy-host>:<port>"   # 可选
#   export WANDB_API_KEY="<your-wandb-key>"                            # 必填（或 wandb offline）
# ---------------------------------------------------------------------------
if [ -n "${HTTP_PROXY_URL:-}" ]; then
  export http_proxy="${HTTP_PROXY_URL}" https_proxy="${HTTP_PROXY_URL}"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY 未设置。请先 export WANDB_API_KEY=<your-key>，" >&2
  echo "       或改用离线模式：export WANDB_MODE=offline" >&2
  exit 1
fi
wandb login "${WANDB_API_KEY}"



# wandb offline
# 续训配置：从 step 1050 续训（stable_resume_checkpoint）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0  # 强制使用 V0 engine，避免显存不足
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEBUG_POST_MORTEM=0

# 【关键】使用 GPU 2,3,6,7
export CUDA_VISIBLE_DEVICES="2,3,6,7"

# 新增：解决vLLM sleep mode问题的Ray配置
export VLLM_DISABLE_SLEEP_MODE=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_FULL_DELAY_MS=5000


export HYDRA_FULL_ERROR=1

# PPO 轨迹日志配置 - 输出完整的 advantages, returns, rewards 等信息
export RAGEN_LOG_PPO_TRAJECTORY=1
# 可读轨迹日志目录
export RAGEN_READABLE_LOG_DIR="${MIND_PROJECT_ROOT}/logs/readable"
mkdir -p ${RAGEN_READABLE_LOG_DIR}

# Ray 集群配置 - 使用独立端口避免与其他实验冲突
# 【run_simple_0127_resume_rag 专用端口】（独立端口，不影响其他脚本）
RAY_PORT=6412
RAY_DASHBOARD_PORT=8298
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

# 注意：不执行 ray stop，因为 GPU 上可能还有其他程序在运行
# 只启动使用独立端口的 Ray 集群，不影响其他实验

# 启动独立的 Ray 集群（使用独立临时目录）
ray start --head --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-gpus=4 --temp-dir=/tmp/ray_simple_0127_${RAY_PORT}

# 设置 Ray 地址环境变量
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

echo "Ray 集群已启动: 端口=${RAY_PORT}, Dashboard=${RAY_DASHBOARD_PORT}"

clip_ratio_low=0.1    # 核心防崩：从 0.2 降到 0.1
clip_ratio_high=0.18  # 相应调整 high（保持 low+0.08 的间距）
entropy_coeff=1e-4

# ========== 从 step 1050 续训配置（使用 RAG 二步流程）==========
# 续训 checkpoint 路径（stable_resume_checkpoint，step 1050）
RESUME_CKPT_DIR="${MIND_PROJECT_ROOT}/checkpoints/RAGEN/run_simple_1227_resume_20260112_202843/20260112_202843/stable_resume_checkpoint"

# model_path 必须是完整的 HuggingFace 模型目录（用于初始化模型结构和 ref 模型）
model_path="${MIND_PROJECT_ROOT}/DoctorLLM-7B-SFT-clean-20251213_000004/DocAgent-8B-SFT-Qwen3-8B-base-final-20251213_152654"

# 续训起始步数（用于 trainer 和 wandb）
RESUME_STEP=1050

# 实验名称 = 脚本名 + 时间戳（标识为 resume + RAG）
script_name=$(basename "$0" .sh)
timestamp=$(date +%Y%m%d_%H%M%S)
exp_name="${script_name}_${timestamp}"
project_name=RAGEN

# 实验产出根目录
PROJECT_ROOT="${MIND_PROJECT_ROOT}"

# 创建时间戳用于日志文件夹命名
log_dir="${PROJECT_ROOT}/logs/${timestamp}"
log_file="${log_dir}/training.log"

# Checkpoint 目录配置（续训使用新的时间戳保存新 checkpoint）
checkpoint_timestamp="${timestamp}"

echo "=== RAG 二步流程 + 续训配置 ==="
echo "续训起始步数: ${RESUME_STEP}"
echo "续训 checkpoint: ${RESUME_CKPT_DIR}"
echo "GPU 配置: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==================================="

# 创建日志目录
mkdir -p ${log_dir}

echo "=== 续训开始时间: $(date) ==="
echo "从 checkpoint 续训: ${RESUME_CKPT_DIR}"
echo "日志文件: ${log_file}"
echo "=================================="

# 使用子shell和后台运行，避免SIGHUP信号影响
(
python -m ragen.trainer.main_ppo \
  hydra.run.dir=outputs/exp_configs/logs/$(date +%Y-%m-%d)/$(date +%H-%M-%S) \
  data.train_files=${MIND_DATA_ROOT}/merged_24k_enhanced_mixed.parquet   \
  data.val_files=${MIND_DATA_ROOT}/balanced_4cat_50each_natural_mixed.parquet     \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=16 \
  data.val_batch_size=32 \
  data.max_prompt_length=3072 \
  data.max_response_length=3200 \
  data.max_start_length=500 \
  data.max_obs_length=3072 \
  data.shuffle=True \
  ++data.use_balanced_sampling=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=${model_path} \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  ++actor_rollout_ref.model.override_config.use_flash_attention_2=True \
  ++actor_rollout_ref.model.use_liger=True \
  ++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  ++actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  ++actor_rollout_ref.actor.optim.override_lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  ++actor_rollout_ref.model.use_remove_padding=False \
  ++actor_rollout_ref.actor.fsdp_config.param_offload=True \
  ++actor_rollout_ref.actor.use_torch_compile=False \
  ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  ++actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \
  ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=5200 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n_agent=8 \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_ctrl.kl_coef=0.02 \
  ++algorithm.kl_ctrl.target_kl=0.02 \
  +algorithm.no_ref_policy=False \
  +actor_rollout_ref.actor.use_ref_policy=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  ++actor_rollout_ref.actor.kl_loss_coef=0.02 \
  ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
  ++actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=bf16 \
  ++actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  ++actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
  +algorithm.no_think_rl=False \
  +algorithm.reward_norm_type=grpo \
  +actor_rollout_ref.actor.optim.betas=[0.9,0.95] \
  ++actor_rollout_ref.actor.grad_clip=0.5 \
  ++actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  ++actor_rollout_ref.rollout.top_k=-1 \
  ++actor_rollout_ref.rollout.min_p=0.0 \
  actor_rollout_ref.rollout.do_sample=True \
  ++actor_rollout_ref.rollout.repetition_penalty=1.0 \
  actor_rollout_ref.actor.state_masking=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  ++actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  trainer.logger=['wandb','console'] \
  +trainer.val_only=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  ++trainer.use_separate_ref_pool=False \
  ++trainer.ref_n_gpus=0 \
  trainer.save_freq=10 \
  trainer.max_actor_ckpt_to_keep=2 \
  ++trainer.save_best_checkpoint=True \
  ++trainer.best_metric_name="validate_metric/diagnosis_accuracy_overall" \
  ++trainer.stable_resume_metric_name="validate_metric/diagnosis_accuracy_overall" \
  trainer.val_before_train=True \
  trainer.test_freq=10 \
  trainer.project_name=${project_name} \
  trainer.experiment_name=${exp_name} \
  trainer.total_epochs=10 \
  trainer.total_training_steps=5000 \
  +trainer.checkpoint_dir=${PROJECT_ROOT}/checkpoints/${project_name}/${exp_name}/${checkpoint_timestamp} \
  +trainer.ref_update_steps=null \
  trainer.default_local_dir=${PROJECT_ROOT}/checkpoints/${project_name}/${exp_name}/${checkpoint_timestamp} \
  trainer.resume_from_path=${RESUME_CKPT_DIR} \
  ++trainer.resume_step=${RESUME_STEP} \
  env.name=medical_consultation_patient_llm_category \
  env.use_env_llm=True \
  +env.max_turns=5 \
  +algorithm.no_critic=True \
  env.env_llm.fsdp_config.fsdp_size=-1 \
  env.env_llm.fsdp_config.param_offload=False \
  env.env_llm.vllm_config.tensor_parallel_size=1 \
  env.env_llm.vllm_config.gpu_memory_utilization=0.3 \
  env.env_llm.vllm_config.max_num_batched_tokens=6000 \
  env.env_llm.vllm_config.max_num_seqs=1 \
  env.env_llm.model.path="${MIND_MODEL_ROOT}/Qwen3-8B" \
  env.env_llm.model.trust_remote_code=False \
  env.env_llm.model.use_liger=True \
  env.env_llm.model.override_config.max_position_embeddings=5500 \
  env.env_llm.generation.prompt_length=2048 \
  env.env_llm.generation.response_length=512 \
  env.env_llm.generation.max_model_len=3000 \
  env.env_llm.generation.temperature=0.0 \
  env.env_llm.generation.top_p=1.0  \
  env.env_llm.generation.top_k=-1 \
  env.env_llm.generation.repetition_penalty=1.0 \
  env.env_llm.generation.do_sample=False \
  env.env_llm.generation.num_beams=1 \
  env.env_llm.generation.best_of=1 \
  env.env_llm.generation.min_p=0.05 \
  env.env_llm.generation.n=1 \
  env.env_llm.generation.use_cache=True \
  env.env_llm.generation.tensor_model_parallel_size=1 \
  env.env_llm.generation.use_beam_search=False \
  env.env_llm.generation.detokenize=False \
  env.env_llm.generation.ignore_eos=False \
  env.env_llm.generation.free_cache_engine=True \
  env.env_llm.generation.prompt_logprobs=null \
  env.env_llm.generation.generation_logprobs=0 \
  env.env_llm.generation.disable_log_stats=True \
  env.env_llm.generation.dtype=bfloat16 \
  env.env_llm.generation.enforce_eager=True \
  env.env_llm.generation.enable_chunked_prefill=False \
  env.env_llm.generation.gpu_memory_utilization=0.25 \
  env.env_llm.generation.load_format=dummy_dtensor \
  env.env_llm.generation.max_tokens_per_batch=6000 \
  env.env_llm.ulysses_sequence_parallel_size=1 \
  max_turns=5 \
  logging.log_images=false \
  logging.log_image_dir=${log_dir}/trajectory \
  logging.log_image_step_size=20 \
  logging.log_n_image_per_batch=16
) > ${log_file} 2>&1 &

# 保存进程PID
echo $! > ${log_dir}/training.pid
echo "续训进程已启动，PID: $(cat ${log_dir}/training.pid)"
echo "使用命令查看实时日志: tail -f ${log_file}"
echo ""
echo "提示：训练在后台运行，可以安全退出终端。"
echo "停止训练: kill \$(cat ${log_dir}/training.pid)"
echo "清理 Ray: pkill -f 'ray_simple_0127_6412'; lsof -i :6412 2>/dev/null | awk 'NR>1 {print \$2}' | sort -u | xargs -r kill -9"

