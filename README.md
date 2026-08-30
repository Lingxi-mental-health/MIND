# MIND 🧠: Unified Inquiry–Diagnosis RL with Criteria-Grounded Clinical Supports for Psychiatric Consultation

<p align="center">
  <a href="https://arxiv.org/abs/2603.03677"><img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP%202026-Accepted-6f42c1.svg"></a>
  <a href="https://arxiv.org/abs/2603.03677"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.03677-b31b1b.svg"></a>
  <a href="https://huggingface.co/datasets/Lyncia/LingxiDiag-16K"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LingxiDiag--16K-yellow.svg"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-8B"><img alt="Base model" src="https://img.shields.io/badge/Base-Qwen3--4B%2F8B-blue.svg"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/Code-Apache--2.0-green.svg"></a>
</p>

<p align="center">
  <b>📄 Paper (EMNLP 2026):</b> <a href="https://arxiv.org/abs/2603.03677">arXiv:2603.03677</a> &nbsp;·&nbsp;
  <b>🤗 Dataset:</b> <a href="https://huggingface.co/datasets/Lyncia/LingxiDiag-16K">Lyncia/LingxiDiag-16K</a> &nbsp;·&nbsp;
  <b>🌏 中文:</b> <a href="README.zh-CN.md">README.zh-CN.md</a>
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Experiment Scripts](#experiment-scripts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

**MIND** is an evidence-grounded, process-supervised reinforcement learning framework that unifies **inquiry** and **diagnosis** for psychiatric consultation. Psychiatric interviewing poses challenges that general medical dialogue does not: subjective ambiguity, comorbidity, and the need to keep extracting psychopathological cues from incomplete and internally inconsistent patient reports. Existing approaches share two root weaknesses:

1. Without criteria grounding, they produce **unsupported clinical assertions**.
2. Across multi-turn interaction, they fail to suppress **inquiry drift** — off-topic or low-yield questioning.

MIND addresses both with three mechanisms:

1. **Criteria-grounded Psychiatric Reasoning Bank (PRB).** Multi-turn dialogue context is distilled into a clinical retrieval state, semantically similar reference consultations are retrieved, and reusable criteria-aligned clinical supports are distilled to guide inquiry and differential diagnosis.
2. **Explicit clinical reasoning with rubric process supervision.** The policy must emit a structured reasoning chain (symptom analysis → differential diagnosis → decision logic), and an LLM judge returns fine-grained per-turn process rewards.
3. **Value-aware trajectory rectification.** Low-yield turns are detected and adaptively trigger self-retry or PRB-guided fallback, suppressing inquiry drift and raising multi-turn information-gain efficiency.

## Key Features

- 🏦 **Criteria-grounded PRB** — reliable retrieval via structured clinical retrieval states, supplying criteria-aligned inquiry hints instead of retrieving directly from metaphorical psychiatric narratives.
- 🔍 **Two-stage retrieve-then-reason format** — each turn first emits `<rag_query>` to query the PRB, then conditions on the retrieved support to emit `<think>…</think><answer>…</answer>`.
- 📊 **Rubric process reward** — an LLM judge scores symptom coverage ($S^{\text{sym}}$), differential diagnosis ($S^{\text{diff}}$), and decision logic ($S^{\text{dec}}$), giving a dense per-turn supervision signal.
- 🔄 **Value-aware trajectory rectification** — rule-based detection of repetition, malformed output, and budget violations triggers self-retry or PRB-guided fallback (SCID-5-style reference inquiry).
- 🎯 **Hybrid reward optimization** — process reward, retrieval shaping, information gain, action penalties, and terminal diagnostic accuracy are optimized jointly over inquiry policy and diagnostic decision.
- ⚡ **Staged SFT → GRPO training** — a Kimi-K2-distilled SFT cold start (DeepSeek-R1 style) establishes the stable two-stage turn format; GRPO then optimizes multi-turn decision making.

## Methodology

<p align="center">
  <img src="figures/framework.png" alt="MIND framework overview" width="100%">
</p>

MIND consists of three core modules.

### Module 1: Criteria-grounded Psychiatric Reasoning Bank (PRB)

The PRB is a knowledge base pairing consultation states with criteria-aligned inquiry supports. Each historical case is distilled into a **clinical retrieval state** $q_i$ — keyword-dense, fact-only, with unspecified fields explicitly marked "not mentioned / unclear". Kimi-K2 then synthesizes textbooks and clinical guidelines into a knowledge support $r_i$ covering known facts, missing checks, and the rationale for the next inquiry. PRB entries are quality-assessed by an LLM judge (1–5 reliability score) to enforce clinical rigor.

### Module 2: Explicit clinical reasoning and process supervision

Each turn executes a two-stage format.

**Stage I** — retrieval query generation:

$$y_t^{(1)} = \texttt{<rag\_query>} q_t \texttt{</rag\_query>}$$

**Stage II** — reason-and-reply generation, conditioned on the retrieved support:

$$y_t^{(2)} = \texttt{<think>} z_t \texttt{</think>} \texttt{<answer>} a_t \texttt{</answer>}$$

The reasoning chain $z_t$ must explicitly cover (1) symptom analysis — confirmed/excluded findings and the single most critical missing piece of information; (2) differential consideration — competing explanations and exclusion cues; (3) decision logic — why this question is the most informative next step.

**Rubric process reward.** An LLM judge scores the reasoning chain along three dimensions:

$$\mathbf{S}_t = (S_t^{\text{sym}}, S_t^{\text{diff}}, S_t^{\text{dec}}), \quad r_t^{\text{proc}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{S_t^c}{S_{\max}}$$

### Module 3: Value-aware trajectory rectification

When an action is low-yield (repetition, malformed output, budget violation), the system triggers **self-retry** — reflective regeneration under tighter constraints. On persistent failure it invokes **PRB-guided fallback**, retrieving the reference inquiry of the nearest PRB entry:

$$i^* = \arg\max_i \cos(q_t, q_i), \quad a_t^{\text{ref}} = \text{RefInq}(i^*)$$

Retrieval injection is reliability-gated: an external hint is injected only when retrieval match strength exceeds a preset threshold.

### Reward aggregation

| Reward component | Computation | Weight |
|---|---|---|
| Diagnostic accuracy (terminal) | $\mathbb{1}[d_p = d^*]$ | 5.0 |
| Information gain | $\lambda_{\text{gain}} \cdot \Delta_t$ (newly revealed clinical cues) | 0.005 |
| Clinical reasoning (process) | normalized mean of the three rubric dimensions | 0.01 |
| Format compliance (penalty) | syntax errors, repetition, budget violations | 0.1 |

Total reward: $r_t = r_t^{\text{proc}} + r_t^{\text{retr}} + r_t^{\text{gain}} + r_t^{\text{pen}}$, and $R = \sum_{t=1}^T \alpha r_t + \beta r^{\text{term}}$.

## Experiments

### Patient-agent simulation quality

The patient agent is evaluated along four dimensions — information control (IC), response completeness (RC), factual conflict rate (FC), and human-likeness (HL) — validated by both an LLM judge and domain experts.

<p align="center">
  <img src="figures/Patient_Performance.png" alt="Patient agent simulation quality" width="100%">
</p>

### Doctor-agent diagnostic performance

<p align="center">
  <img src="figures/Doctor_Performance.png" alt="Doctor agent diagnostic performance" width="100%">
</p>

MIND achieves the best performance under both patient simulators (PsySim-Std and PsySim-Adapt):

| Method | PsySim-Std Acc | PsySim-Std F1 | PsySim-Adapt Acc | PsySim-Adapt F1 |
|---|---|---|---|---|
| GPT-4o | 49.5 | — | 40.5 | — |
| DDO | 53.0 | — | 45.7 | — |
| DoctorAgent-RL | 56.5 | — | 47.8 | — |
| **MIND-8B (ours)** | **71.5** | **72.5** | **62.5** | **63.1** |

### Support faithfulness

MIND scores best on all three dimensions — factual consistency (FC), support groundedness (SG), and patient faithfulness (PF) — with a mean of **8.6/10**, ahead of DDO (8.1) and DoctorAgent-RL (8.0).

### Ablation study

<p align="center">
  <img src="figures/Ablation_Study_1.png" alt="Ablation study" width="100%">
</p>

Key findings: removing thinking supervision causes the largest degradation (F1 drops ~12–14%); removing the PRB drops F1 by ~5–6%; removing the fallback mechanism drops F1 by ~3%.

### Dynamic turn-budget analysis

<p align="center">
  <img src="figures/Dynamic_Turn_Budget.png" alt="Dynamic turn budget analysis" width="85%">
</p>

## Repository Structure

The repository contains two implementations. **`mind_v2/` is the paper-aligned reference implementation and the recommended entry point**; `ragen/` + `verl/` is the original training stack it is built on.

```
MIND/
├── mind_v2/              # Paper-aligned reference implementation (recommended)
│   ├── mind/
│   │   ├── evidence_state/   # Evidence-state interface E_t = (O, M, D, S, R)
│   │   ├── prb/              # PRB: build / judge / retrieve / state operator
│   │   ├── env/              # MIND environment + PsySim patient simulators
│   │   ├── reward/           # Process, info-gain, format, terminal rewards
│   │   ├── rectification/    # Trajectory rectification triggers and rectifier
│   │   ├── evaluation/       # MCR, field masking, support faithfulness, MDD-5k
│   │   └── training/         # SFT and GRPO runners
│   ├── configs/          # sft_lora / rl_grpo / inference / llm_providers
│   └── scripts/          # PRB build, training, and evaluation entry points
├── ragen/                # Original RL stack: multi-turn agent, med_dialogue env
├── verl/                 # Vendored veRL (GRPO/PPO infrastructure, Apache-2.0)
├── config/               # Hydra configs for the ragen stack
├── scripts/              # Data processing, training, and checkpoint scripts
├── figures/              # Figures used in this README
├── data/                 # Place datasets here (not tracked in git)
└── requirements.txt
```

See [`mind_v2/README.md`](mind_v2/README.md) for the paper-section ↔ code index.

## Setup

### 1. Environment

The two stacks were developed against different toolchains. Use the one matching the code you intend to run.

```bash
# Recommended: matches scripts/setup_ragen.sh (CUDA 12.1)
conda create -n mind python=3.10 -y
conda activate mind
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt

# Vendored veRL, without pulling its own pins
pip install -e verl --no-dependencies

# Optional: Flash Attention 2
pip install flash-attn --no-build-isolation
```

`requirements.txt` pins the versions this work was validated with, notably `vllm==0.6.4.post1` and `ray[default]==2.10.0`.

> **Ray security note.** Ray's dashboard and job-submission API are unauthenticated by default (CVE-2023-48022). Keep the head node bound to localhost — the training scripts use `RAY_ADDRESS=127.0.0.1:<port>` — and never expose the dashboard port on a public interface.

### 2. Credentials

Training and evaluation read all credentials from the environment. Nothing is committed to the repository.

```bash
export WANDB_API_KEY="<your-wandb-key>"     # or: export WANDB_MODE=offline
export HTTP_PROXY_URL="http://user:pass@proxy-host:port"   # optional
export OPENROUTER_API_KEY="<key>"          # PRB build / LLM judge (Kimi-K2, GLM)
export DEEPSEEK_API_KEY="<key>"            # support-faithfulness judge
```

### 3. Paths

No absolute paths are baked into the code. Everything resolves relative to the
repository, and each root is overridable:

| Variable | Default | Purpose |
|---|---|---|
| `MIND_PROJECT_ROOT` | current directory | run outputs: logs, checkpoints |
| `MIND_DATA_ROOT` | `./data` | datasets and PRB artifacts |
| `MIND_MODEL_ROOT` | `./models` | local model weights |
| `MIND_LOG_ROOT` | `./logs/readable` | readable trajectory debug logs |
| `MIND_BGE_MODEL_PATH` | `BAAI/bge-large-zh-v1.5` | retrieval encoder |
| `MIND_CONDA_SH` | unset (skips activation) | conda init script for the training scripts |
| `MIND_CONDA_ENV` | `mind` | conda environment the training scripts activate |

### 4. Security defaults

Two switches are off by default and should stay off unless you have a specific,
reviewed reason:

- **`trust_remote_code` is `False`.** Enabling it makes `from_pretrained` execute
  code shipped inside a model repository. Qwen3 does not need it. To opt in for a
  model whose repository you have reviewed: `export MIND_TRUST_REMOTE_CODE=1`.
- **Checkpoints load with `weights_only=True`**, and PRB indexes are stored as
  JSONL rather than pickle. Both deserialization paths used to be able to execute
  arbitrary code from a file. Legacy pickle indexes are refused; convert them
  explicitly with `mind_v2/scripts/prb/convert_legacy_index.py` or
  `scripts/convert_legacy_rag_index.py`, and only for files you produced.

### 5. Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 4× A100 (40GB) | 8× A100 (80GB) |
| RAM | 128GB | 256GB |
| Storage | 50GB | 200GB |

### 6. Base models

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) — doctor-agent backbone for MIND-8B, and the frozen patient agent
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) — doctor-agent backbone for MIND-4B

## Dataset

**🤗 [Lyncia/LingxiDiag-16K](https://huggingface.co/datasets/Lyncia/LingxiDiag-16K)** — synthetic psychiatric consultation dialogues paired with EMR-derived patient profiles (Chinese, CC BY-NC 4.0).

```python
from datasets import load_dataset

ds = load_dataset("Lyncia/LingxiDiag-16K")
```

Place the downloaded files under `data/` (or point `MIND_DATA_ROOT` elsewhere). The test split is isolated throughout: the PRB is built from the training split only.

## Experiment Scripts

### 1. Data preprocessing

```bash
# Rebuild patient profiles from EMRs and generate dialogues
python scripts/data_process/extract_medical_data.py
python scripts/data_process/convert_dialog_format.py

# Produce train/validation splits
python scripts/data_process/split_train_val.py
```

### 2. Build the Psychiatric Reasoning Bank (PRB)

```bash
# Build clinical retrieval states (training split only)
python -m mind_v2.scripts.prb.build_clinical_states --split train --out data/prb/states.jsonl

# Distill criteria-aligned supports (--builder_llm selects the teacher, e.g. kimi-k2)
python -m mind_v2.scripts.prb.build_with_llm --builder_llm kimi-k2 \
    --in data/prb/states.jsonl --out data/prb/supports.jsonl

# LLM-judge reliability scoring (1–5) and filtering
python -m mind_v2.scripts.prb.judge_reliability \
    --in data/prb/supports.jsonl --out data/prb/supports_scored.jsonl

# Build the ANN index
python -m mind_v2.scripts.prb.build_index \
    --in data/prb/supports_scored.jsonl --out data/prb/index/
```

### 3. Train the doctor agent

**Stage 1 — SFT cold start** (Kimi-K2 distillation, DeepSeek-R1 style)

| Hyperparameter | Value |
|---|---|
| Method | LoRA |
| LoRA rank | 64 |
| LoRA alpha | 32 |
| Learning rate | 1e-4 |
| Warmup ratio | 0.03 |

```bash
bash mind_v2/scripts/training/train_sft.sh
```

**Stage 2 — GRPO reinforcement learning**

| Hyperparameter | Value |
|---|---|
| Algorithm | GRPO |
| Compute | 8× NVIDIA A100 |
| Actor learning rate | 5e-6 |
| KL coefficient | 0.02 |
| Clip ratio | 0.10–0.18 |
| Diagnostic accuracy weight | 5.0 |
| Information gain weight | 0.005 |
| Clinical reasoning weight | 0.01 |
| Format compliance weight | 0.1 |

```bash
# Paper-aligned GRPO training
bash mind_v2/scripts/training/train_grpo.sh

# Original ragen/veRL stack (PPO/GRPO baseline; requires WANDB_API_KEY)
bash scripts/training/train_ppo_baseline.sh

# Resume from a checkpoint
bash scripts/training/train_ppo_resume.sh
```

Training is monitored through WandB. The maximum dialogue length is $L=10$ turns.

### 4. Run evaluation

```bash
# Main table: end-to-end inference under both patient simulators
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_std
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_adapt

# Support faithfulness (FC / SG / PF), DeepSeek judge
python -m mind_v2.scripts.evaluation.run_faithfulness \
    --results results/run.jsonl --judge deepseek_v32

# Automatic MCR evaluation
python -m mind_v2.scripts.evaluation.run_mcr --results results/run.jsonl

# Field-masking ablation over O / M / D / S / R
python -m mind_v2.scripts.evaluation.run_field_masking --ckpt path/to/MIND-8B
```

The original stack's evaluation entry points remain available:

```bash
python ragen/env/med_dialogue/evaluation/inference_fast_for_patientllm_zh_1018_3_best.py \
    --model_path /path/to/checkpoint \
    --data_path data/test.parquet \
    --output_dir results/

python ragen/env/med_dialogue/evaluation/evaluation_for_patientllm_category_zh_optimized_best.py \
    --result_path results/inference_output.json
```

### 5. Checkpoint conversion

```bash
# Convert a distributed training checkpoint to HuggingFace format
bash scripts/convert_best_checkpoint.sh
```

## Citation

If MIND is useful for your research, please cite our work:

MIND has been **accepted to EMNLP 2026**.

```bibtex
@inproceedings{li2026mind,
  title={MIND: Unified Inquiry and Diagnosis RL with Criteria Grounded Clinical Supports for Psychiatric Consultation},
  author={Li, Guoyi and Xu, Shihao and Ma, Jiatong and Han, Yunyun and Chen, Jianhua and Deng, Yafeng},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026},
  url={https://arxiv.org/abs/2603.03677}
}
```

## Acknowledgements

- [veRL](https://github.com/volcengine/verl) — GRPO training infrastructure
- [vLLM](https://github.com/vllm-project/vllm) — efficient LLM inference engine
- [Ray](https://github.com/ray-project/ray) — distributed computing framework
- [Qwen3](https://huggingface.co/Qwen) — base language models (Qwen3-4B/8B for the doctor agent, Qwen3-8B for the patient agent)
- [Kimi-K2](https://arxiv.org/abs/2507.20534) — teacher model for PRB construction and SFT distillation

## License

This project's code is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE). The vendored `verl/` directory retains its own Apache-2.0 license (see [`verl/LICENSE`](verl/LICENSE)). The **LingxiDiag-16K** dataset is released under CC BY-NC 4.0 — non-commercial use only.

> **Intended use.** MIND is a research artifact for studying multi-turn clinical inquiry and diagnostic reasoning. It is **not** a medical device and must not be used for diagnosis or treatment of real patients.
