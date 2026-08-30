# MIND 🧠：面向精神科问诊的统一问询-诊断强化学习框架

**MIND: Unified Inquiry–Diagnosis RL with Criteria-Grounded Clinical Supports for Psychiatric Consultation**

<p align="center">
  <a href="https://arxiv.org/abs/2603.03677"><img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP%202026-Accepted-6f42c1.svg"></a>
  <a href="https://arxiv.org/abs/2603.03677"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.03677-b31b1b.svg"></a>
  <a href="https://huggingface.co/datasets/Lyncia/LingxiDiag-16K"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LingxiDiag--16K-yellow.svg"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-8B"><img alt="Base model" src="https://img.shields.io/badge/Base-Qwen3--4B%2F8B-blue.svg"></a>
  <a href="#许可协议"><img alt="License" src="https://img.shields.io/badge/Code-Apache--2.0-green.svg"></a>
</p>

<p align="center">
  <b>📄 论文（EMNLP 2026）：</b><a href="https://arxiv.org/abs/2603.03677">arXiv:2603.03677</a> &nbsp;·&nbsp;
  <b>🤗 数据集：</b><a href="https://huggingface.co/datasets/Lyncia/LingxiDiag-16K">Lyncia/LingxiDiag-16K</a> &nbsp;·&nbsp;
  <b>🌏 English:</b> <a href="README.md">README.md</a>
</p>

---

## 目录

- [介绍](#介绍)
- [主要特性](#主要特性)
- [方法论](#方法论)
- [实验](#实验)
- [仓库结构](#仓库结构)
- [设置](#设置)
- [数据集](#数据集)
- [实验脚本](#实验脚本)
- [引用](#引用)
- [致谢](#致谢)
- [许可协议](#许可协议)

## 介绍

**MIND** 是一个面向精神科问诊的**证据支撑、过程监督**统一问询-诊断强化学习框架。精神科问诊面临通用医疗对话所没有的挑战：主观模糊性、共病复杂性，以及从不完整、不一致的患者陈述中持续提取精神病理线索的需求。现有方法存在两大根本缺陷：

1. 缺乏标准支撑时易产生**无据可查的临床断言**；
2. 多轮交互中难以抑制**问询漂移**（偏题或低效问询）。

为解决上述问题，MIND 提出三项核心机制：

1. **标准支撑的精神科推理库（PRB）**：将多轮对话上下文提炼为临床检索状态，检索语义相近的参考问诊案例，蒸馏出可复用的标准对齐临床支撑，引导问询与差异诊断。
2. **显式临床推理与 Rubric 过程监督**：强制生成结构化推理链（症状分析 → 鉴别诊断 → 决策逻辑），并以 LLM 评判器提供逐轮细粒度的过程奖励。
3. **价值感知轨迹纠正**：检测低效轮次，自适应触发自我重试或 PRB 引导回退，抑制问询漂移，提升多轮信息获取效率。

## 主要特性

- 🏦 **标准支撑的精神科推理库（PRB）**：以结构化临床检索状态实现可靠检索，提供标准对齐的问询提示，避免直接从隐喻性精神叙述中检索的不可靠性。
- 🔍 **两阶段检索-推理格式**：每轮先生成 `<rag_query>` 检索 PRB，再基于检索支撑生成 `<think>…</think><answer>…</answer>` 的显式推理与回复。
- 📊 **Rubric 过程奖励**：LLM 评判器从症状覆盖（$S^{\text{sym}}$）、鉴别诊断（$S^{\text{diff}}$）、决策逻辑（$S^{\text{dec}}$）三维度评分，提供逐轮密集监督信号。
- 🔄 **价值感知轨迹纠正**：基于规则检测重复、格式异常、预算违规等低效轮次，触发自我重试或 PRB 引导回退（SCID-5 风格参考问询）。
- 🎯 **混合奖励优化**：综合过程奖励、检索塑形奖励、信息增益奖励、操作惩罚与终态诊断准确性奖励，联合优化问询策略与诊断决策。
- ⚡ **SFT→GRPO 分阶段训练**：Kimi-K2 蒸馏的 SFT 冷启动（DeepSeek-R1 风格）建立稳定的两阶段轮格式，GRPO 进一步优化多轮决策。

## 方法论

<p align="center">
  <img src="figures/framework.png" alt="MIND 框架总览" width="100%">
</p>

MIND 框架由三个核心模块构成。

### 模块一：标准支撑的精神科推理库（PRB）

PRB 是一个将问诊状态与标准对齐问询支撑配对的知识库。每条历史案例被提炼为**临床检索状态** $q_i$（关键词密集、仅含事实、未明确字段标注为"未提及/不清楚"），并经由 Kimi-K2 综合教材与临床指南，蒸馏出知识支撑 $r_i$（涵盖已知事实、缺失检查项、下一步问询依据）。PRB 条目经 LLM 评判器进行质量评估（1–5 分可靠性打分）以确保临床严谨性。

### 模块二：显式临床推理与过程监督

每轮执行两阶段格式：

**阶段 I** — 检索查询生成：

$$y_t^{(1)} = \texttt{<rag\_query>} q_t \texttt{</rag\_query>}$$

**阶段 II** — 推理-回复生成（以检索支撑为条件）：

$$y_t^{(2)} = \texttt{<think>} z_t \texttt{</think>} \texttt{<answer>} a_t \texttt{</answer>}$$

推理链 $z_t$ 需显式覆盖：（1）症状分析（已确认/排除发现及最关键缺失信息）；（2）鉴别考量（竞争解释与排除线索）；（3）决策逻辑（说明为何该问题是当前最具信息量的步骤）。

**Rubric 过程奖励**：LLM 评判器对推理链进行三维评分：

$$\mathbf{S}_t = (S_t^{\text{sym}}, S_t^{\text{diff}}, S_t^{\text{dec}}), \quad r_t^{\text{proc}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{S_t^c}{S_{\max}}$$

### 模块三：价值感知轨迹纠正

当动作表现出低效（重复、格式错误、预算违规）时，系统触发**自我重试**（更严格约束下的反思重生成）；持续失败则调用 **PRB 引导回退**，检索最近 PRB 条目的参考问询：

$$i^* = \arg\max_i \cos(q_t, q_i), \quad a_t^{\text{ref}} = \text{RefInq}(i^*)$$

检索注入受可靠性门控：仅当检索匹配强度超过预设阈值时才注入外部提示。

### 奖励聚合

| 奖励组件 | 计算方式 | 权重 |
|---|---|---|
| 诊断准确性（终态） | $\mathbb{1}[d_p = d^*]$ | 5.0 |
| 信息增益 | $\lambda_{\text{gain}} \cdot \Delta_t$（新揭示临床线索数） | 0.005 |
| 临床推理（过程） | Rubric 三维归一化均值 | 0.01 |
| 格式合规（惩罚） | 语法错误、重复、预算违规 | 0.1 |

总奖励：$r_t = r_t^{\text{proc}} + r_t^{\text{retr}} + r_t^{\text{gain}} + r_t^{\text{pen}}$，$R = \sum_{t=1}^T \alpha r_t + \beta r^{\text{term}}$。

## 实验

### 患者智能体仿真评估

患者智能体从信息控制（IC）、响应完整性（RC）、事实冲突率（FC）、拟人度（HL）四个维度评估，经 LLM 评判器与领域专家双重验证。

<p align="center">
  <img src="figures/Patient_Performance.png" alt="患者智能体仿真质量" width="100%">
</p>

### 医生智能体诊断性能

<p align="center">
  <img src="figures/Doctor_Performance.png" alt="医生智能体诊断性能" width="100%">
</p>

MIND 在两种患者模拟器（PsySim-Std 和 PsySim-Adapt）下均取得最优表现：

| 方法 | PsySim-Std Acc | PsySim-Std F1 | PsySim-Adapt Acc | PsySim-Adapt F1 |
|---|---|---|---|---|
| GPT-4o | 49.5 | — | 40.5 | — |
| DDO | 53.0 | — | 45.7 | — |
| DoctorAgent-RL | 56.5 | — | 47.8 | — |
| **MIND-8B（Ours）** | **71.5** | **72.5** | **62.5** | **63.1** |

### 支撑忠实度评估

MIND 在事实一致性（FC）、支撑接地性（SG）、患者忠实度（PF）三维度均取得最优均分 **8.6/10**，优于 DDO（8.1）和 DoctorAgent-RL（8.0）。

### 消融研究

<p align="center">
  <img src="figures/Ablation_Study_1.png" alt="消融研究" width="100%">
</p>

关键发现：去除 thinking 监督导致最大衰减（F1 下降 ~12–14%）；去除 PRB 导致 F1 下降 ~5–6%；去除回退机制导致 F1 下降 ~3%。

### 动态轮次预算分析

<p align="center">
  <img src="figures/Dynamic_Turn_Budget.png" alt="动态轮次预算分析" width="85%">
</p>

## 仓库结构

本仓库包含两套实现。**`mind_v2/` 是与论文对齐的参考实现，推荐从这里入手**；`ragen/` + `verl/` 是其所基于的原始训练栈。

```
MIND/
├── mind_v2/              # 与论文对齐的参考实现（推荐）
│   ├── mind/
│   │   ├── evidence_state/   # 证据状态接口 E_t = (O, M, D, S, R)
│   │   ├── prb/              # PRB：构建 / 评判 / 检索 / 状态构造算子
│   │   ├── env/              # MIND 环境 + PsySim 患者模拟器
│   │   ├── reward/           # 过程、信息增益、格式合规、终态奖励
│   │   ├── rectification/    # 轨迹纠正触发器与纠正器
│   │   ├── evaluation/       # MCR、字段消融、支撑忠实度、MDD-5k
│   │   └── training/         # SFT 与 GRPO runner
│   ├── configs/          # sft_lora / rl_grpo / inference / llm_providers
│   └── scripts/          # PRB 构建、训练、评估入口
├── ragen/                # 原始 RL 栈：多轮智能体、med_dialogue 环境
├── verl/                 # 内嵌 veRL（GRPO/PPO 基础设施，Apache-2.0）
├── config/               # ragen 栈的 Hydra 配置
├── scripts/              # 数据处理、训练、检查点脚本
├── figures/              # README 使用的图表
├── data/                 # 数据集放置目录（不纳入 git 跟踪）
└── requirements.txt
```

论文章节 ↔ 代码位置的对照索引见 [`mind_v2/README.md`](mind_v2/README.md)。

## 设置

### 1. 环境配置

两套栈的工具链版本不同，请按实际要运行的代码选择。

```bash
# 推荐：与 scripts/setup_ragen.sh 一致（CUDA 12.1）
conda create -n mind python=3.10 -y
conda activate mind
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install -r requirements.txt

# 内嵌 veRL，不拉取其自带的版本约束
pip install -e verl --no-dependencies

# 可选：Flash Attention 2
pip install flash-attn --no-build-isolation
```

`requirements.txt` 固定了本工作验证过的版本，其中关键的是 `vllm==0.6.4.post1` 与 `ray[default]==2.10.0`。

> **Ray 安全提示**：Ray 的 dashboard 与作业提交 API 默认无鉴权（CVE-2023-48022）。请保持 head 节点绑定在 localhost——训练脚本使用 `RAY_ADDRESS=127.0.0.1:<port>`——切勿把 dashboard 端口暴露到公网接口。

### 2. 凭据配置

训练与评估的所有凭据均从环境变量读取，仓库内不存放任何密钥。

```bash
export WANDB_API_KEY="<你的 wandb key>"     # 或改用离线模式：export WANDB_MODE=offline
export HTTP_PROXY_URL="http://user:pass@proxy-host:port"   # 可选
export OPENROUTER_API_KEY="<key>"          # PRB 构建 / LLM 评判器（Kimi-K2、GLM）
export DEEPSEEK_API_KEY="<key>"            # 支撑忠实度评判器
```

### 3. 路径配置

代码中不含任何绝对路径，全部相对本仓库解析，各根目录均可覆盖：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `MIND_PROJECT_ROOT` | 当前目录 | 运行产出：日志、checkpoint |
| `MIND_DATA_ROOT` | `./data` | 数据集与 PRB 产物 |
| `MIND_MODEL_ROOT` | `./models` | 本地模型权重 |
| `MIND_LOG_ROOT` | `./logs/readable` | 可读轨迹调试日志 |
| `MIND_BGE_MODEL_PATH` | `BAAI/bge-large-zh-v1.5` | 检索编码器 |
| `MIND_CONDA_SH` | 未设置（跳过激活） | 训练脚本使用的 conda 初始化脚本 |
| `MIND_CONDA_ENV` | `mind` | 训练脚本激活的 conda 环境名 |

### 4. 安全默认值

以下两项默认关闭，除非有明确且已审阅的理由，否则请保持关闭：

- **`trust_remote_code` 默认为 `False`**。开启后 `from_pretrained` 会执行模型仓库自带的代码；Qwen3 并不需要它。若确需对某个已审阅过的模型开启：`export MIND_TRUST_REMOTE_CODE=1`。
- **checkpoint 以 `weights_only=True` 加载**，PRB 索引改用 JSONL 而非 pickle 存储。这两条反序列化路径原本都可能从文件中执行任意代码。遗留的 pickle 索引会被拒绝加载，需用 `mind_v2/scripts/prb/convert_legacy_index.py` 或 `scripts/convert_legacy_rag_index.py` 显式转换，且仅限你自己产出的文件。

### 5. 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|---|---|---|
| GPU | 4× A100 (40GB) | 8× A100 (80GB) |
| 内存 | 128GB | 256GB |
| 存储 | 50GB | 200GB |

### 6. 基座模型

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)：MIND-8B 医生智能体基座，同时作为固定权重的患者智能体
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)：MIND-4B 医生智能体基座

## 数据集

**🤗 [Lyncia/LingxiDiag-16K](https://huggingface.co/datasets/Lyncia/LingxiDiag-16K)** —— 合成精神科问诊对话，配对 EMR 派生的患者档案（中文，CC BY-NC 4.0）。

```python
from datasets import load_dataset

ds = load_dataset("Lyncia/LingxiDiag-16K")
```

下载后放置于 `data/` 目录（或用 `MIND_DATA_ROOT` 指向别处）。测试划分全程隔离：PRB 仅从训练划分构建。

## 实验脚本

### 1. 数据预处理

```bash
# 从 EMR 重建患者档案并生成对话
python scripts/data_process/extract_medical_data.py
python scripts/data_process/convert_dialog_format.py

# 生成训练/验证划分
python scripts/data_process/split_train_val.py
```

### 2. 构建精神科推理库（PRB）

```bash
# 构建临床检索状态（仅使用训练划分）
python -m mind_v2.scripts.prb.build_clinical_states --split train --out data/prb/states.jsonl

# 蒸馏标准对齐支撑（--builder_llm 选择教师模型，如 kimi-k2）
python -m mind_v2.scripts.prb.build_with_llm --builder_llm kimi-k2 \
    --in data/prb/states.jsonl --out data/prb/supports.jsonl

# LLM 评判器可靠性打分（1–5）与过滤
python -m mind_v2.scripts.prb.judge_reliability \
    --in data/prb/supports.jsonl --out data/prb/supports_scored.jsonl

# 构建 ANN 索引
python -m mind_v2.scripts.prb.build_index \
    --in data/prb/supports_scored.jsonl --out data/prb/index/
```

### 3. 训练医生智能体

**阶段一：SFT 冷启动**（Kimi-K2 蒸馏，DeepSeek-R1 风格）

| 超参数 | 值 |
|---|---|
| 训练方法 | LoRA |
| LoRA rank | 64 |
| LoRA alpha | 32 |
| 学习率 | 1e-4 |
| Warmup ratio | 0.03 |

```bash
bash mind_v2/scripts/training/train_sft.sh
```

**阶段二：GRPO 强化学习**

| 超参数 | 值 |
|---|---|
| 算法 | GRPO |
| 计算资源 | 8× NVIDIA A100 |
| Actor 学习率 | 5e-6 |
| KL 系数 | 0.02 |
| Clip ratio | 0.10–0.18 |
| 诊断准确性权重 | 5.0 |
| 信息增益权重 | 0.005 |
| 临床推理权重 | 0.01 |
| 格式合规权重 | 0.1 |

```bash
# 与论文对齐的 GRPO 训练
bash mind_v2/scripts/training/train_grpo.sh

# 原始 ragen/veRL 栈（PPO/GRPO 基线；需先设置 WANDB_API_KEY）
bash scripts/training/train_ppo_baseline.sh

# 从检查点续训
bash scripts/training/train_ppo_resume.sh
```

训练过程可通过 WandB 监控，最大对话轮次 $L=10$。

### 4. 运行评估

```bash
# 主表：两种患者模拟器下的端到端推理
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_std
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_adapt

# 支撑忠实度（FC / SG / PF），DeepSeek 评判器
python -m mind_v2.scripts.evaluation.run_faithfulness \
    --results results/run.jsonl --judge deepseek_v32

# 自动 MCR 评估
python -m mind_v2.scripts.evaluation.run_mcr --results results/run.jsonl

# O / M / D / S / R 字段消融
python -m mind_v2.scripts.evaluation.run_field_masking --ckpt path/to/MIND-8B
```

原始栈的评估入口同样可用：

```bash
python ragen/env/med_dialogue/evaluation/inference_fast_for_patientllm_zh_1018_3_best.py \
    --model_path /path/to/checkpoint \
    --data_path data/test.parquet \
    --output_dir results/

python ragen/env/med_dialogue/evaluation/evaluation_for_patientllm_category_zh_optimized_best.py \
    --result_path results/inference_output.json
```

### 5. 检查点转换

```bash
# 将分布式训练检查点转换为 HuggingFace 格式
bash scripts/convert_best_checkpoint.sh
```

## 引用

如果 MIND 对您的研究有所帮助，请引用我们的工作：

MIND 已被 **EMNLP 2026** 录用。

```bibtex
@inproceedings{li2026mind,
  title={MIND: Unified Inquiry and Diagnosis RL with Criteria Grounded Clinical Supports for Psychiatric Consultation},
  author={Li, Guoyi and Xu, Shihao and Ma, Jiatong and Han, Yunyun and Chen, Jianhua and Deng, Yafeng},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026},
  url={https://arxiv.org/abs/2603.03677}
}
```

## 致谢

- [veRL](https://github.com/volcengine/verl) —— GRPO 训练基础设施
- [vLLM](https://github.com/vllm-project/vllm) —— 高效 LLM 推理引擎
- [Ray](https://github.com/ray-project/ray) —— 分布式计算框架
- [Qwen3](https://huggingface.co/Qwen) —— 基座语言模型（Qwen3-4B/8B 用于医生智能体，Qwen3-8B 用于患者智能体）
- [Kimi-K2](https://arxiv.org/abs/2507.20534) —— PRB 构建与 SFT 蒸馏教师模型

## 许可协议

本项目代码以 **Apache License 2.0** 发布，见 [`LICENSE`](LICENSE)。内嵌的 `verl/` 目录沿用其自带的 Apache-2.0 许可（见 [`verl/LICENSE`](verl/LICENSE)）。**LingxiDiag-16K** 数据集以 CC BY-NC 4.0 发布，仅限非商业使用。

> **预期用途**：MIND 是用于研究多轮临床问询与诊断推理的科研成果，**不是**医疗器械，不得用于真实患者的诊断或治疗。
