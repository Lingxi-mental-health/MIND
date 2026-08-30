# MIND v2 — 与 ACL 投稿对齐的精神科问诊 RL 框架

本目录是对 `MIND/`（基于 RAGEN/veRL 的原始实现）的"对齐论文"重构版本。
代码完全按照 `latex/acl_latex.tex` 中的方法论与实验设置组织，便于复现论文 Table~1/3/4/5/6/11/12/13 与附录 D--K 的所有实验。

> 设计目标：在不破坏 `MIND/ragen/`、`MIND/verl/` 既有训练栈的前提下，提供一份**自包含、与论文一一对应**的实现与脚手架。
> 静态对齐版本（仅做语法/导入级校验，未运行）。

## 论文 ↔ 代码索引

| 论文位置 | 代码位置 |
|----------|----------|
| §3 Methodology · Evidence-state interface $\mathcal{E}_t=(\mathcal{O},\mathcal{M},\mathcal{D},\mathcal{S},\mathcal{R})$ | `mind/evidence_state/state.py`、`parser.py` |
| §3.1 PRB · 离线构建/支撑合成/可靠性评估 | `mind/prb/build.py`、`prb/synthesis.py`、`prb/judge.py` |
| §3.1 PRB · 状态构造算子 $g_{\rm PRB}(q_t)\to(\mathcal{S}_t,\mathcal{R}_t)$ | `mind/prb/state_operator.py` |
| §3.2 Stage I/II 两阶段格式 `<rag_query>` / `<think><answer>` | `mind/env/mind_env.py`、`mind/utils/format.py` |
| §3.2 过程奖励 $S^{\rm sym}/S^{\rm diff}/S^{\rm dec}$（Table 12 rubric） | `mind/reward/process_reward.py` |
| §3.2 信息增益奖励（reduce $\|\mathcal{M}_t\|$） | `mind/reward/info_gain.py` |
| §3.2 RL 目标 $J(\theta)=\mathbb{E}[\sum r_t + r^{\rm term}]$、奖励权重（Table 11） | `mind/reward/aggregator.py`、`configs/rl_grpo.yaml` |
| §3.2 状态条件轨迹纠正（Table 13 触发器） | `mind/rectification/triggers.py`、`rectifier.py` |
| §4 Experiments · PsySim-Std / PsySim-Adapt 患者模拟器 | `mind/env/patient_sim.py` |
| §4 支撑忠实度评估（FC/SG/PF） | `mind/evaluation/support_faithfulness.py` |
| 附录 D 自动 MCR 评估 | `mind/evaluation/mcr.py` |
| 附录 G 字段消融（masking $\mathcal{O}/\mathcal{M}/\mathcal{D}/\mathcal{S}/\mathcal{R}$） | `mind/evaluation/field_masking.py` |
| 附录 H Evidence-state 构造质量审计 | `mind/evaluation/state_construction_quality.py` |
| 附录 J 参数分析（top-$k$、$L$、$\tau_{\rm rel}$） | `mind/evaluation/parameter_analysis.py` |
| 附录 F PRB 构建 LLM 敏感性 | `scripts/prb/build_with_llm.py` 通过 `--builder_llm` 切换 |
| 附录 表 11 SFT 超参 / 表 12 RL 超参 / 表 13 推理超参 | `configs/sft_lora.yaml` / `configs/rl_grpo.yaml` / `configs/inference.yaml` |

## 端到端工作流

```bash
# 0. 准备：将 MIND/data/*.parquet 软链到 mind_v2/data/
mkdir -p mind_v2/data && ln -sf $(pwd)/MIND/data/* mind_v2/data/

# 1. 构建 PRB（仅来自 train 划分；test EMR 全程隔离）
python -m mind_v2.scripts.prb.build_clinical_states  --split train  --out data/prb/states.jsonl
python -m mind_v2.scripts.prb.build_with_llm         --builder_llm kimi-k2  --in data/prb/states.jsonl --out data/prb/supports.jsonl
python -m mind_v2.scripts.prb.judge_reliability      --in data/prb/supports.jsonl --out data/prb/supports_scored.jsonl
python -m mind_v2.scripts.prb.build_index            --in data/prb/supports_scored.jsonl --out data/prb/index/

# 2. SFT 冷启动（见论文 Table 11）
bash mind_v2/scripts/training/train_sft.sh

# 3. GRPO 强化学习（见论文 Table 12，奖励权重 5.0/0.005/0.01/0.1）
bash mind_v2/scripts/training/train_grpo.sh

# 4. 评估
#   PsySim-Std / PsySim-Adapt 主表
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_std
bash mind_v2/scripts/evaluation/eval_patientsim.sh psysim_adapt
#   附录 D MCR；附录 H 状态构造质量；§4 Support Faithfulness（FC/SG/PF）
python -m mind_v2.scripts.evaluation.run_mcr            --results results/run.jsonl
python -m mind_v2.scripts.evaluation.run_faithfulness   --results results/run.jsonl --judge deepseek_v32
python -m mind_v2.scripts.evaluation.run_field_masking  --ckpt path/to/MIND-8B
```

## 与原 `MIND/ragen/` 的关系

- mind_v2 重用 `MIND/ragen/env/med_dialogue/rag_retriever.py` 的向量索引能力，但只把它视作**底层 ANN**；状态构造、可靠性门控、字段化输出全部在 `mind/prb/state_operator.py` 内完成。
- mind_v2 不直接修改 `MIND/verl/` 与 `MIND/ragen/trainer/`；GRPO 训练入口包装在 `mind/training/grpo_runner.py`，通过传入 `MindEnv` 与 `MindRewardManager` 调起原 `ragen/trainer/main_ppo.py`。
- 任何 OpenRouter / DeepSeek / GPT-4o / Kimi-K2 调用都集中在 `mind/utils/llm_client.py`，便于离线/在线切换与缓存。
