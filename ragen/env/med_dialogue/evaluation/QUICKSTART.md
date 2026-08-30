# 快速开始指南 - 优化版评估系统

## 5分钟快速上手

### 步骤1：确认环境

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查已安装的包
pip list | grep -E "torch|transformers|numpy"
```

### 步骤2：配置模型路径

编辑 `evaluation_for_patientllm_category_zh_optimized.py`：

```python
# 修改第17行为你的模型路径
QWEN_MODEL_PATH = "/your/path/to/Qwen3-32B"
```

### 步骤3：准备数据

确保你的数据格式正确：

**simulation_data.json** 示例：
```json
[
  {
    "id": "case_001",
    "self_report": "患者自述...",
    "true_diagnosis": "Anxiety",
    "true_recommendation": "建议进行心理咨询...",
    "extra_info": {...},
    "simulation_dialogue": [
      {
        "role": "doctor",
        "content": "您好，请问哪里不舒服？",
        "tokens": 15
      },
      {
        "role": "patient",
        "content": "我最近总是感到焦虑..."
      },
      ...
    ],
    "total_turns": 10
  }
]
```

### 步骤4：运行评估

#### 方式A：使用提供的脚本（推荐）

```bash
# 修改脚本中的数据路径
vim script_eva/run_eval_optimized.sh

# 运行评估
bash script_eva/run_eval_optimized.sh
```

#### 方式B：直接命令行

```bash
python ragen/env/med_dialogue/evaluation/evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data /path/to/simulation_data.json \
    --reference_data /path/to/reference_data.json \
    --output /path/to/output.json \
    --batch_size 16
```

### 步骤5：查看结果

```bash
# 查看结果文件
cat /path/to/output.json | jq '.average_result.diagnostic_performance.combined_score'

# 或者查看完整的平均结果
cat /path/to/output.json | jq '.average_result'
```

## 常见场景

### 场景1：小批量测试（10-50 cases）

```bash
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data test_data_small.json \
    --output test_results.json \
    --batch_size 8
```

**预计时间**：2-5分钟（单GPU）

### 场景2：中等规模评估（100-500 cases）

```bash
# 单GPU
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data_medium.json \
    --output results_medium.json \
    --batch_size 16
```

**预计时间**：10-30分钟（单GPU）

### 场景3：大规模评估（500+ cases）

```bash
# 多GPU加速
torchrun --nproc_per_node=4 \
    evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data_large.json \
    --output results_large.json \
    --batch_size 16
```

**预计时间**：20-60分钟（4×GPU）

## 参数调优指南

### batch_size 选择

| GPU内存 | 推荐batch_size | 说明 |
|---------|---------------|------|
| 8-12GB  | 4-8           | 较小批次 |
| 16-24GB | 8-16          | 中等批次 |
| 32-40GB | 16-32         | 较大批次 |
| 48GB+   | 32-64         | 大批次 |

### GPU数量选择

| 数据规模 | GPU数量 | 预计时间 |
|---------|---------|---------|
| <100    | 1       | 5-10分钟 |
| 100-500 | 1-2     | 10-30分钟 |
| 500-1000| 2-4     | 20-40分钟 |
| 1000+   | 4-8     | 30-60分钟 |

## 故障排除

### 问题1：CUDA Out of Memory

**解决方案**：
```bash
# 减小batch_size
python evaluation_for_patientllm_category_zh_optimized.py \
    --batch_size 4  # 从16降到4
```

### 问题2：模型加载失败

**检查**：
```bash
# 确认模型路径
ls -la /your/path/to/Qwen3-32B

# 应该包含：config.json, pytorch_model.bin 等
```

### 问题3：JSON解析错误

**解决方案**：
- 检查prompt模板格式
- 查看模型输出日志
- 尝试调整temperature参数（在代码中）

### 问题4：评估速度慢

**优化建议**：
1. 增大batch_size（如GPU内存允许）
2. 使用多GPU：`torchrun --nproc_per_node=N`
3. 减少数据集大小进行测试
4. 检查是否使用了FP16推理

## 结果解读

### 关键指标说明

```json
{
  "average_result": {
    "diagnostic_performance": {
      "combined_score": 0.85,  // 综合诊断得分（0-1）
      "diagnosis": {
        "semantic_score": 4.2,  // 诊断语义得分（0-5）
        "classification_f1": {
          "overall_f1": 0.88    // 分类F1得分（0-1）
        }
      }
    },
    "empathy_assessment": {
      "average_empathy_score": 3.8  // 同理心得分（0-5）
    },
    "symptom_retrieval": {
      "retrieval_rate": 0.72  // 症状检索率（0-1）
    },
    "diagnostic_efficiency_index": 0.045  // 诊断效率指数
  }
}
```

### 得分参考

| 指标 | 优秀 | 良好 | 一般 | 需改进 |
|------|------|------|------|--------|
| combined_score | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| empathy_score | >4.0 | 3.0-4.0 | 2.0-3.0 | <2.0 |
| retrieval_rate | >0.7 | 0.5-0.7 | 0.3-0.5 | <0.3 |

## 自定义扩展

### 修改评分标准

编辑 `eval_per_turn_prompt_template.txt` 或 `eval_final_dialogue_prompt_template.txt`：

```
**评估标准：**
同理心得分（0-5分）：
- 5分：充分理解患者的情绪  // 可以修改这里
- 4分：较好地理解患者的情绪
...
```

### 添加新指标

1. 在 `eval_final_dialogue_prompt_template.txt` 添加新任务：
```
## 任务7：评估医生的专业性
...
```

2. 在JSON输出中添加字段：
```json
{
  "diagnosis_semantic_score": 4.5,
  ...
  "professionalism_score": 4.0  // 新增字段
}
```

3. 在代码中处理新字段：
```python
def calculate_metrics_with_final_eval(self, final_eval_result):
    ...
    professionalism = final_eval_result.get("professionalism_score", 0.0)
```

## 进阶使用

### 使用配置文件

创建 `eval_config.json`：
```json
{
  "simulation_data": "/path/to/data.json",
  "reference_data": "/path/to/ref.json",
  "output": "/path/to/output.json",
  "batch_size": 16,
  "alpha": 1.0,
  "beta": 0.01
}
```

修改脚本读取配置：
```python
import json

with open('eval_config.json', 'r') as f:
    config = json.load(f)

results = evaluate_all_cases(**config)
```

### 并行评估多个模型

```bash
#!/bin/bash
models=("model_v1" "model_v2" "model_v3")

for model in "${models[@]}"; do
    echo "Evaluating ${model}..."
    python evaluation_for_patientllm_category_zh_optimized.py \
        --simulation_data "results/${model}/simulation.json" \
        --output "results/${model}/evaluation.json" &
done

wait
echo "All evaluations completed!"
```

## 性能监控

### 实时监控GPU使用

```bash
# 终端1：运行评估
python evaluation_for_patientllm_category_zh_optimized.py ...

# 终端2：监控GPU
watch -n 1 nvidia-smi
```

### 记录评估日志

```bash
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data.json \
    --output results.json \
    2>&1 | tee evaluation.log
```

## 最佳实践

1. **先小后大**：先用小数据集测试，确认无误后再运行大规模评估
2. **保存中间结果**：定期保存评估结果，避免意外中断
3. **监控资源**：实时监控GPU/CPU/内存使用情况
4. **批量处理**：尽量使用大的batch_size以提高效率
5. **多GPU加速**：对于大规模评估，优先考虑多GPU并行

## 获取帮助

```bash
# 查看所有参数说明
python evaluation_for_patientllm_category_zh_optimized.py --help

# 查看详细文档
cat README_OPTIMIZED.md

# 查看优化对比
cat OPTIMIZATION_COMPARISON.md
```

## 下一步

- 阅读 [README_OPTIMIZED.md](README_OPTIMIZED.md) 了解详细功能
- 阅读 [OPTIMIZATION_COMPARISON.md](OPTIMIZATION_COMPARISON.md) 了解优化细节
- 根据需求自定义prompt模板
- 调整参数以优化性能

## 联系支持

如遇到问题，请提供：
1. 错误信息截图
2. 数据集规模
3. GPU型号和内存
4. 使用的参数

祝评估顺利！🎉



## 5分钟快速上手

### 步骤1：确认环境

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查已安装的包
pip list | grep -E "torch|transformers|numpy"
```

### 步骤2：配置模型路径

编辑 `evaluation_for_patientllm_category_zh_optimized.py`：

```python
# 修改第17行为你的模型路径
QWEN_MODEL_PATH = "/your/path/to/Qwen3-32B"
```

### 步骤3：准备数据

确保你的数据格式正确：

**simulation_data.json** 示例：
```json
[
  {
    "id": "case_001",
    "self_report": "患者自述...",
    "true_diagnosis": "Anxiety",
    "true_recommendation": "建议进行心理咨询...",
    "extra_info": {...},
    "simulation_dialogue": [
      {
        "role": "doctor",
        "content": "您好，请问哪里不舒服？",
        "tokens": 15
      },
      {
        "role": "patient",
        "content": "我最近总是感到焦虑..."
      },
      ...
    ],
    "total_turns": 10
  }
]
```

### 步骤4：运行评估

#### 方式A：使用提供的脚本（推荐）

```bash
# 修改脚本中的数据路径
vim script_eva/run_eval_optimized.sh

# 运行评估
bash script_eva/run_eval_optimized.sh
```

#### 方式B：直接命令行

```bash
python ragen/env/med_dialogue/evaluation/evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data /path/to/simulation_data.json \
    --reference_data /path/to/reference_data.json \
    --output /path/to/output.json \
    --batch_size 16
```

### 步骤5：查看结果

```bash
# 查看结果文件
cat /path/to/output.json | jq '.average_result.diagnostic_performance.combined_score'

# 或者查看完整的平均结果
cat /path/to/output.json | jq '.average_result'
```

## 常见场景

### 场景1：小批量测试（10-50 cases）

```bash
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data test_data_small.json \
    --output test_results.json \
    --batch_size 8
```

**预计时间**：2-5分钟（单GPU）

### 场景2：中等规模评估（100-500 cases）

```bash
# 单GPU
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data_medium.json \
    --output results_medium.json \
    --batch_size 16
```

**预计时间**：10-30分钟（单GPU）

### 场景3：大规模评估（500+ cases）

```bash
# 多GPU加速
torchrun --nproc_per_node=4 \
    evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data_large.json \
    --output results_large.json \
    --batch_size 16
```

**预计时间**：20-60分钟（4×GPU）

## 参数调优指南

### batch_size 选择

| GPU内存 | 推荐batch_size | 说明 |
|---------|---------------|------|
| 8-12GB  | 4-8           | 较小批次 |
| 16-24GB | 8-16          | 中等批次 |
| 32-40GB | 16-32         | 较大批次 |
| 48GB+   | 32-64         | 大批次 |

### GPU数量选择

| 数据规模 | GPU数量 | 预计时间 |
|---------|---------|---------|
| <100    | 1       | 5-10分钟 |
| 100-500 | 1-2     | 10-30分钟 |
| 500-1000| 2-4     | 20-40分钟 |
| 1000+   | 4-8     | 30-60分钟 |

## 故障排除

### 问题1：CUDA Out of Memory

**解决方案**：
```bash
# 减小batch_size
python evaluation_for_patientllm_category_zh_optimized.py \
    --batch_size 4  # 从16降到4
```

### 问题2：模型加载失败

**检查**：
```bash
# 确认模型路径
ls -la /your/path/to/Qwen3-32B

# 应该包含：config.json, pytorch_model.bin 等
```

### 问题3：JSON解析错误

**解决方案**：
- 检查prompt模板格式
- 查看模型输出日志
- 尝试调整temperature参数（在代码中）

### 问题4：评估速度慢

**优化建议**：
1. 增大batch_size（如GPU内存允许）
2. 使用多GPU：`torchrun --nproc_per_node=N`
3. 减少数据集大小进行测试
4. 检查是否使用了FP16推理

## 结果解读

### 关键指标说明

```json
{
  "average_result": {
    "diagnostic_performance": {
      "combined_score": 0.85,  // 综合诊断得分（0-1）
      "diagnosis": {
        "semantic_score": 4.2,  // 诊断语义得分（0-5）
        "classification_f1": {
          "overall_f1": 0.88    // 分类F1得分（0-1）
        }
      }
    },
    "empathy_assessment": {
      "average_empathy_score": 3.8  // 同理心得分（0-5）
    },
    "symptom_retrieval": {
      "retrieval_rate": 0.72  // 症状检索率（0-1）
    },
    "diagnostic_efficiency_index": 0.045  // 诊断效率指数
  }
}
```

### 得分参考

| 指标 | 优秀 | 良好 | 一般 | 需改进 |
|------|------|------|------|--------|
| combined_score | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| empathy_score | >4.0 | 3.0-4.0 | 2.0-3.0 | <2.0 |
| retrieval_rate | >0.7 | 0.5-0.7 | 0.3-0.5 | <0.3 |

## 自定义扩展

### 修改评分标准

编辑 `eval_per_turn_prompt_template.txt` 或 `eval_final_dialogue_prompt_template.txt`：

```
**评估标准：**
同理心得分（0-5分）：
- 5分：充分理解患者的情绪  // 可以修改这里
- 4分：较好地理解患者的情绪
...
```

### 添加新指标

1. 在 `eval_final_dialogue_prompt_template.txt` 添加新任务：
```
## 任务7：评估医生的专业性
...
```

2. 在JSON输出中添加字段：
```json
{
  "diagnosis_semantic_score": 4.5,
  ...
  "professionalism_score": 4.0  // 新增字段
}
```

3. 在代码中处理新字段：
```python
def calculate_metrics_with_final_eval(self, final_eval_result):
    ...
    professionalism = final_eval_result.get("professionalism_score", 0.0)
```

## 进阶使用

### 使用配置文件

创建 `eval_config.json`：
```json
{
  "simulation_data": "/path/to/data.json",
  "reference_data": "/path/to/ref.json",
  "output": "/path/to/output.json",
  "batch_size": 16,
  "alpha": 1.0,
  "beta": 0.01
}
```

修改脚本读取配置：
```python
import json

with open('eval_config.json', 'r') as f:
    config = json.load(f)

results = evaluate_all_cases(**config)
```

### 并行评估多个模型

```bash
#!/bin/bash
models=("model_v1" "model_v2" "model_v3")

for model in "${models[@]}"; do
    echo "Evaluating ${model}..."
    python evaluation_for_patientllm_category_zh_optimized.py \
        --simulation_data "results/${model}/simulation.json" \
        --output "results/${model}/evaluation.json" &
done

wait
echo "All evaluations completed!"
```

## 性能监控

### 实时监控GPU使用

```bash
# 终端1：运行评估
python evaluation_for_patientllm_category_zh_optimized.py ...

# 终端2：监控GPU
watch -n 1 nvidia-smi
```

### 记录评估日志

```bash
python evaluation_for_patientllm_category_zh_optimized.py \
    --simulation_data data.json \
    --output results.json \
    2>&1 | tee evaluation.log
```

## 最佳实践

1. **先小后大**：先用小数据集测试，确认无误后再运行大规模评估
2. **保存中间结果**：定期保存评估结果，避免意外中断
3. **监控资源**：实时监控GPU/CPU/内存使用情况
4. **批量处理**：尽量使用大的batch_size以提高效率
5. **多GPU加速**：对于大规模评估，优先考虑多GPU并行

## 获取帮助

```bash
# 查看所有参数说明
python evaluation_for_patientllm_category_zh_optimized.py --help

# 查看详细文档
cat README_OPTIMIZED.md

# 查看优化对比
cat OPTIMIZATION_COMPARISON.md
```

## 下一步

- 阅读 [README_OPTIMIZED.md](README_OPTIMIZED.md) 了解详细功能
- 阅读 [OPTIMIZATION_COMPARISON.md](OPTIMIZATION_COMPARISON.md) 了解优化细节
- 根据需求自定义prompt模板
- 调整参数以优化性能

## 联系支持

如遇到问题，请提供：
1. 错误信息截图
2. 数据集规模
3. GPU型号和内存
4. 使用的参数

祝评估顺利！🎉









