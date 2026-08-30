#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分割SFT数据脚本（平衡版）：
1. 提取包含诊断结果的数据（每个对话的最终诊断）
2. 提取每个诊断类别的前5000条数据，总共20000条
"""

import json
from pathlib import Path
from collections import defaultdict
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")


def split_sft_data_balanced(input_file: str, output_dir: str, samples_per_category: int = 5000):
    """
    分割SFT数据，每个类别取相同数量
    
    Args:
        input_file: 输入的SFT数据文件
        output_dir: 输出目录
        samples_per_category: 每个类别提取的样本数
    """
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}")
    
    # 1. 提取包含诊断结果的数据
    print("\n=== 提取包含诊断结果的数据 ===")
    diagnosis_data = []
    
    for item in data:
        response = item.get('response', '')
        extra_info = item.get('extra_info', {})
        
        # 判断是否包含诊断结果
        if extra_info.get('is_final', False) or '诊断：' in response:
            diagnosis_data.append(item)
    
    print(f"包含诊断结果的数据量: {len(diagnosis_data)}")
    
    # 统计诊断数据的分布
    diagnosis_stats = {}
    for item in diagnosis_data:
        diagnosis = item['reward_model']['ground_truth']['diagnosis']
        diagnosis_stats[diagnosis] = diagnosis_stats.get(diagnosis, 0) + 1
    
    print("诊断类别分布:")
    for diagnosis, count in diagnosis_stats.items():
        print(f"  {diagnosis}: {count}")
    
    # 保存诊断数据
    output_diagnosis_file = f"{output_dir}/sft_diagnosis_only.json"
    print(f"\n保存诊断数据到: {output_diagnosis_file}")
    with open(output_diagnosis_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_data, f, ensure_ascii=False, indent=2)
    print(f"成功保存 {len(diagnosis_data)} 条诊断数据")
    
    # 2. 按诊断类别分组，每个类别提取前N条
    print(f"\n=== 提取每个类别的前{samples_per_category}条数据 ===")
    
    # 按诊断类别分组
    data_by_diagnosis = defaultdict(list)
    for item in data:
        diagnosis = item['reward_model']['ground_truth']['diagnosis']
        data_by_diagnosis[diagnosis].append(item)
    
    print("\n各类别可用数据量:")
    for diagnosis in sorted(data_by_diagnosis.keys()):
        print(f"  {diagnosis}: {len(data_by_diagnosis[diagnosis])} 条")
    
    # 从每个类别提取前N条
    balanced_data = []
    for diagnosis in sorted(data_by_diagnosis.keys()):
        category_data = data_by_diagnosis[diagnosis]
        samples_to_take = min(samples_per_category, len(category_data))
        selected = category_data[:samples_to_take]
        balanced_data.extend(selected)
        print(f"  {diagnosis}: 提取了 {len(selected)} 条")
    
    print(f"\n总共提取: {len(balanced_data)} 条数据")
    
    # 统计提取数据的分布
    balanced_diagnosis_stats = {}
    balanced_turn_stats = {}
    for item in balanced_data:
        diagnosis = item['reward_model']['ground_truth']['diagnosis']
        turn = item['extra_info']['turn']
        balanced_diagnosis_stats[diagnosis] = balanced_diagnosis_stats.get(diagnosis, 0) + 1
        balanced_turn_stats[turn] = balanced_turn_stats.get(turn, 0) + 1
    
    print("\n提取后的诊断类别分布:")
    for diagnosis in sorted(balanced_diagnosis_stats.keys()):
        print(f"  {diagnosis}: {balanced_diagnosis_stats[diagnosis]}")
    
    print("\n提取后的对话轮次分布（前10轮）:")
    for turn in sorted(list(balanced_turn_stats.keys())[:10]):
        print(f"  第 {turn} 轮: {balanced_turn_stats[turn]} 条")
    
    # 保存平衡数据
    output_balanced_file = f"{output_dir}/sft_balanced_{len(balanced_data)}.json"
    print(f"\n保存平衡数据到: {output_balanced_file}")
    with open(output_balanced_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    print(f"成功保存 {len(balanced_data)} 条数据")
    
    # 3. 输出统计摘要
    print("\n=== 统计摘要 ===")
    print(f"原始数据总量: {len(data)}")
    print(f"诊断数据量: {len(diagnosis_data)} ({len(diagnosis_data)/len(data)*100:.2f}%)")
    print(f"平衡数据量: {len(balanced_data)}")
    print(f"每个类别采样数: {samples_per_category}")
    print(f"\n生成的文件:")
    print(f"  1. {output_diagnosis_file} - 所有包含诊断的数据")
    print(f"  2. {output_balanced_file} - 每个类别{samples_per_category}条的平衡数据")

def main():
    # 设置文件路径
    input_file = os.path.join(DATA_ROOT, "storehouse/dialog_20000_sft_multiturn_data.json")
    output_dir = os.path.join(DATA_ROOT, "storehouse")
    samples_per_category = 5000
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 执行分割
    split_sft_data_balanced(input_file, output_dir, samples_per_category)

if __name__ == "__main__":
    main()






