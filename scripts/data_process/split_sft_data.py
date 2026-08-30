#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分割SFT数据脚本：
1. 提取包含诊断结果的数据
2. 提取前N条数据
"""

import json
from pathlib import Path
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")


def split_sft_data(input_file: str, output_dir: str, first_n: int = 20000):
    """
    分割SFT数据
    
    Args:
        input_file: 输入的SFT数据文件
        output_dir: 输出目录
        first_n: 提取前N条数据
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
        # 方法1: 检查 extra_info 中的 is_final 标记
        # 方法2: 检查 response 中是否包含 "诊断："
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
    
    # 2. 提取前N条数据
    print(f"\n=== 提取前{first_n}条数据 ===")
    first_n_data = data[:first_n]
    print(f"实际提取数据量: {len(first_n_data)}")
    
    # 统计前N条数据的分布
    first_n_diagnosis_stats = {}
    first_n_turn_stats = {}
    for item in first_n_data:
        diagnosis = item['reward_model']['ground_truth']['diagnosis']
        turn = item['extra_info']['turn']
        first_n_diagnosis_stats[diagnosis] = first_n_diagnosis_stats.get(diagnosis, 0) + 1
        first_n_turn_stats[turn] = first_n_turn_stats.get(turn, 0) + 1
    
    print("诊断类别分布:")
    for diagnosis, count in first_n_diagnosis_stats.items():
        print(f"  {diagnosis}: {count}")
    
    print("\n对话轮次分布（前10轮）:")
    for turn in sorted(list(first_n_turn_stats.keys())[:10]):
        print(f"  第 {turn} 轮: {first_n_turn_stats[turn]} 条")
    
    # 保存前N条数据
    output_first_n_file = f"{output_dir}/sft_first_{first_n}.json"
    print(f"\n保存前{first_n}条数据到: {output_first_n_file}")
    with open(output_first_n_file, 'w', encoding='utf-8') as f:
        json.dump(first_n_data, f, ensure_ascii=False, indent=2)
    print(f"成功保存 {len(first_n_data)} 条数据")
    
    # 3. 输出统计摘要
    print("\n=== 统计摘要 ===")
    print(f"原始数据总量: {len(data)}")
    print(f"诊断数据量: {len(diagnosis_data)} ({len(diagnosis_data)/len(data)*100:.2f}%)")
    print(f"前{first_n}条数据量: {len(first_n_data)}")
    print(f"\n生成的文件:")
    print(f"  1. {output_diagnosis_file}")
    print(f"  2. {output_first_n_file}")

def main():
    # 设置文件路径
    input_file = os.path.join(DATA_ROOT, "storehouse/dialog_20000_sft_multiturn_data.json")
    output_dir = os.path.join(DATA_ROOT, "storehouse")
    first_n = 20000
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 执行分割
    split_sft_data(input_file, output_dir, first_n)

if __name__ == "__main__":
    main()






