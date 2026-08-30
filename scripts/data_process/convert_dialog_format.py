#!/usr/bin/env python3
"""
数据格式转换脚本：将 dialog_sft_multiturn_data_with_extra_info_updated_with_thinking.parquet 
转换为与 MTMedDialog_sft_train.parquet 相同的格式，以便用于训练。
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")



def convert_dialog_format(input_file, output_file):
    """
    转换对话数据格式
    
    Args:
        input_file (str): 输入parquet文件路径
        output_file (str): 输出parquet文件路径
    """
    print(f"正在读取数据文件: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"原始数据量: {len(df)} 行")
    
    converted_data = []
    
    for idx, row in df.iterrows():
        try:
            # 解析 prompt 字段（JSON字符串）
            prompt_data = json.loads(row['prompt'])
            
            # 确保 prompt_data 是列表格式
            if isinstance(prompt_data, list):
                # 转换为 numpy array 格式
                prompt_array = np.array(prompt_data, dtype=object)
            else:
                # 如果不是列表，包装成列表
                prompt_array = np.array([prompt_data], dtype=object)
            
            # 创建转换后的行
            converted_row = {
                'data_source': row['data_source'],
                'prompt': prompt_array,
                'reward_model': row['reward_model'],
                'extra_info': row['extra_info'],
                'response': row['response']
            }
            
            converted_data.append(converted_row)
            
        except Exception as e:
            print(f"警告: 处理第 {idx} 行时出错: {e}")
            print(f"Prompt 内容: {row['prompt'][:100]}...")
            continue
    
    # 创建新的 DataFrame
    converted_df = pd.DataFrame(converted_data)
    print(f"转换后数据量: {len(converted_df)} 行")
    
    # 保存转换后的数据
    converted_df.to_parquet(output_file, index=False)
    print(f"转换后的数据已保存到: {output_file}")
    
    # 验证转换结果
    print("\n验证转换结果:")
    print(f"列名: {list(converted_df.columns)}")
    print(f"Prompt 类型: {type(converted_df['prompt'].iloc[0])}")
    print(f"Response 类型: {type(converted_df['response'].iloc[0])}")
    
    # 显示一个样本
    if len(converted_df) > 0:
        print(f"\n样本数据:")
        print(f"Prompt 样本: {converted_df['prompt'].iloc[0]}")
        print(f"Response 样本: {converted_df['response'].iloc[0][:200]}...")
    
    return converted_df


def main():
    parser = argparse.ArgumentParser(description='转换对话数据格式以适配训练器')
    parser.add_argument('--input_file', 
                       default=os.path.join(DATA_ROOT, 'dialog_sft_multiturn_data_with_extra_info_updated_with_thinking_train.parquet'),
                       help='输入parquet文件路径')
    parser.add_argument('--output_file', 
                       default=os.path.join(DATA_ROOT, 'dialog_sft_multiturn_data_with_extra_info_updated_with_thinking_train_converted.parquet'),
                       help='输出parquet文件路径')
    
    args = parser.parse_args()
    
    try:
        converted_df = convert_dialog_format(
            input_file=args.input_file,
            output_file=args.output_file
        )
        print(f"\n✅ 数据格式转换完成！")
        print(f"输出文件: {args.output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


