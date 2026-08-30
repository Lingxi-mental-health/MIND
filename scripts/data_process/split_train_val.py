#!/usr/bin/env python3
"""
数据分割脚本：将 dialog_sft_multiturn_data_with_extra_info_updated_with_thinking.parquet 
分割为训练集和验证集，其中验证集包含100个样本，其余作为训练集。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")



def split_data(input_file, output_dir, val_size=100, random_seed=42):
    """
    分割数据为训练集和验证集
    
    Args:
        input_file (str): 输入parquet文件路径
        output_dir (str): 输出目录
        val_size (int): 验证集大小，默认100
        random_seed (int): 随机种子，默认42
    """
    # 设置随机种子确保可重复性
    np.random.seed(random_seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取数据文件: {input_file}")
    # 读取数据
    df = pd.read_parquet(input_file)
    total_rows = len(df)
    print(f"总数据量: {total_rows} 行")
    
    # 检查验证集大小是否合理
    if val_size >= total_rows:
        raise ValueError(f"验证集大小 ({val_size}) 不能大于等于总数据量 ({total_rows})")
    
    # 随机打乱数据
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 分割数据
    val_df = df_shuffled.iloc[:val_size].reset_index(drop=True)
    train_df = df_shuffled.iloc[val_size:].reset_index(drop=True)
    
    print(f"训练集大小: {len(train_df)} 行")
    print(f"验证集大小: {len(val_df)} 行")
    
    # 保存训练集
    train_file = output_path / "dialog_sft_multiturn_data_with_extra_info_updated_with_thinking_train.parquet"
    train_df.to_parquet(train_file, index=False)
    print(f"训练集已保存到: {train_file}")
    
    # 保存验证集
    val_file = output_path / "dialog_sft_multiturn_data_with_extra_info_updated_with_thinking_val.parquet"
    val_df.to_parquet(val_file, index=False)
    print(f"验证集已保存到: {val_file}")
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    print(f"原始数据列: {list(df.columns)}")
    print(f"训练集列: {list(train_df.columns)}")
    print(f"验证集列: {list(val_df.columns)}")
    
    # 检查数据分布
    if 'data_source' in df.columns:
        print(f"\n原始数据源分布:")
        print(df['data_source'].value_counts())
        print(f"\n训练集数据源分布:")
        print(train_df['data_source'].value_counts())
        print(f"\n验证集数据源分布:")
        print(val_df['data_source'].value_counts())
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description='分割医疗对话数据为训练集和验证集')
    parser.add_argument('--input_file', 
                       default=os.path.join(DATA_ROOT, 'dialog_sft_multiturn_data_with_extra_info_updated_with_thinking.parquet'),
                       help='输入parquet文件路径')
    parser.add_argument('--output_dir', 
                       default=DATA_ROOT,
                       help='输出目录')
    parser.add_argument('--val_size', 
                       type=int, 
                       default=100,
                       help='验证集大小，默认100')
    parser.add_argument('--random_seed', 
                       type=int, 
                       default=42,
                       help='随机种子，默认42')
    
    args = parser.parse_args()
    
    try:
        train_file, val_file = split_data(
            input_file=args.input_file,
            output_dir=args.output_dir,
            val_size=args.val_size,
            random_seed=args.random_seed
        )
        print(f"\n✅ 数据分割完成！")
        print(f"训练集: {train_file}")
        print(f"验证集: {val_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


