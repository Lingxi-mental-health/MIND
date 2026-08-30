#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据visit_number将CSV文件中的enhanced_description填充到JSON文件中
"""

import pandas as pd
import json
import sys
from tqdm import tqdm
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")


def load_csv_data(csv_path):
    """加载CSV文件并创建visit_number到主诉的映射"""
    print("正在加载CSV文件...")
    df = pd.read_csv(csv_path)
    
    # 创建visit_number到主诉的映射
    visit_to_description = {}
    
    for _, row in df.iterrows():
        visit_number = row.get('VisitNumber')  # 使用VisitNumber列而不是visit_number列
        chief_complaint = row.get('chief_complaint')  # 使用chief_complaint列
        
        if pd.notna(visit_number) and pd.notna(chief_complaint):
            # 提取主诉内容，去掉JSON格式包装
            chief_complaint_str = str(chief_complaint).strip()
            
            # 如果是JSON格式，提取ChiefComplaint的值
            if chief_complaint_str.startswith('{"ChiefComplaint":') and chief_complaint_str.endswith('}'):
                try:
                    import json
                    chief_data = json.loads(chief_complaint_str)
                    chief_complaint_str = chief_data.get('ChiefComplaint', chief_complaint_str)
                except:
                    pass  # 如果解析失败，使用原始字符串
            
            # 去掉"主诉："前缀（如果存在）
            if chief_complaint_str.startswith('主诉：'):
                chief_complaint_str = chief_complaint_str[3:].strip()
            
            visit_to_description[str(visit_number)] = chief_complaint_str
    
    print(f"从CSV文件中加载了 {len(visit_to_description)} 条visit_number映射")
    return visit_to_description

def update_json_data(json_path, visit_to_description, output_json_path, output_parquet_path):
    """更新JSON文件，添加主诉信息到用户消息中"""
    print("正在加载JSON文件...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    not_found_count = 0
    
    print("正在更新JSON数据...")
    for item in tqdm(data, desc="处理数据"):
        # 获取visit_number
        extra_info = item.get('extra_info', {})
        visit_number = extra_info.get('visit_number')
        
        if visit_number and visit_number in visit_to_description:
            # 去掉指定的字段
            fields_to_remove = ['personal_history', 'chief_complaint', 'family_history', 'physical_condition', 'enhanced_description']
            for field in fields_to_remove:
                if field in extra_info:
                    del extra_info[field]
            
            item['extra_info'] = extra_info
            
            # 修改用户消息，在开头添加主诉信息
            prompt = item.get('prompt', [])
            for msg in prompt:
                if msg.get('role') == 'user':
                    original_content = msg.get('content', '')
                    if original_content.startswith('您好'):
                        # 在"您好"前面添加主诉信息
                        chief_complaint = visit_to_description[visit_number]
                        new_content = f"病人信息：{chief_complaint}\n\n{original_content}"
                        msg['content'] = new_content
                        break
            
            updated_count += 1
        else:
            not_found_count += 1
    
    print(f"成功更新了 {updated_count} 条数据")
    print(f"未找到匹配的visit_number: {not_found_count} 条")
    
    # 保存更新后的JSON文件
    print(f"正在保存JSON到 {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 转换为DataFrame并保存为Parquet格式
    print(f"正在保存Parquet到 {output_parquet_path}...")
    df = pd.DataFrame(data)
    df.to_parquet(output_parquet_path, index=False)
    
    print("更新完成！")

def main():
    # 文件路径
    csv_path = os.path.join(DATA_ROOT, "dialog_train_with_patient_info_enhanced.csv")
    json_path = os.path.join(DATA_ROOT, "dialog_sft_multiturn_data_with_extra_info.json")
    output_json_path = os.path.join(DATA_ROOT, "dialog_sft_multiturn_data_with_extra_info_updated_chief_complaint.json")
    output_parquet_path = os.path.join(DATA_ROOT, "dialog_sft_multiturn_data_with_extra_info_updated_chief_complaint.parquet")
    
    try:
        # 加载CSV数据
        visit_to_description = load_csv_data(csv_path)
        
        # 更新JSON数据
        update_json_data(json_path, visit_to_description, output_json_path, output_parquet_path)
        
        print(f"\n处理完成！")
        print(f"原始JSON文件: {json_path}")
        print(f"更新后的JSON文件: {output_json_path}")
        print(f"更新后的Parquet文件: {output_parquet_path}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
