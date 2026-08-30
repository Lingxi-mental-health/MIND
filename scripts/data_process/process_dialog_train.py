#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理dialog_train.csv数据脚本：
1. 将CSV转换为JSON，只保留OverallDiagnosis_smhc有值的数据
2. 从每个诊断类别随机抽取25个样本
3. 转换为多轮SFT格式，参考convert_to_multiturn_sft.py
4. 在extra_info中添加patient_id_mdd字段
"""

import json
import re
import random
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

def load_and_filter_csv(csv_file: str) -> pd.DataFrame:
    """
    加载CSV文件并过滤OverallDiagnosis_smhc有值的数据
    """
    print(f"开始读取CSV文件: {csv_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"原始数据总数: {len(df)}")
    
    # 过滤OverallDiagnosis_smhc有值的数据
    filtered_df = df.dropna(subset=['OverallDiagnosis_smhc'])
    print(f"过滤后有OverallDiagnosis_smhc值的数据: {len(filtered_df)}")
    
    # 显示各诊断类别的数量
    diagnosis_counts = filtered_df['OverallDiagnosis_smhc'].value_counts()
    print("各诊断类别数量:")
    for diagnosis, count in diagnosis_counts.items():
        print(f"  {diagnosis}: {count}")
    
    return filtered_df

def sample_balanced_data(df: pd.DataFrame, target_diagnoses: List[str], samples_per_diagnosis: int = 25) -> pd.DataFrame:
    """
    从每个诊断类别随机抽取指定数量的样本，确保每个patient_id唯一
    """
    print(f"\n开始从每个诊断类别抽取{samples_per_diagnosis}个样本（确保patient_id唯一）...")
    
    sampled_data = []
    
    for diagnosis in target_diagnoses:
        diagnosis_data = df[df['OverallDiagnosis_smhc'] == diagnosis]
        
        # 检查唯一患者数量
        unique_patients = diagnosis_data['patient_id_mdd'].nunique()
        total_records = len(diagnosis_data)
        print(f"  {diagnosis}: 总记录数={total_records}, 唯一患者数={unique_patients}")
        
        if unique_patients < samples_per_diagnosis:
            print(f"警告: {diagnosis}类别只有{unique_patients}个唯一患者，少于要求的{samples_per_diagnosis}个")
            # 如果唯一患者不足，则每个患者只取一条记录
            unique_patient_data = diagnosis_data.drop_duplicates(subset=['patient_id_mdd'], keep='first')
            sampled_data.append(unique_patient_data)
            print(f"  {diagnosis}: 抽取了{len(unique_patient_data)}个样本（所有唯一患者）")
        else:
            # 先按patient_id去重，每个患者随机选择一条记录
            unique_patient_data = diagnosis_data.groupby('patient_id_mdd').apply(
                lambda x: x.sample(n=1, random_state=42)
            ).reset_index(drop=True)
            
            # 然后从唯一患者中随机抽取指定数量
            sampled = unique_patient_data.sample(n=samples_per_diagnosis, random_state=42)
            sampled_data.append(sampled)
            print(f"  {diagnosis}: 抽取了{len(sampled)}个样本（{len(sampled)}个唯一患者）")
    
    # 合并所有抽样数据
    result_df = pd.concat(sampled_data, ignore_index=True)
    print(f"总共抽取样本数: {len(result_df)}")
    print(f"总唯一患者数: {result_df['patient_id_mdd'].nunique()}")
    
    return result_df

def parse_dialogue(cleaned_text: str) -> List[Dict[str, str]]:
    """
    解析cleaned_text中的对话，提取医生和患者的对话轮次
    参考convert_to_multiturn_sft.py的逻辑
    
    Args:
        cleaned_text: 原始对话文本
        
    Returns:
        对话轮次列表，每个元素包含role和content
    """
    # 按"未知发言人："分割对话
    dialogue_parts = cleaned_text.split("未知发言人：")
    dialogue_parts = [part.strip() for part in dialogue_parts if part.strip()]
    
    conversations = []
    for i, part in enumerate(dialogue_parts):
        # 医生先说话(偶数索引)，患者回复(奇数索引)
        if i % 2 == 0:
            role = "doctor"
        else:
            role = "patient"
            
        # 清理文本，移除多余的换行和空格
        content = re.sub(r'\n+', ' ', part).strip()
        conversations.append({
            "role": role,
            "content": content
        })
    
    return conversations

def create_system_prompt() -> str:
    """
    创建系统提示词，定义医生的角色和任务
    """
    return """你是一名经验丰富的精神科医生，需要通过问诊为患者提供专业的诊断和建议。请仔细倾听患者的描述，通过有针对性的提问收集充分信息，然后给出准确的诊断和合适的治疗建议。

目标：
1. 通过有效提问获取关键信息，每轮问题应基于前一轮内容进行调整，避免重复类似问题
2. 全面分析患者病情，提供准确的诊断和合适的治疗建议

规则：
1. 你只能选择一种回应方式，不能同时提问和给出诊断
2. 绝对不要重复或询问与之前类似或相同的问题

回应格式：
<think> [你的思考过程] </think>
<answer>如果信息不足，请只提一个问题，格式如下：
问题：(你的问题)
</answer> | <answer>如果信息充足，请只提供诊断和建议，格式如下：
诊断：(患者最可能的疾病或症状)
建议：(相应的治疗方案或建议)
</answer>"""

def extract_patient_initial_description(conversations: List[Dict[str, str]]) -> str:
    """
    提取患者的初始描述 - 统一使用"您好"作为初始描述
    """
    return "您好"

def create_multiturn_sft_data(row: pd.Series, conversations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    从单个对话创建多轮SFT训练数据
    每一轮对话生成一个训练样本
    """
    system_prompt = create_system_prompt()
    sft_samples = []
    
    if not conversations:
        return sft_samples
    
    # 获取行数据
    visit_number = str(row.get('VisitNumber', ''))
    patient_id_mdd = str(row.get('patient_id_mdd', ''))
    diagnosis = str(row.get('OverallDiagnosis_smhc', ''))
    # 尝试获取治疗建议，如果没有则使用空字符串
    recommendation = str(row.get('TreatmentRecommendation', ''))
    if pd.isna(recommendation) or recommendation == 'nan':
        recommendation = ""
    
    # 构建渐进式的prompt和response
    current_prompt = [{"content": system_prompt, "role": "system"}]
    
    # 添加初始患者问候作为第一个user输入
    initial_description = extract_patient_initial_description(conversations)
    current_prompt.append({
        "content": f"{initial_description}\n请决定下一步行动：\n总是输出：<think> [你的思考] </think> <answer> [你的回应] </answer> 不要额外文字。严格遵循此格式。",
        "role": "user"
    })
    
    # 处理每一轮对话
    turn_index = 0
    for i in range(0, len(conversations), 2):
        # 每轮：医生问题 -> 患者回答
        if i < len(conversations) and conversations[i]["role"] == "doctor":
            doctor_question = conversations[i]["content"]
            
            # 创建训练样本：当前prompt -> 医生回复
            sft_samples.append({
                "data_source": "medical_consultation",
                "prompt": current_prompt.copy(),
                "response": doctor_question,
                "reward_model": {
                    "ground_truth": {
                        "diagnosis": diagnosis,
                        "recommendation": recommendation
                    }
                },
                "extra_info": {
                    "index": turn_index,
                    "visit_number": visit_number,
                    "turn": turn_index + 1,
                    "patient_id_mdd": patient_id_mdd  # 添加patient_id_mdd字段
                }
            })
            
            # 更新prompt：加入医生问题
            current_prompt.append({"content": doctor_question, "role": "assistant"})
            
            # 如果有患者回答，加入到prompt中
            if i + 1 < len(conversations) and conversations[i + 1]["role"] == "patient":
                patient_answer = conversations[i + 1]["content"]
                current_prompt.append({"content": patient_answer, "role": "user"})
            
            turn_index += 1
    
    # 如果有诊断和建议，且最后没有诊断，添加一个最终诊断样本
    if sft_samples and diagnosis and not any("诊断：" in sample["response"] for sample in sft_samples):
        final_response = f"诊断：{diagnosis}"
        if recommendation:
            final_response += f"\n建议：{recommendation}"
            
        sft_samples.append({
            "data_source": "medical_consultation",
            "prompt": current_prompt.copy(),
            "response": final_response,
            "reward_model": {
                "ground_truth": {
                    "diagnosis": diagnosis,
                    "recommendation": recommendation
                }
            },
            "extra_info": {
                "index": turn_index,
                "visit_number": visit_number,
                "turn": turn_index + 1,
                "patient_id_mdd": patient_id_mdd,  # 添加patient_id_mdd字段
                "is_final": True
            }
        })
    
    return sft_samples

def process_dialog_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    处理DataFrame数据，转换为多轮SFT格式
    """
    print("\n开始处理对话数据...")
    
    all_sft_samples = []
    
    for idx, row in df.iterrows():
        try:
            cleaned_text = row.get('cleaned_text', '')
            
            if pd.isna(cleaned_text) or not cleaned_text:
                print(f"跳过空对话: {row.get('VisitNumber', '')}")
                continue
            
            # 解析对话
            conversations = parse_dialogue(cleaned_text)
            
            if len(conversations) < 2:
                print(f"跳过对话轮次不足的数据: {row.get('VisitNumber', '')}")
                continue
            
            # 创建多轮SFT数据
            sft_samples = create_multiturn_sft_data(row, conversations)
            all_sft_samples.extend(sft_samples)
            
            if (idx + 1) % 10 == 0:
                print(f"已处理对话: {idx + 1}/{len(df)}, 生成样本: {len(all_sft_samples)}")
                
        except Exception as e:
            print(f"处理数据时出错 {row.get('VisitNumber', '')}: {str(e)}")
            continue
    
    print(f"转换完成，共生成 {len(all_sft_samples)} 条SFT格式数据")
    return all_sft_samples

def main():
    """主函数"""
    # 文件路径
    csv_file = "/dialog_train.csv"
    output_json = "/data_process/filtered_dialog_data.json"
    output_sft_json = "/data_process/dialog_sft_multiturn_data.json"
    
    # 目标诊断类别
    target_diagnoses = ['Anxiety', 'Depression', 'Mix', 'Other']
    samples_per_diagnosis = 25
    
    # 确保输出目录存在
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # 第一步：加载和过滤CSV数据
    print("=== 第一步：加载和过滤CSV数据 ===")
    df = load_and_filter_csv(csv_file)
    
    # 第二步：平衡抽样
    print("\n=== 第二步：平衡抽样 ===")
    sampled_df = sample_balanced_data(df, target_diagnoses, samples_per_diagnosis)
    
    # 保存过滤后的JSON数据
    print(f"\n保存过滤后的数据到: {output_json}")
    sampled_df.to_json(output_json, orient='records', force_ascii=False, indent=2)
    
    # 第三步：转换为SFT格式
    print("\n=== 第三步：转换为多轮SFT格式 ===")
    sft_data = process_dialog_data(sampled_df)
    
    # 保存SFT数据
    print(f"\n保存SFT数据到: {output_sft_json}")
    with open(output_sft_json, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    # 显示样本统计
    if sft_data:
        print("\n=== 样本统计 ===")
        print(f"总样本数: {len(sft_data)}")
        
        # 按诊断类别统计
        diagnosis_stats = {}
        for sample in sft_data:
            diagnosis = sample['reward_model']['ground_truth']['diagnosis']
            diagnosis_stats[diagnosis] = diagnosis_stats.get(diagnosis, 0) + 1
        
        print("按诊断类别统计:")
        for diagnosis, count in diagnosis_stats.items():
            print(f"  {diagnosis}: {count}")
        
        # 显示第一个样本
        print("\n=== 样本展示 ===")
        sample = sft_data[0]
        print(f"Visit Number: {sample['extra_info']['visit_number']}")
        print(f"Patient ID MDD: {sample['extra_info']['patient_id_mdd']}")
        print(f"Turn: {sample['extra_info']['turn']}")
        print(f"Prompt length: {len(sample['prompt'])}")
        print(f"Response: {sample['response'][:150]}...")
        print(f"Diagnosis: {sample['reward_model']['ground_truth']['diagnosis']}")

if __name__ == "__main__":
    main()
