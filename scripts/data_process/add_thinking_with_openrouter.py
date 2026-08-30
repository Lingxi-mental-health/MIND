#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为SFT数据添加思维过程的脚本 - 使用OpenRouter API
使用OpenRouter API (kimi-k2中文模型) 为医生回复添加<think>标签
"""

import json
import os
from openai import OpenAI
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import pandas as pd
import copy

# 创建OpenRouter API客户端
def create_openrouter_client(api_key):
    """创建OpenRouter API客户端"""
    if not api_key:
        raise ValueError("请提供API密钥")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return client

def make_thinking_prompt(ori_prompt, ori_response):
    """
    创建添加思维过程的提示词
    """
    ori_prompt_copy = list(ori_prompt)
    ori_prompt_copy.append({"content": ori_response, "role": "assistant"})
    
    return (
        f"以下是一段医患对话记录：\n{str(ori_prompt_copy)}\n\n"
        f"请帮我为这段医生回复补充思维过程，并修改为 <think> </think><answer> </answer> 的格式。\n"
        f"思维过程应该体现医生的临床推理，包括：\n"
        f"1. 对患者症状的分析\n"
        f"2. 可能的诊断考虑\n"
        f"3. 为什么选择这样的问题或给出这样的诊断\n"
        f"4. 下一步的诊疗思路\n\n"
        f"重要：只返回最后一个assistant的回复内容，格式为：\n"
        f"<think>你的思维过程</think>\n<answer>你的回复内容</answer>\n\n"
        f"不要返回完整的对话列表，不要用json格式，直接输出assistant的回复内容即可。"
    )

def add_thinking_process(case, client, model_name, max_retries=3):
    """
    为单个案例添加思维过程
    """
    
    ori_prompt = case["prompt"]
    ori_response = case["response"]
    
    # 如果已经有思维过程，跳过
    if "<think>" in ori_response and "</think>" in ori_response:
        return case
    
    prompt_text = make_thinking_prompt(ori_prompt, ori_response)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个医学助手，帮助为医生的回复添加思维过程，以提高临床推理的透明度。请严格按照要求的格式输出。"},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            enhanced_content = response.choices[0].message.content.strip()
            
            # 直接使用返回的内容作为新的response，保持prompt不变
            enhanced_case = copy.deepcopy(case)
            enhanced_case["response"] = enhanced_content
            
            return enhanced_case

        except Exception as e:
            print(f"[重试 {attempt+1}] 请求失败，错误: {e}")
            time.sleep(2 + random.uniform(0, 2))  # 添加随机延时避免频率限制

    print(f"[失败] 添加思维过程失败，跳过此条数据")
    return None  # 失败时返回None，而不是原始数据

def fix_multiturn_references(cases):
    """
    保持原始prompt不变，只对数据按turn排序
    """
    from collections import defaultdict
    
    print("开始对数据按turn排序...")
    
    # 按对话分组
    dialogue_groups = defaultdict(list)
    for case in cases:
        visit_number = case.get("extra_info", {}).get("visit_number", "unknown")
        dialogue_groups[visit_number].append(case)
    
    # 为每个对话组排序（按turn顺序）
    for visit_number in dialogue_groups:
        dialogue_groups[visit_number].sort(key=lambda x: x.get("extra_info", {}).get("turn", 0))
    
    print(f"找到 {len(dialogue_groups)} 个对话组")
    
    # 保持原始数据不变，只排序
    final_cases = []
    for visit_number, group in dialogue_groups.items():
        group.sort(key=lambda x: x.get("extra_info", {}).get("turn", 0))
        final_cases.extend(group)
    
    print("数据排序完成")
    return final_cases

def process_all_cases(input_path, output_path, api_key, model_name, start_index=0, max_workers=8):
    """
    处理所有案例，添加思维过程，并自动修复多轮引用
    """
    # 创建客户端
    try:
        client = create_openrouter_client(api_key)
        print(f"OpenRouter API客户端创建成功，使用模型: {model_name}")
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    # 读取数据
    print(f"正在读取数据文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"加载了 {len(cases)} 个案例，从索引 {start_index} 开始... 使用 {max_workers} 线程并行处理")
    
    if start_index >= len(cases):
        print("起始索引超出数据范围")
        return
    
    enhanced_cases = []

    # 如果输出文件已存在，加载已处理的数据
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            enhanced_cases = json.load(f)
        print(f"加载了已存在的 {len(enhanced_cases)} 个处理过的案例")
    except FileNotFoundError:
        print("输出文件不存在，将创建新文件")

    # 计算需要处理的案例
    remaining_cases = cases[start_index + len(enhanced_cases):]
    
    if not remaining_cases:
        print("所有案例都已处理完成")
        # 如果已有数据，仍需修复引用
        if enhanced_cases:
            enhanced_cases = fix_multiturn_references(enhanced_cases)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_cases, f, ensure_ascii=False, indent=2)
            print(f"引用修复完成，保存到: {output_path}")
        return

    print(f"准备处理 {len(remaining_cases)} 个案例（{max_workers}线程并行处理）")
    
    # 使用线程池并行处理
    completed_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_case = {
            executor.submit(add_thinking_process, case, client, model_name): case 
            for case in remaining_cases
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_case), total=len(remaining_cases), desc="添加思维过程"):
            try:
                enhanced = future.result()
                if enhanced is not None:  # 只添加成功生成的数据
                    enhanced_cases.append(enhanced)
                    completed_count += 1

                    # 每处理100个或处理完成时保存一次
                    if completed_count % 100 == 0:
                        # 临时保存，但不修复引用
                        temp_output = output_path.replace('.json', '_temp.json')
                        with open(temp_output, "w", encoding="utf-8") as f:
                            json.dump(enhanced_cases, f, ensure_ascii=False, indent=2)
                        print(f"\n[进度保存] 已成功处理 {completed_count} 个案例，失败 {failed_count} 个")
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f"\n[错误] 案例处理失败: {e}")
    
    print(f"\n处理完成统计：成功 {completed_count} 个，失败 {failed_count} 个")
    
    # 修复多轮引用
    enhanced_cases = fix_multiturn_references(enhanced_cases)
    
    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_cases, f, ensure_ascii=False, indent=2)
    print(f"全部完成！最终结果保存到: {output_path}")
    
    # 删除临时文件
    temp_output = output_path.replace('.json', '_temp.json')
    if os.path.exists(temp_output):
        os.remove(temp_output)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='为SFT数据添加思维过程 - 使用OpenRouter API')
    parser.add_argument('--input_file', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出JSON文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='OpenRouter API密钥')
    parser.add_argument('--model', type=str, default='moonshot/moonshot-v1-auto', help='使用的模型名称')
    parser.add_argument('--start_index', type=int, default=0, help='开始处理的索引')
    parser.add_argument('--max_workers', type=int, default=8, help='并行线程数')
    
    args = parser.parse_args()
    
    print("=== 添加思维过程 - OpenRouter API ===")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"使用模型: {args.model}")
    print(f"并行线程数: {args.max_workers}")
    print(f"开始索引: {args.start_index}")
    
    # 添加思维过程
    process_all_cases(
        args.input_file, 
        args.output_file, 
        args.api_key,
        args.model,
        start_index=args.start_index,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()

