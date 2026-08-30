#!/usr/bin/env python3
"""
Data processing script to extract medical consultation data from CSV and convert to JSON/Parquet format.
Extracts patient_id_mdd, diagnosis, recommendation, and enhanced_description columns.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any
import argparse

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")



def create_prompt_template() -> str:
    """Create the standard prompt template for medical consultation."""
    return """<|im_start|>用户
您是一位经验丰富的医生，需要通过咨询为患者提供专业的诊断和建议。请仔细聆听患者的描述，提出有针对性的问题，并在给出诊断和治疗建议之前收集足够的信息。

快速指南
目标：
1. 通过有效提问获取关键信息，每一轮问题都应根据前一轮的内容进行修改，即不应提出类似问题。
2. 全面分析患者病情，以提供准确诊断和适当的治疗建议。

规则：
1. 您只能选择其中一个选项进行回应，不能同时回答问题并提供诊断。
2. 绝对不要重复或提出与之前问题类似或相同的问题。

回应：
<think> [您的思考] </think>
<answer>如果您认为信息不足，请只提一个问题，格式如下：
问题：（您的问题）。
</answer> | <answer>如果您认为已获得足够信息，请只提供诊断和建议，格式如下：
诊断：（患者最可能的疾病或症状）。
建议：（相应的治疗方案或建议）。
</answer>
奖励：
格式错误：-2.0
有效问题（患者能提供答案且对诊断有帮助）：+1.0
无效问题不计入分数
重复问题：-2.0
对话轮次有限。达到最大互动轮次而未提供诊断：-5.0
完全正确的诊断和建议：+10.0

患者描述：急性肠胃炎呕吐和腹泻后，还需要静脉滴注吗？


决定下一步行动：
始终输出：<think> [您的思考] </think> <answer> [您的回应] </answer> 无额外文本。请严格遵循此格式。<|im_end|>
<|im_start|>助手
<think>，"""


def process_row(row: pd.Series, index: int) -> Dict[str, Any]:
    """Process a single row from the CSV and convert to the required format."""
    
    # Extract the required columns
    patient_id = row.get('patient_id_mdd', index)
    diagnosis = row.get('OverallDiagnosis_mdd', '')
    enhanced_description = row.get('enhanced_description', '')
    
    # Process enhanced_description to keep only content after </think>
    if enhanced_description:
        if '</think>' in enhanced_description:
            # Split by </think> and take everything after it
            parts = enhanced_description.split('</think>', 1)
            if len(parts) > 1:
                enhanced_description = parts[1].strip()
            else:
                enhanced_description = ''
        elif '<think>' in enhanced_description:
            # If there's <think> but no </think>, remove everything including <think>
            enhanced_description = ''
    
    # Get medication recommendation (TreatmentRecommendation)
    medication_recommendation = row.get('TreatmentRecommendation', '')
    if pd.isna(medication_recommendation):
        medication_recommendation = ''
    
    # Get discharge advice (离院建议)
    discharge_advice = row.get('离院建议', '')
    if pd.isna(discharge_advice):
        discharge_advice = ''
    
    # Create the data structure
    data = {
        "data_source": "medical_consultation",
        "prompt": [
            {
                "content": create_prompt_template(),
                "role": "user"
            }
        ],
        "ground_truth": {
            "diagnosis": diagnosis,
            "recommendation": medication_recommendation,  # 用药建议
            "advice": discharge_advice  # 离院建议
        },
        "enhanced_description": enhanced_description,
        "extra_info": {
            "patient_id": int(patient_id) if pd.notna(patient_id) else index
        }
    }
    
    return data


def process_csv_to_json_parquet(input_csv_path: str, output_dir: str = None):
    """
    Process the CSV file and generate both JSON and Parquet outputs.
    
    Args:
        input_csv_path: Path to the input CSV file
        output_dir: Directory to save output files (default: same as input file)
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(input_csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV file: {input_csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['patient_id_mdd', 'OverallDiagnosis_mdd', 'enhanced_description', 'TreatmentRecommendation', '离院建议']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", list(df.columns))
    
    # Check for medication and discharge advice columns
    diagnosis_col = 'OverallDiagnosis_mdd' if 'OverallDiagnosis_mdd' in df.columns else None
    medication_col = 'TreatmentRecommendation' if 'TreatmentRecommendation' in df.columns else None
    discharge_col = '离院建议' if '离院建议' in df.columns else None
    
    if diagnosis_col:
        print(f"Using '{diagnosis_col}' as diagnosis column")
    else:
        print("Warning: No diagnosis column found")
        
    if medication_col:
        print(f"Using '{medication_col}' as medication recommendation column")
    else:
        print("Warning: No medication recommendation column found")
        
    if discharge_col:
        print(f"Using '{discharge_col}' as discharge advice column")
    else:
        print("Warning: No discharge advice column found")
    
    # Process each row
    processed_data = []
    
    for index, row in df.iterrows():
        try:
            processed_row = process_row(row, index)
            
            # Only keep records that have diagnosis, medication recommendation, and discharge advice
            has_diagnosis = processed_row['ground_truth']['diagnosis'] and processed_row['ground_truth']['diagnosis'].strip()
            has_recommendation = processed_row['ground_truth']['recommendation'] and processed_row['ground_truth']['recommendation'].strip()
            has_advice = processed_row['ground_truth']['advice'] and processed_row['ground_truth']['advice'].strip()
            
            if has_diagnosis and has_recommendation and has_advice:
                processed_data.append(processed_row)
            
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1} rows...")
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} rows")
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    json_output_path = os.path.join(output_dir, f"{base_name}_processed.json")
    parquet_output_path = os.path.join(output_dir, f"{base_name}_processed.parquet")
    
    # Save as JSON
    print(f"Saving JSON file: {json_output_path}")
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(processed_data)} records to JSON")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
    
    # Save as Parquet
    print(f"Saving Parquet file: {parquet_output_path}")
    try:
        # Convert to DataFrame for Parquet
        parquet_df = pd.DataFrame(processed_data)
        parquet_df.to_parquet(parquet_output_path, index=False)
        print(f"Successfully saved {len(processed_data)} records to Parquet")
    except Exception as e:
        print(f"Error saving Parquet file: {e}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total records processed: {len(processed_data)}")
    print(f"Records with diagnosis: {sum(1 for item in processed_data if item['ground_truth']['diagnosis'])}")
    print(f"Records with medication recommendation: {sum(1 for item in processed_data if item['ground_truth']['recommendation'])}")
    print(f"Records with discharge advice: {sum(1 for item in processed_data if item['ground_truth']['advice'])}")
    print(f"Records with enhanced_description: {sum(1 for item in processed_data if item['enhanced_description'])}")
    
    return processed_data


def main():
    """Main function to run the data processing."""
    parser = argparse.ArgumentParser(description='Extract medical consultation data from CSV')
    parser.add_argument('--input', '-i', 
                       default=os.path.join(DATA_ROOT, 'dialog_test_with_patient_info_enhanced.csv'),
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', 
                       default=DATA_ROOT,
                       help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Process the data
    process_csv_to_json_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
