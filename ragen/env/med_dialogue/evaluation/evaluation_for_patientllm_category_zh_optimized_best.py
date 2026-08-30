import json
import numpy as np
import torch
import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
from ragen.utils.trust import trust_remote_code as _trust_remote_code

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
MODEL_ROOT = os.environ.get("MIND_MODEL_ROOT", "models")


# Global variables for model initialization
QWEN_MODEL_PATH = os.path.join(MODEL_ROOT, "Qwen3-32B")
global_model = None
global_tokenizer = None
global_model_loaded = False
global_device = None
global_rank = None
global_world_size = None
global_local_rank = None

# Global prompt template cache
global_prompt_templates = {}

# Pre-compiled regex patterns for better performance
COMPILED_PATTERNS = {
    'error_patterns': [
        re.compile(r"Sorry, you've asked this question before"),
        re.compile(r"Sorry, I cannot answer your question")
    ],
    'score_pattern': re.compile(r'(\d{1,3})(?:\s*\/\s*5)?$'),
    'answer_pattern': re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
    # 支持中英文格式的诊断和建议提取
    'diagnosis_pattern': re.compile(r"(?:诊断|Diagnosis)[：:]\s*(.*?)(?=建议|Recommendation|$)", re.DOTALL | re.IGNORECASE),
    'recommendation_pattern': re.compile(r"(?:建议|Recommendation)[：:]\s*(.*?)$", re.DOTALL | re.IGNORECASE),
    'category_patterns': {
        'Anxiety': re.compile(r'焦虑|anxiety', re.IGNORECASE),
        'Depression': re.compile(r'抑郁|depression', re.IGNORECASE),
        'Mix': re.compile(r'混合|mix|焦虑.*抑郁|抑郁.*焦虑', re.IGNORECASE),
        'Other': re.compile(r'其他|other', re.IGNORECASE)
    }
}

def setup_distributed():
    """Setup distributed training environment"""
    global global_rank, global_world_size, global_local_rank
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return False
    
    global_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if global_local_rank == -1:
        print("LOCAL_RANK environment variable not set. Using single GPU.")
        return False
    
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    
    global_device = f"cuda:{global_local_rank}"
    torch.cuda.set_device(global_local_rank)
    
    print(f"Process {global_rank}: Using device: {global_device}")
    return True

def load_model_if_needed():
    """Load model if not already loaded"""
    global global_model, global_tokenizer, global_device, global_model_loaded
    
    if global_model_loaded:
        return True
    
    try:
        print("Loading model for critical evaluations...")
        global_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=_trust_remote_code())
        global_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=_trust_remote_code(),
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        if global_local_rank != -1:
            global_device = f"cuda:{global_local_rank}"
        else:
            global_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global_model_loaded = True
        print(f"Process {global_rank if global_rank is not None else 0}: Model loaded successfully on {global_device}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_entities_batch(texts):
    """
    批量从文本中提取信息实体（症状、体征等）
    使用大模型批量提取
    
    Parameters:
        texts (list of str): List of input texts
    
    Returns:
        list of list: List of extracted entities for each text
    """
    if not texts:
        return []
    
    if not global_model_loaded:
        print(f"Warning: Model not loaded, cannot extract entities. Returning empty results.")
        return [[] for _ in texts]
    
    prompts = []
    for text in texts:
        if not text:
            prompts.append(None)
            continue
        
        try:
            # 使用缓存的prompt模板
            prompt_template = load_prompt_template('eval_entity_extraction_prompt_template.txt')
            prompt = prompt_template.format(patient_text=text)
            prompts.append(prompt)
        except Exception as e:
            print(f"Error reading entity extraction prompt template: {e}")
            prompts.append(None)
    
    results = []
    # Process prompts in batches, handling None placeholders
    valid_prompts = [p for p in prompts if p is not None]
    empty_indices = [i for i, p in enumerate(prompts) if p is None]
    
    if valid_prompts:
        try:
            # Tokenize the batch of prompts
            inputs = global_tokenizer(valid_prompts, return_tensors="pt", padding=True, truncation=True).to(global_device)
            
            with torch.no_grad():
                # Generate responses for the batch
                outputs = global_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.01,
                    top_p=0.9,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=global_tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            
            # Decode and extract entities from the batch responses
            response_texts = global_tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Debug: Print first response
            if response_texts:
                print(f"\n[DEBUG] Entity extraction - First model response (first 300 chars):")
                print(f"{response_texts[0][:300]}")
            
            valid_results = []
            for idx, response_text in enumerate(response_texts):
                # Extract entities list
                entities = []
                match = COMPILED_PATTERNS['answer_pattern'].search(response_text)
                if match:
                    answer_content = match.group(1).strip()
                    # Split entities by lines
                    entity_lines = [line.strip() for line in answer_content.split('\n') if line.strip()]
                    # 改进的过滤逻辑：过滤掉prompt标记和格式说明
                    entities = [
                        entity for entity in entity_lines 
                        if entity 
                        and not entity.startswith('-')
                        and not entity.startswith('*')
                        and 'tags:' not in entity.lower()
                        and '<answer>' not in entity
                        and '</answer>' not in entity
                        and 'provide a list' not in entity.lower()
                        and len(entity) < 100  # 避免提取到过长的文本
                    ]
                    if idx == 0:  # Debug first response
                        print(f"[DEBUG] Found <answer> tag, extracted {len(entities)} entities")
                        if entities:
                            print(f"[DEBUG] First 3 entities: {entities[:3]}")
                else:
                    if idx == 0:  # Debug first response
                        print(f"[DEBUG] No <answer> tag found in response")
                
                valid_results.append(entities)
            
            # Reconstruct the full list of results, including placeholders for empty cases
            full_results = [[] for _ in prompts]
            valid_result_index = 0
            for i in range(len(prompts)):
                if i in empty_indices:
                    full_results[i] = []  # Empty list for empty cases
                else:
                    full_results[i] = valid_results[valid_result_index]
                    valid_result_index += 1
            
            results = full_results
            
        except Exception as e:
            print(f"Error in entity extraction batch: {e}")
            # Return empty lists for all texts in case of a batch error
            return [[] for _ in texts]
    
    else:  # All prompts were None (empty texts)
        results = [[] for _ in texts]
    
    return results

def calculate_optimized_info_retrieval_rate_batch(simulation_data_list, batch_size=16):
    """
    批量优化的信息检索率计算
    每个案例只调用2次大模型，但使用批量处理提高效率
    
    Parameters:
        simulation_data_list (list): List of simulation data items
        batch_size (int): Batch size for processing
    
    Returns:
        list: List of information retrieval statistics for each item
    """
    try:
        # 准备所有需要提取的文本
        true_info_texts = []
        patient_texts = []
        
        for sim_item in simulation_data_list:
            # 真集信息文本
            true_info_text = sim_item.get("enhanced_description", "")
            true_info_texts.append(true_info_text)
            
            # 收集患者所有回答
            patient_responses = []
            for turn in sim_item.get("simulation_dialogue", []):
                if turn.get("role") == "patient" and "content" in turn:
                    response = turn["content"].strip()
                    # 过滤错误回答
                    if not any(pattern.search(response) for pattern in COMPILED_PATTERNS['error_patterns']):
                        patient_responses.append(response)
            
            # 合并患者回答
            if patient_responses:
                combined_patient_text = " ".join(patient_responses)
                patient_texts.append(combined_patient_text)
            else:
                patient_texts.append("")
        
        # 批量提取真集信息实体
        print(f"Batch extracting true entities from {len(true_info_texts)} texts...")
        all_true_entities = []
        for i in tqdm(range(0, len(true_info_texts), batch_size), desc="True Entities Extraction"):
            batch_texts = true_info_texts[i:i + batch_size]
            batch_entities = extract_entities_batch(batch_texts)
            all_true_entities.extend(batch_entities)
        
        # 批量提取患者收集的信息实体
        print(f"Batch extracting patient entities from {len(patient_texts)} texts...")
        all_gathered_entities = []
        for i in tqdm(range(0, len(patient_texts), batch_size), desc="Patient Entities Extraction"):
            batch_texts = patient_texts[i:i + batch_size]
            batch_entities = extract_entities_batch(batch_texts)
            all_gathered_entities.extend(batch_entities)
        
        # 计算每个案例的检索率
        results = []
        for i in range(len(simulation_data_list)):
            true_entities = list(set(all_true_entities[i]))  # 去重
            gathered_entities = list(set(all_gathered_entities[i]))  # 去重
            
            # 计算匹配的实体
            matched_entities = [entity for entity in gathered_entities if entity in true_entities]
            
            # 计算检索率
            if len(true_entities) == 0:
                retrieval_rate = 0.0
            else:
                retrieval_rate = len(matched_entities) / len(true_entities)
            
            results.append({
                "retrieval_rate": retrieval_rate,
                "true_entities_count": len(true_entities),
                "gathered_entities_count": len(gathered_entities),
                "matched_entities_count": len(matched_entities),
                "true_entities": true_entities,
                "gathered_entities": gathered_entities,
                "matched_entities": matched_entities
            })
        
        return results
        
    except Exception as e:
        print(f"Error calculating optimized info retrieval rate batch: {e}")
        # Return empty results for all items
        return [{
            "retrieval_rate": 0.0,
            "true_entities_count": 0,
            "gathered_entities_count": 0,
            "matched_entities_count": 0,
            "true_entities": [],
            "gathered_entities": [],
            "matched_entities": []
        } for _ in simulation_data_list]

def calculate_recommendation_semantic_score_batch(data_pairs):
    """
    使用大模型计算建议语义分数
    
    Parameters:
        data_pairs (list of tuples): List of (candidate_text, reference_text) tuples
    
    Returns:
        list of float: List of semantic similarity scores (0-5)
    """
    if not data_pairs:
        return []
    
    if not global_model_loaded and not load_model_if_needed():
        print("Warning: Model not loaded, skipping recommendation semantic scoring")
        return [0.0] * len(data_pairs)
    
    prompts = []
    for candidate, reference in data_pairs:
        if not candidate or not reference:
            prompts.append(None)
            continue
        
        # 使用建议评估的prompt模板
        template = load_prompt_template('eval_prompt_template_v2.txt')
        prompt = template.format(candidate=candidate, reference=reference)
        prompts.append(prompt)
    
    scores = []
    valid_prompts = [p for p in prompts if p is not None]
    empty_indices = [i for i, p in enumerate(prompts) if p is None]
    
    if valid_prompts:
        try:
            inputs = global_tokenizer(valid_prompts, return_tensors="pt", padding=True, truncation=True).to(global_device)
            
            with torch.no_grad():
                outputs = global_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.01,
                    top_p=0.9,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=global_tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            
            response_texts = global_tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            valid_scores = []
            score_pattern = COMPILED_PATTERNS['score_pattern']
            answer_pattern = COMPILED_PATTERNS['answer_pattern']
            
            for response_text in response_texts:
                match = answer_pattern.search(response_text)
                if match:
                    matched_response = match.group(0)
                else:
                    matched_response = response_text.strip()
                
                match = score_pattern.search(matched_response)
                if match:
                    score = float(match.group(1))
                    valid_scores.append(min(max(score, 0), 5))
                else:
                    numbers = re.findall(r'\b\d{1,3}\b', response_text.strip())
                    found_score = False
                    if numbers:
                        for num in numbers:
                            num_val = float(num)
                            if 0 <= num_val <= 5:
                                valid_scores.append(num_val)
                                found_score = True
                                break
                    if not found_score:
                        print(f"Warning: Could not extract score from: {response_text.strip()}")
                        valid_scores.append(0.0)
            
            full_scores = [0.0] * len(prompts)
            valid_score_index = 0
            for i in range(len(prompts)):
                if i in empty_indices:
                    full_scores[i] = 0.0
                else:
                    full_scores[i] = valid_scores[valid_score_index]
                    valid_score_index += 1
            
            scores = full_scores
            
        except Exception as e:
            print(f"Error in recommendation semantic scoring batch: {e}")
            return [0.0] * len(data_pairs)
    
    else:
        scores = [0.0] * len(data_pairs)
    
    return scores

def calculate_diagnosis_category_rule_based(model_output):
    """
    基于规则的诊断分类（无需大模型）
    
    Parameters:
        model_output (str): Model's diagnosis output
    
    Returns:
        str: Diagnosis category
    """
    try:
        category_patterns = COMPILED_PATTERNS['category_patterns']
        
        for category, pattern in category_patterns.items():
            if pattern.search(model_output):
                return category
        
        return "Other"
    except Exception as e:
        print(f"Error in rule-based diagnosis classification: {e}")
        return "Other"

def calculate_bleu_metrics(candidate, reference, max_n=4):
    """
    计算BLEU指标（无需大模型）
    
    Parameters:
        candidate (str): Candidate text
        reference (str): Reference text
        max_n (int): Maximum n-gram order
    
    Returns:
        dict: BLEU scores for different n-grams
    """
    def tokenize(text):
        if not text:
            return []
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return list(text)
        else:
            return text.split()
    
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def calculate_bleu_n(candidate_tokens, reference_tokens, n):
        if len(candidate_tokens) < n:
            return 0.0
        
        candidate_ngrams = get_ngrams(candidate_tokens, n)
        reference_ngrams = get_ngrams(reference_tokens, n)
        
        if not reference_ngrams:
            return 0.0
        
        # Calculate precision
        matches = 0
        for ngram in candidate_ngrams:
            if ngram in reference_ngrams:
                matches += 1
        
        precision = matches / len(candidate_ngrams) if candidate_ngrams else 0.0
        
        # Calculate brevity penalty
        if len(candidate_tokens) < len(reference_tokens):
            bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
        else:
            bp = 1.0
        
        return bp * precision
    
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    bleu_scores = {}
    for n in range(1, max_n + 1):
        bleu_scores[f"bleu_{n}"] = calculate_bleu_n(candidate_tokens, reference_tokens, n)
    
    return bleu_scores

def calculate_interaction_efficiency_metrics(simulation_data_item):
    """
    计算交互效率指标（无需大模型）
    
    Parameters:
        simulation_data_item (dict): Simulation data item
    
    Returns:
        dict: Interaction efficiency metrics
    """
    dialogue = simulation_data_item.get("simulation_dialogue", [])
    
    # 计算总轮数
    total_turns = len(dialogue)
    
    # 计算平均token数
    total_tokens = 0
    for turn in dialogue:
        content = turn.get("content", "")
        if content:
            # 简单估算token数（中文字符按1个token计算，英文按单词数计算）
            if any('\u4e00' <= c <= '\u9fff' for c in content):
                total_tokens += len(content)
            else:
                total_tokens += len(content.split())
    
    avg_tokens = total_tokens / total_turns if total_turns > 0 else 0
    
    # 计算交互效率（轮数越少，token数越少，效率越高）
    interaction_efficiency = 1.0 / (1.0 + total_turns * 0.1 + avg_tokens * 0.01)
    
    return {
        "total_turns": total_turns,
        "avg_tokens": avg_tokens,
        "interaction_efficiency": interaction_efficiency
    }

def evaluate_optimized_metrics(simulation_data_list, reference_data_list=None, alpha=1.0, beta=0.01):
    """
    优化评估：按新的权重分配和优化的大模型调用策略
    
    Parameters:
        simulation_data_list (list): List of simulation dialogue data items
        reference_data_list (list, optional): Reference data list
        alpha (float): Weight for interaction turns in DEI calculation
        beta (float): Weight for average token count in DEI calculation
    
    Returns:
        dict: Evaluation results
    """
    if not simulation_data_list:
        print("No simulation data provided")
        return {}
    
    # Load model for critical evaluations only
    if not load_model_if_needed():
        print("Model loading failed. Cannot proceed with evaluation.")
        return {}
    
    # Filter simulation data
    if reference_data_list and "self_report" in reference_data_list[0]:
        all_ref_report = [ref_item["self_report"] for ref_item in reference_data_list]
        simulation_data_list = [sim_item for sim_item in simulation_data_list if sim_item.get("self_report") in all_ref_report]
        print("Filtered simulation data list: ", len(simulation_data_list))
        
        des_index = {}
        for ref_item in reference_data_list:
            des_index[ref_item["self_report"]] = ref_item["enhanced_description"]
        for sim_item in simulation_data_list:
            if "enhanced_description" not in sim_item:
                sim_item["enhanced_description"] = des_index[sim_item["self_report"]]
    else:
        print("No reference data or missing self_report field, using all simulation data: ", len(simulation_data_list))
    
    case_results = []
    all_recommendation_pairs = []
    all_predictions = []
    all_true_labels = []
    
    # Step 1: Calculate information retrieval quality first (most time-consuming) - BATCH PROCESSING
    print(f"Process {global_rank if global_rank is not None else 0}: Calculating information retrieval quality with batch processing...")
    
    batch_size = 16
    all_info_retrieval_stats = calculate_optimized_info_retrieval_rate_batch(simulation_data_list, batch_size)
    
    # Step 2: Extract model outputs
    print(f"Process {global_rank if global_rank is not None else 0}: Extracting model outputs...")
    
    for sim_item in simulation_data_list:
        diagnosis_turns = [turn for turn in sim_item["simulation_dialogue"]
                          if turn.get("role") == "doctor" and turn.get("is_diagnosis", False)]
        
        if diagnosis_turns:
            final_diagnosis_turn = diagnosis_turns[-1]
            model_output = final_diagnosis_turn["content"]
            
            model_diagnosis, model_recommendation = extract_diagnosis_and_recommendation_text(model_output)
            predicted_category = calculate_diagnosis_category_rule_based(model_output)
        else:
            model_diagnosis = ""
            model_recommendation = ""
            predicted_category = "Other"
        
        true_diagnosis = sim_item.get("true_diagnosis", "")
        true_recommendation = sim_item.get("true_recommendation", "")
        true_category = true_diagnosis
        if true_category == "Others":
            true_category = "Other"
        
        all_recommendation_pairs.append((model_recommendation, true_recommendation))
        all_predictions.append(predicted_category)
        all_true_labels.append(true_category)
    
    # Step 3: Calculate recommendation semantic scores with LLM
    print(f"Process {global_rank if global_rank is not None else 0}: Calculating recommendation semantic scores with LLM...")
    
    batch_size = 16
    all_recommendation_semantic_scores = []
    
    for i in tqdm(range(0, len(all_recommendation_pairs), batch_size), desc="Recommendation Semantic Scoring"):
        batch_pairs = all_recommendation_pairs[i:i + batch_size]
        batch_scores = calculate_recommendation_semantic_score_batch(batch_pairs)
        all_recommendation_semantic_scores.extend(batch_scores)
    
    # Calculate other metrics
    print(f"Process {global_rank if global_rank is not None else 0}: Calculating other metrics...")
    
    for i in tqdm(range(len(simulation_data_list)), desc="Calculating Other Metrics"):
        sim_item = simulation_data_list[i]
        
        # Extract diagnosis and recommendation
        diagnosis_turns = [turn for turn in sim_item["simulation_dialogue"]
                          if turn.get("role") == "doctor" and turn.get("is_diagnosis", False)]
        
        if diagnosis_turns:
            final_diagnosis_turn = diagnosis_turns[-1]
            model_output = final_diagnosis_turn["content"]
            model_diagnosis, model_recommendation = extract_diagnosis_and_recommendation_text(model_output)
        else:
            model_diagnosis = ""
            model_recommendation = ""
        
        true_diagnosis = sim_item.get("true_diagnosis", "")
        true_recommendation = sim_item.get("true_recommendation", "")
        
        # 1. 诊断分类 (25% 权重) - 无需大模型
        predicted_category = calculate_diagnosis_category_rule_based(model_output)
        classification_f1 = calculate_classification_f1(predicted_category, true_diagnosis)
        
        # 2. 建议语义一致性 (30% 权重) - 使用大模型
        recommendation_semantic_score = all_recommendation_semantic_scores[i]
        
        # 3. 信息收集质量 (25% 权重) - 使用预先计算的结果
        info_retrieval_stats = all_info_retrieval_stats[i]
        
        # 4. 基础文本指标 (10% 权重) - 无需大模型
        diagnosis_bleu = calculate_bleu_metrics(model_diagnosis, true_diagnosis)
        recommendation_bleu = calculate_bleu_metrics(model_recommendation, true_recommendation)
        
        # 5. 效率指标 (10% 权重) - 无需大模型
        efficiency_stats = calculate_interaction_efficiency_metrics(sim_item)
        
        # 计算综合评分
        combined_score = (
            classification_f1["overall_f1"] * 0.25 +  # 诊断分类
            recommendation_semantic_score / 5.0 * 0.30 +  # 建议语义一致性
            info_retrieval_stats["retrieval_rate"] * 0.25 +  # 信息收集质量
            (diagnosis_bleu["bleu_4"] + recommendation_bleu["bleu_4"]) / 2 * 0.10 +  # 基础文本指标
            efficiency_stats["interaction_efficiency"] * 0.10  # 效率指标
        )
        
        result = {
            "id": sim_item.get("id"),
            "diagnostic_performance": {
                "combined_score": combined_score,
                "classification_f1": classification_f1,
                "recommendation_semantic_score": recommendation_semantic_score,
                "diagnosis_bleu": diagnosis_bleu,
                "recommendation_bleu": recommendation_bleu
            },
            "information_retrieval": info_retrieval_stats,
            "interaction_efficiency": efficiency_stats,
            "evaluation_mode": "optimized"
        }
        
        case_results.append(result)
    
    # Calculate global metrics
    global_metrics = calculate_global_classification_metrics(all_predictions, all_true_labels)
    
    # Calculate average results
    if case_results:
        avg_results = {}
        for key in case_results[0].keys():
            if key == "id" or key == "evaluation_mode":
                continue
            elif isinstance(case_results[0][key], dict):
                avg_results[key] = {}
                for sub_key in case_results[0][key].keys():
                    if isinstance(case_results[0][key][sub_key], (int, float)):
                        avg_results[key][sub_key] = sum(result[key][sub_key] for result in case_results) / len(case_results)
                    else:
                        avg_results[key][sub_key] = case_results[0][key][sub_key]
            elif isinstance(case_results[0][key], (int, float)):
                avg_results[key] = sum(result[key] for result in case_results) / len(case_results)
            else:
                avg_results[key] = case_results[0][key]
        
        avg_results["global_classification_metrics"] = global_metrics
        
        # 添加额外的统计指标
        if "information_retrieval" in avg_results and "interaction_efficiency" in avg_results:
            avg_gathered = avg_results["information_retrieval"].get("gathered_entities_count", 0)
            avg_turns = avg_results["interaction_efficiency"].get("total_turns", 1)
            
            # 平均每轮对话提取的信息量
            entities_per_turn = avg_gathered / avg_turns if avg_turns > 0 else 0
            
            avg_results["information_retrieval"]["avg_gathered_entities_per_case"] = avg_gathered
            avg_results["information_retrieval"]["avg_entities_per_turn"] = entities_per_turn
        
        return {
            "case_results": case_results,
            "average_results": avg_results,
            "global_classification_metrics": global_metrics,
            "evaluation_mode": "optimized"
        }
    else:
        return {"case_results": [], "average_results": {}, "global_classification_metrics": {}}

def extract_diagnosis_and_recommendation_text(model_output):
    """Extract diagnosis and recommendation text"""
    try:
        diagnosis_match = COMPILED_PATTERNS['diagnosis_pattern'].search(model_output)
        recommendation_match = COMPILED_PATTERNS['recommendation_pattern'].search(model_output)
        
        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
        recommendation = recommendation_match.group(1).strip() if recommendation_match else ""
        
        return diagnosis, recommendation
    except Exception as e:
        print(f"Error extracting diagnosis and recommendation: {e}")
        return "", ""

def calculate_classification_f1(predicted_category, true_category):
    """Calculate classification F1 score"""
    if predicted_category == true_category:
        return {
            "overall_f1": 1.0,
            "anxiety_f1": 1.0 if predicted_category == "Anxiety" else 0.0,
            "depression_f1": 1.0 if predicted_category == "Depression" else 0.0,
            "mix_f1": 1.0 if predicted_category == "Mix" else 0.0,
            "other_f1": 1.0 if predicted_category == "Other" else 0.0
        }
    else:
        return {
            "overall_f1": 0.0,
            "anxiety_f1": 0.0,
            "depression_f1": 0.0,
            "mix_f1": 0.0,
            "other_f1": 0.0
        }

def calculate_global_classification_metrics(all_predictions, all_true_labels):
    """Calculate global classification metrics"""
    categories = ['Anxiety', 'Depression', 'Mix', 'Other']
    
    confusion_matrix = {}
    for true_cat in categories:
        confusion_matrix[true_cat] = {}
        for pred_cat in categories:
            confusion_matrix[true_cat][pred_cat] = 0
    
    for pred, true in zip(all_predictions, all_true_labels):
        confusion_matrix[true][pred] += 1
    
    category_metrics = {}
    for category in categories:
        tp = confusion_matrix[category][category]
        fp = sum(confusion_matrix[other_cat][category] for other_cat in categories if other_cat != category)
        fn = sum(confusion_matrix[category][other_cat] for other_cat in categories if other_cat != category)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        true_count = sum(confusion_matrix[category].values())
        pred_count = sum(confusion_matrix[other_cat][category] for other_cat in categories)
        total_count = len(all_predictions)
        
        true_ratio = true_count / total_count if total_count > 0 else 0
        pred_ratio = pred_count / total_count if total_count > 0 else 0
        
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_count': true_count,
            'pred_count': pred_count,
            'true_ratio': true_ratio,
            'pred_ratio': pred_ratio
        }
    
    overall_precision = sum(category_metrics[cat]['precision'] for cat in categories) / len(categories)
    overall_recall = sum(category_metrics[cat]['recall'] for cat in categories) / len(categories)
    overall_f1 = sum(category_metrics[cat]['f1'] for cat in categories) / len(categories)
    
    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'categories': category_metrics,
        'confusion_matrix': confusion_matrix
    }

def load_prompt_template(template_name):
    """Load and cache prompt template"""
    global global_prompt_templates
    
    if template_name not in global_prompt_templates:
        try:
            template_path = f'ragen/env/med_dialogue/evaluation/{template_name}'
            with open(template_path, 'r', encoding='utf-8') as file:
                global_prompt_templates[template_name] = file.read()
        except Exception as e:
            print(f"Error loading prompt template {template_name}: {e}")
            global_prompt_templates[template_name] = ""
    
    return global_prompt_templates[template_name]

def main():
    parser = argparse.ArgumentParser(description='Medical Dialogue Diagnosis Evaluation - Optimized Version')
    parser.add_argument('--simulation_data', type=str, required=True, help='Path to simulation data JSON file')
    parser.add_argument('--reference_data', type=str, required=True, help='Path to reference data JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output results JSON file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for interaction turns in DEI calculation')
    parser.add_argument('--beta', type=float, default=0.01, help='Weight for average token count in DEI calculation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for semantic similarity calculation')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--max_cases', type=int, default=None, help='Maximum number of cases to evaluate (for testing)')

    args = parser.parse_args()

    setup_distributed()

    # Load simulation data
    try:
        with open(args.simulation_data, 'r', encoding='utf-8') as f:
            simulation_data = json.load(f)
            simulation_data.sort(key=lambda x: x["id"])
    except FileNotFoundError:
        print(f"Simulation data file not found: {args.simulation_data}")
        return

    # Limit number of cases if specified (for testing)
    if args.max_cases is not None:
        simulation_data = simulation_data[:args.max_cases]
        print(f"Limited to {args.max_cases} cases for testing")

    # Load reference data
    reference_data = None
    try:
        with open(args.reference_data, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
    except FileNotFoundError:
        print(f"Reference data file not found: {args.reference_data}, skipping info retrieval rate.")

    # Run evaluation
    results = evaluate_optimized_metrics(
        simulation_data_list=simulation_data,
        reference_data_list=reference_data,
        alpha=args.alpha,
        beta=args.beta
    )

    # Save results
    if results and (not global_rank or global_rank == 0):
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Display summary results
        if results.get("average_results"):
            avg_results = results["average_results"]
            
            print(f"\n=== Optimized Evaluation Results Summary ===")
            print(f"Total cases evaluated: {len(results.get('case_results', []))}")
            
            if "diagnostic_performance" in avg_results:
                print(f"\n=== Diagnostic Performance ===")
                print(f"Combined Score: {avg_results['diagnostic_performance']['combined_score']:.3f}")
                print(f"Classification F1: {avg_results['diagnostic_performance']['classification_f1']['overall_f1']:.3f}")
                print(f"Recommendation Semantic Score: {avg_results['diagnostic_performance']['recommendation_semantic_score']:.3f}")
            
            if "information_retrieval" in avg_results:
                print(f"\n=== Information Retrieval ===")
                print(f"Retrieval Rate: {avg_results['information_retrieval']['retrieval_rate']:.3f}")
                print(f"True Entities Count: {avg_results['information_retrieval']['true_entities_count']:.1f}")
                print(f"Gathered Entities Count: {avg_results['information_retrieval']['gathered_entities_count']:.1f}")
                print(f"Avg Gathered Entities Per Case: {avg_results['information_retrieval'].get('avg_gathered_entities_per_case', 0):.1f}")
                print(f"Avg Entities Per Turn: {avg_results['information_retrieval'].get('avg_entities_per_turn', 0):.3f}")
            
            if "interaction_efficiency" in avg_results:
                print(f"\n=== Interaction Efficiency ===")
                print(f"Total Turns: {avg_results['interaction_efficiency']['total_turns']:.1f}")
                print(f"Average Tokens: {avg_results['interaction_efficiency']['avg_tokens']:.1f}")
                print(f"Interaction Efficiency: {avg_results['interaction_efficiency']['interaction_efficiency']:.3f}")
            
            print(f"\nFull results saved to: {args.output}")
        else:
            print("\nNo results to display.")
    else:
        print(f"Process {global_rank}: Skipping result saving (only rank 0 saves results)")
    
    if global_world_size and global_world_size > 1:
        dist.barrier()
        print(f"Process {global_rank}: Evaluation completed")

if __name__ == "__main__":
    main()
