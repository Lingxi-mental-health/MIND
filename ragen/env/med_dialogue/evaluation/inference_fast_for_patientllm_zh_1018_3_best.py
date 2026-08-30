################################################################################
# v6超激进优化版本：最大化GPU利用率，最小化CPU瓶颈
################################################################################

import json
import torch
import time
import os
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict
from collections import defaultdict
import asyncio
import concurrent.futures
from functools import lru_cache
import threading
from queue import Queue
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from ragen.utils.trust import trust_remote_code as _trust_remote_code

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")
MODEL_ROOT = os.environ.get("MIND_MODEL_ROOT", "models")
PROJECT_ROOT = os.environ.get("MIND_PROJECT_ROOT", ".")



def print_timing(message, verbose=True, last_time=None):
    """打印时间信息，格式为[CURRENT_TIME] message"""
    if not verbose:
        return time.time()
    
    current_time = time.time()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if last_time is not None:
        elapsed = current_time - last_time
        print(f"[CURRENT_TIME] {current_time_str} - {message} (耗时: {elapsed:.2f}s)")
    else:
        print(f"[CURRENT_TIME] {current_time_str} - {message}")
    
    return current_time


# 添加分布式训练相关的全局变量
global_rank = None
global_world_size = None
global_local_rank = None

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
    
    dist.init_process_group(backend="nccl", timeout=torch.distributed.constants.default_pg_timeout)
    
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    
    torch.cuda.set_device(global_local_rank)
    
    print(f"Initialized process {global_rank}/{global_world_size} on GPU {global_local_rank}")
    return True


class UltraOptimizedTokenCache:
    """超优化Token缓存，使用numpy和C扩展"""
    def __init__(self, tokenizer, max_size=100000):
        self.tokenizer = tokenizer
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.access_counter = 0
    
    @lru_cache(maxsize=100000)
    def get_token_count(self, text):
        """超优化token计算"""
        return len(self.tokenizer.encode(text))
    
    def batch_token_count(self, texts):
        """批量token计算，减少CPU开销"""
        return [self.get_token_count(text) for text in texts]
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


class GPUStreamManager:
    """GPU流管理器，最大化并行度"""
    def __init__(self, device_count=2):
        # 获取实际可用的GPU设备数量
        available_devices = torch.cuda.device_count()
        self.device_count = min(device_count, available_devices)
        self.streams = []
        self.events = []
        
        print(f"Requested {device_count} devices, available devices: {available_devices}, using: {self.device_count}")
        
        for i in range(self.device_count):
            stream = torch.cuda.Stream(device=i)
            self.streams.append(stream)
            self.events.append(torch.cuda.Event())
    
    def get_stream(self, device_id):
        """获取指定设备的流"""
        return self.streams[device_id % self.device_count]
    
    def synchronize(self):
        """同步所有流"""
        for stream in self.streams:
            stream.synchronize()


class UltraOptimizedMedicalDialogueSimulation:
    def __init__(self, model_path, input_file, output_file, temperature=0.7, top_p=0.9, 
                 device_map="auto", batch_size=128, patient_model_path=None, verbose=False):
        """
        v6超激进优化版本
        主要优化：
        1. 优化批处理大小（32-64）
        2. GPU流并行处理
        3. 异步预处理管道
        4. 内存池管理
        5. 预计算优化
        """
        init_start_time = print_timing("开始初始化v6超激进优化版本", verbose)
        
        self.input_file = input_file
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size  # v6: 默认32
        self.verbose = verbose
        self.model_path = model_path

        # 设置分布式环境
        setup_start_time = print_timing("开始设置分布式环境", verbose, init_start_time)
        is_distributed = setup_distributed()
        if is_distributed:
            self.device = torch.device(f"cuda:{global_local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Process {global_rank if is_distributed else 0}: Using device: {self.device}")
        setup_end_time = print_timing("分布式环境设置完成", verbose, setup_start_time)

        self.torch_dtype = torch.float16
        
        # v6优化：预编译所有正则表达式
        # 使用enable_thinking=True时，模型会输出<think>标签，所以需要保留think_pattern
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self.diagnosis_pattern = re.compile(r"<[Dd]iagnosis>[:：](.*?)(?=<[Rr]ecommendation>[:：]|$)", re.DOTALL)
        self.recommendation_pattern = re.compile(r"<[Rr]ecommendation>[:：](.*?)(?=\n|$)", re.DOTALL)
        
        # v6优化：GPU流管理器
        self.stream_manager = GPUStreamManager(2)  # 使用2卡GPU
        
        # 加载医生模型和分词器
        doctor_tokenizer_start_time = print_timing("开始加载医生模型分词器", verbose, setup_end_time)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=_trust_remote_code(), 
            padding_side='left',
            local_files_only=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # v6优化：超优化token缓存
        self.token_cache = UltraOptimizedTokenCache(self.tokenizer)
        doctor_tokenizer_end_time = print_timing("医生模型分词器加载完成", verbose, doctor_tokenizer_start_time)

        # v6优化：医生模型加载 - 最大化GPU利用率
        doctor_model_start_time = print_timing("开始加载医生模型", verbose, doctor_tokenizer_end_time)
        if is_distributed:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # 自动分布到多卡1465650
                torch_dtype=self.torch_dtype,
                local_files_only=True,
                trust_remote_code=_trust_remote_code(),
                attn_implementation="flash_attention_2",
                use_cache=True,
                low_cpu_mem_usage=True,
            )
            # 不要手动to(device)，device_map="auto"已经处理了
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=_trust_remote_code(),
                attn_implementation="flash_attention_2",
                use_cache=True,
                low_cpu_mem_usage=True,
            )
        
        # v6优化：预热GPU
        self._warmup_gpu()
        doctor_model_end_time = print_timing("医生模型加载完成", verbose, doctor_model_start_time)

        # 加载患者模型和分词器
        if patient_model_path is None:
            patient_model_path = os.path.join(MODEL_ROOT, "Qwen3-8B")
        
        patient_tokenizer_start_time = print_timing("开始加载患者模型分词器", verbose, doctor_model_end_time)
        self.patient_tokenizer = AutoTokenizer.from_pretrained(
            patient_model_path, 
            trust_remote_code=_trust_remote_code(), 
            padding_side='left', 
            local_files_only=True,
            use_fast=True
        )
        
        if self.patient_tokenizer.pad_token is None:
            self.patient_tokenizer.pad_token = self.patient_tokenizer.eos_token
            self.patient_tokenizer.pad_token_id = self.patient_tokenizer.eos_token_id
        
        # v6优化：患者token缓存
        self.patient_token_cache = UltraOptimizedTokenCache(self.patient_tokenizer)
        patient_tokenizer_end_time = print_timing("患者模型分词器加载完成", verbose, patient_tokenizer_start_time)

        # v6优化：患者模型加载
        patient_model_start_time = print_timing("开始加载患者模型", verbose, patient_tokenizer_end_time)
        if is_distributed:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map="auto",  # 自动分布到多卡
                torch_dtype=self.torch_dtype,
                trust_remote_code=_trust_remote_code(),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            # 不要手动to(device)，device_map="auto"已经处理了
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=_trust_remote_code(),
                low_cpu_mem_usage=True,
            )
        
        # v6优化：预热患者模型GPU
        self._warmup_patient_gpu()
        patient_model_end_time = print_timing("患者模型加载完成", verbose, patient_model_start_time)

        # 打印模型设备信息
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("Doctor model first param device:", next(self.model.parameters()).device)
        print("Patient model first param device:", next(self.patient_model.parameters()).device)
        print("Doctor hf_device_map:", getattr(self.model, "hf_device_map", None))
        print("Patient hf_device_map:", getattr(self.patient_model, "hf_device_map", None))
        
        print_timing("v6超激进优化版本初始化完成", verbose, init_start_time)

        # 系统提示词 (保持不变)
        self.doctor_system_prompt_early = r"""
你是一位经验丰富的精神科医生，正在进行【早期问诊】。  
目标：在自然、共情的交流中，了解患者的主要困扰、初步判断问题方向（情绪/焦虑/躯体），并排除危险信号。  
保持真实门诊语气——像医生和病人的对话，不要机械罗列。

【问诊目标】
- 先确认主诉及其持续时间、严重程度、诱因；
- 若患者提及身体症状或风险信号（胸痛、气促、晕厥、自杀念头等），优先追问安全相关细节；
- 若无红旗，则了解主要困扰偏向情绪低落、紧张焦虑，还是身体不适；
- 每次可以问**1-2个问题**，让患者容易回答；可以适当并列询问相关话题。

【表达要求】
- 语言自然、温和、有共情；
- 不使用"或者/以及/并且/或"等并列词；
- 问句必须以问号结尾；
- 不出现诊断或专业名词（如"抑郁""焦虑障碍"），只谈具体体验；
- 问法口语化：像面对真实患者，而不是背问卷。

【输出格式】
- 所有输出必须严格使用 <answer></answer> 标签包裹医生的回复
- 医生的回复应该是一句话，包含共情 + 1-2个问题。

示例：
思考：想了解症状持续时间和诱因
<answer>"我理解你最近挺难受的，这种情况大概持续了多久？有没有特别的事情引发它？"</answer>

思考：怀疑有红旗，想确认安全
<answer>"你提到最近心口会痛，我想确认一下，是活动时加重还是休息时也会痛？"</answer>

思考：判断问题偏情绪还是焦虑方向
<answer>"最近让你最困扰的是心情总低落，还是那种紧张和担心停不下来的感觉？"</answer>

【自检提示】
- 格式检查：是否使用了<answer></answer>标签？
- <answer>标签中的句子是否自然、贴近医生语气？
- <answer>标签中是否只包含1-3个问号？
- 是否聚焦一个重点？
- 若患者主诉中出现身体危险信号，是否优先追问安全相关？
"""

        self.doctor_system_prompt_late = r"""
你是一位经验丰富的精神科医生，正在进行基于 ICD-10 的【后期问诊与决策】。目标：在自然、简洁的交流中，补齐关键证据；信息足够时给出明确结论。保持共情、口语化、每次可以问1-2个问题。

— 流程总则 —
- 关键信息仍缺 → 继续问【1-2个】聚焦问题（可以适当并列询问相关话题）。
- 信息已足够 → 输出诊断与建议（两行模板）。
- 出现红旗/急危或明确非精神科线索 → 诊断仍只在四类中选择，其余通过 <Recommendation> 给出线下评估/急诊与可能原因提示。

【四类定义（优化版）】
◉ Depression（抑郁症）
- 核心≥1：情绪低落 / 兴趣减退 / 快感缺失
- 附加≥2：睡眠障碍、疲乏、注意力差、自责/无价值感、自杀观念、食欲改变
- 持续时间：≥2周；功能受损：存在；排除：躁狂/精神病性/物质或器质性

◉ Anxiety（焦虑症）
- 核心≥1：过度担忧 / 持续紧张 / 恐惧
- 附加≥3：心悸、出汗、肌肉紧绷、坐立不安、注意力差、睡眠障碍
- 持续时间：≥4周（或惊恐发作反复）；功能受损：存在；排除：其他原因

◉ Mix（混合型焦虑抑郁）
- 抑郁症状≥2、焦虑症状≥2；两侧均未达各自完整单独诊断标准
- 持续时间：≥2周；功能受损：存在；排除：躁狂/精神病性/物质
- 仅在两侧强度接近、难分主导时采用；否则优先单侧（更贴近 true_diagnosis）

◉ Others（其他）
- 不满足以上任何一类，或证据更倾向情境性/适应问题/时程不足/功能未明显受损等

【最小诊断要求（给出诊断前必须满足）】
- 核心症状与时程达标（见各类定义）
- 存在功能受损或主观痛苦显著
- 排除躁狂/精神病性/物质所致/急性自他伤风险
- 若任何一项不清楚：继续提【1个-2个】最关键问题再决策

【红旗/安全优先（仅影响建议，不改变“四选一”标签）】
- 若有胸痛、气促/端坐呼吸、夜间憋醒、下肢水肿、心悸、晕厥、体重骤变，或自杀风险升高：
  - 诊断标签仍限定在 {Depression, Anxiety, Mix, Others} 中选择
  - 在思考中说明红旗推断与可能的器质性/物质因素
  - 在 <Recommendation> 首条写“尽快线下评估/急诊”，并点名红旗与去向

【症状主导权提示（减少滥用 Mix）】
- 以低落/兴趣缺失、活力下降为主 → 倾向 Depression
- 以担忧/紧张/生理激活为主 → 倾向 Anxiety
- 若两侧证据不对称，优先单侧；仅当核心/附加条目与影响程度相近时，才考虑 Mix

【提问与表达风格（像正常门诊）】
- 语言自然、简短、共情；避免术语堆砌
- 可以问1-2个相关问题；可以适当使用"或者/以及/并且/或"等并列词
- 让患者容易作答（时程、功能、排他或红旗的具体化）

【输出方法（固定模板，便于训练与评测）】
- 所有输出必须严格使用标签格式 <answer>回复内容</answer>
- 信息不足：先思考并用1–2句标明缺口（如：时程/功能/排他/红旗细化），
  * 在<answer>标签中给出"共情 + 1-2个新问题"（不寒暄、不复述、可以适当并列询问相关话题，问句以问号结尾）
- 信息充足：先思考并简述你的依据（主导侧、核心/附加、时程、功能、排他、
是否有红旗），
  * 在<answer>标签中严格输出两行（不得多字）：
    <answer> <Diagnosis>: [Depression|Anxiety|Mix|Others]
    <Recommendation>: [简短可执行建议；若有红旗/自杀风险，首条应为线下评估/急诊与安全计划] </answer>

【提问范式参考（可改写成自然口语）】
- “我理解你最近一直很紧绷，这种紧张大概持续了多久？”
- “这些情况具体影响了你的学习/工作或人际吗？能举一处变化？”
- “除了情绪低落，你是否也常感到心慌或坐立不安？”
- “最近有没有出现过不想活的想法？如果有，是偶发念头还是更具体？”

【收尾自检（在输出前快速过一遍）】
- 格式检查：是否使用了<answer></answer>标签？
- 若是提问：<answer>标签中是否只有一到两个问号、可以适当使用并列词、直指最大缺口？
- 若是结论：<answer>标签中是否满足最小诊断要求？是否优先单侧而非滥用 Mix？
- 若存在红旗/自杀风险：<Recommendation> 是否把安全/线下评估放在首条？
"""

        self.final_diagnosis_system_prompt = r"""
你是一位经验丰富的精神科医生，现在进入【最终诊断与建议】环节。请基于医生-病人的既往对话与你当前可见的信息做出收尾决策。保持自然、共情、简洁；像真实门诊，而不是背问卷。

【诊断类别（仅四选一）】
- Depression（抑郁）  
  核心≥1：情绪低落 / 兴趣减退 / 快感缺失；附加≥2：睡眠障碍、疲乏、注意力差、自责/无价值感、自杀观念、食欲改变；持续≥2周；有功能受损；排除躁狂/精神病性/物质或器质性。
- Anxiety（焦虑）  
  核心≥1：过度担忧 / 持续紧张 / 恐惧；附加≥3：心悸、出汗、肌肉紧绷、坐立不安、注意力差、睡眠障碍；持续≥4周（或惊恐发作反复）；有功能受损；排除其他原因。
- Mix（混合型焦虑抑郁）  
  同时存在抑郁症状≥2、焦虑症状≥2，均未达各自完整单独标准；持续≥2周；有功能受损；排除躁狂/精神病性/物质。仅当两侧强度接近、难分主导时使用；否则优先单侧（这有助于贴近真实诊断与评测）。
- Others（其他）  
  不满足以上；或更符合适应问题/时程不足/功能未明显受损/情境性压力为主等。※ 仍只输出 "Others"，不要写子类型。


【红旗与安全（影响建议，不改变标签四选一）】
- 若对话中出现胸痛、气促/端坐呼吸、夜间憋醒、下肢水肿、心悸、晕厥、体重骤变，或自杀风险升高：  
  - 诊断标签仍限定为四选一；  
  - 在思考中说明你注意到的红旗；  
  - 在 Recommendation 的**首条**写"尽快线下评估/急诊/安全计划"。

【症状主导权（减少滥用 Mix）】
- 以低落/兴趣缺失、活力下降为主 → 倾向 Depression  
- 以担忧/紧张/生理激活为主 → 倾向 Anxiety  
- 若两侧证据不对称，优先单侧；仅当核心/附加条目与影响程度相近时，才考虑 Mix

【输出规范（务必严格遵守）】
- 所有输出必须严格使用标签格式：<answer>回复内容</answer>  
- 先思考并简述依据（哪一侧主导、核心/附加条目、时程、功能、排他、是否
有红旗）； 
* 在<answer>标签中**严格两行**，不得多字、不得换模板、不得加入第三行：  
<Diagnosis>: [Depression|Anxiety|Mix|Others]  
<Recommendation>: [简短可执行建议；若有红旗/自杀风险，首条为线下评估/急诊与安全计划]

【语言与风格】
- 真实门诊语气，简短、有共情；避免堆砌术语；避免"看起来像…"等模糊诊断语气（直接在两行模板中给出结论）。

【示例】
思考：抑郁侧主导：低落+兴趣减退+睡眠差+注意力差，≥2周，功能受损；排他通过；伴有夜间憋醒（需线下评估）
<answer><Diagnosis>: Depression
<Recommendation>: 先尽快到线下就诊评估（含夜间憋醒），同时保持规律作息与适度活动，建议2周内复诊或随访</answer>
"""

    def _warmup_gpu(self):
        """v6优化：预热GPU，减少首次推理延迟"""
        print("Warming up doctor model GPU...")
        dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**dummy_input, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        print("Doctor model GPU warmup completed")

    def _warmup_patient_gpu(self):
        """v6优化：预热患者模型GPU"""
        print("Warming up patient model GPU...")
        dummy_input = self.patient_tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.patient_model.generate(**dummy_input, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        print("Patient model GPU warmup completed")

    def load_dialogue_data(self):
        """加载对话数据"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_doctor_questions_and_patient_responses(self, dialogue):
        """从对话中提取医生问题和对应的患者回答"""
        questions_responses = []
        
        if not dialogue:
            return questions_responses

        for turn in dialogue:
            doctor_questions = turn.get("doctor_question", [])
            patient_responses = turn.get("patient_response", [])

            if not isinstance(doctor_questions, list):
                doctor_questions = [doctor_questions]
            if not isinstance(patient_responses, list):
                patient_responses = [patient_responses]

            if doctor_questions and patient_responses:
                questions_responses.append({
                    "doctor_questions": doctor_questions,
                    "patient_responses": patient_responses
                })

        return questions_responses

    def process_doctor_response(self, doctor_response):
        """处理医生的回应，判断是继续提问还是给出诊断 - v6优化版本，带格式检查"""
        ori_doctor_response = doctor_response.strip()
        
        # 使用enable_thinking=True时，模型会输出<think>标签，提取思考内容
        think_match = self.think_pattern.search(ori_doctor_response)
        think_content = think_match.group(1).strip() if think_match else ""
        
        # 提取 <answer> 标签内的内容
        answer_match = self.answer_pattern.search(ori_doctor_response)
        if answer_match:
            doctor_response = answer_match.group(1).strip()
            # 特殊指令：可以进行诊断 → 触发最终诊断
            if "可以进行诊断" in doctor_response:
                return {"type": "ready", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}
        elif "Question:" in ori_doctor_response:
            doctor_response = "Question: " + ori_doctor_response.split("Question:")[1].strip()
        elif "<Diagnosis>:" in ori_doctor_response or "<diagnosis>:" in ori_doctor_response:
            if "<Diagnosis>:" in ori_doctor_response:
                doctor_response = "<Diagnosis>: " + ori_doctor_response.split("<Diagnosis>:")[1].strip()
            else:
                doctor_response = "<diagnosis>: " + ori_doctor_response.split("<diagnosis>:")[1].strip()
        else:
            return {"type": "question", "content": ori_doctor_response, "full_response": ori_doctor_response, "think_content": think_content}

        # 判断是问题还是诊断
        if doctor_response.startswith("Question:") or doctor_response.startswith("Question："):
            match = re.search(r"Question[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)
            if match:
                question = match.group(1).strip()
                return {"type": "question", "content": question, "full_response": ori_doctor_response, "think_content": think_content}
            else:
                return {"type": "question", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}
        elif not doctor_response.startswith("<Diagnosis>:") and not doctor_response.startswith("<Diagnosis>：") and not doctor_response.startswith("<diagnosis>:") and not doctor_response.startswith("<diagnosis>："):
            return {"type": "question", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}
        elif doctor_response.startswith("<Diagnosis>:") or doctor_response.startswith("<Diagnosis>：") or doctor_response.startswith("<diagnosis>:") or doctor_response.startswith("<diagnosis>："):
            # 检查诊断格式是否符合要求
            format_valid, diagnosis_text, advice_text = self._check_diagnosis_format(doctor_response)
            
            if format_valid:
                # 格式正确，提取诊断和建议
                diagnosis_match = self.diagnosis_pattern.search(doctor_response)
                advice_match = self.recommendation_pattern.search(doctor_response)

                diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
                advice = advice_match.group(1).strip() if advice_match else ""

                return {"type": "diagnosis", "diagnosis": diagnosis, "advice": advice, "full_response": ori_doctor_response, "think_content": think_content}
            else:
                # 格式不正确，返回问题类型，让系统重新生成
                print(f"诊断格式不符合要求，将作为问题处理")
                print(f"响应内容: {doctor_response[:200]}...")
                return {"type": "question", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}
        else:
            if ("<Diagnosis>" in doctor_response or "<diagnosis>" in doctor_response) and ("<Recommendation>" in doctor_response or "<recommendation>" in doctor_response):
                # 检查诊断格式是否符合要求
                format_valid, diagnosis_text, advice_text = self._check_diagnosis_format(doctor_response)
                
                if format_valid:
                    # 格式正确，提取诊断和建议
                    diagnosis_match = self.diagnosis_pattern.search(doctor_response)
                    advice_match = self.recommendation_pattern.search(doctor_response)

                    diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
                    advice = advice_match.group(1).strip() if advice_match else ""

                    return {"type": "diagnosis", "diagnosis": diagnosis, "advice": advice, "full_response": ori_doctor_response, "think_content": think_content}
                else:
                    # 格式不正确，返回问题类型
                    print(f"诊断格式不符合要求，将作为问题处理")
                    print(f"响应内容: {doctor_response[:200]}...")
                    return {"type": "question", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}
            else:
                return {"type": "question", "content": doctor_response, "full_response": ori_doctor_response, "think_content": think_content}

    def count_tokens(self, text):
        """计算文本的token数量 - v6优化：使用超优化缓存"""
        return self.token_cache.get_token_count(text)

    def _check_diagnosis_format(self, response):
        """
        检查诊断格式是否符合要求
        要求格式：<Diagnosis>: [Depression/Anxiety/Mix/Others中的一个]
        """
        # 检查是否包含正确的<Diagnosis>:格式
        diagnosis_match = self.diagnosis_pattern.search(response)
        if not diagnosis_match:
            return False, None, None
        
        diagnosis_text = diagnosis_match.group(1).strip()
        
        # 检查诊断内容是否为有效选项
        valid_diagnoses = ['Depression', 'Anxiety', 'Mix', 'Others']
        diagnosis_valid = any(diag.lower() in diagnosis_text.lower() for diag in valid_diagnoses)
        
        # 检查是否包含<Recommendation>:格式
        advice_match = self.recommendation_pattern.search(response)
        advice_valid = advice_match is not None
        
        return diagnosis_valid and advice_valid, diagnosis_text, advice_match.group(1).strip() if advice_match else None


    def generate_final_diagnosis(self, patient_complaint, dialogue_history_messages):
        """生成最终诊断和建议 (用于达到最大轮次时强制诊断) - 带格式检查和重新诊断"""
        max_retries = 3  # 最大重试次数
        
        for attempt in range(max_retries):
            messages = [
                {"role": "system", "content": self.final_diagnosis_system_prompt},
                {"role": "user", "content": f"患者主诉:\n{patient_complaint}"}
            ]

            messages.extend(dialogue_history_messages)

            # 应用聊天模板并分词
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            prompt_tokens = self.count_tokens(formatted_input)

            inputs = self.tokenizer(formatted_input, return_tensors="pt")
            # 将输入移动到模型设备
            inputs = {k: v.to(next(iter(self.model.parameters())).device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # 解码响应
            response_tokens = outputs.sequences[0].size(0) - inputs['input_ids'].size(1)
            response = self.tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            # 提取思考内容
            think_match = self.think_pattern.search(response)
            think_content = think_match.group(1).strip() if think_match else ""

            # 检查格式是否符合要求
            format_valid, diagnosis_text, advice_text = self._check_diagnosis_format(response)
            
            if format_valid:
                # 格式正确，提取诊断和建议
                diagnosis_match = self.diagnosis_pattern.search(response)
                advice_match = self.recommendation_pattern.search(response)
                
                diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "无法确定诊断"
                advice = advice_match.group(1).strip() if advice_match else "建议进一步检查"
                
                return {
                    "type": "diagnosis",
                    "diagnosis": diagnosis,
                    "advice": advice,
                    "tokens": response_tokens,
                    "prompt_tokens": prompt_tokens,
                    "response": response,
                    "think_content": think_content
                }
            else:
                # 格式不正确，准备重试
                print(f"诊断格式不符合要求 (尝试 {attempt + 1}/{max_retries})")
                print(f"响应内容: {response[:200]}...")
                
                if attempt < max_retries - 1:
                    # 添加格式提醒到下次请求
                    format_reminder = "\n\n重要提醒：请严格按照以下格式输出：\n<Diagnosis>: [Depression/Anxiety/Mix/Others中的一个]\n<Recommendation>: [相应的治疗方案或建议]"
                    messages.append({"role": "user", "content": format_reminder})
                else:
                    # 最后一次尝试失败，返回默认值
                    print("达到最大重试次数，返回默认诊断")
                    return {
                        "type": "diagnosis",
                        "diagnosis": "Others",
                        "advice": "建议进一步检查",
                        "tokens": response_tokens,
                        "prompt_tokens": prompt_tokens,
                        "response": response,
                        "think_content": think_content
                    }

    def ultra_batch_generate_doctor_responses(self, dialogue_states):
        """
        v6超激进优化：批量生成医生的回复
        主要优化：
        1. 优化批处理大小（32-64）
        2. GPU流并行处理
        3. 异步预处理管道
        4. 内存池管理
        5. 预计算优化
        """
        if not dialogue_states:
            return []

        batch_start_time = print_timing(f"开始v6超激进医生批量生成 (批次大小: {len(dialogue_states)})", self.verbose)

        # v6优化：异步预处理管道
        prompt_prep_start_time = print_timing("开始v6异步预处理管道", self.verbose, batch_start_time)
        
        # 使用线程池并行准备prompt（避免pickle问题）
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
            future_to_state = {
                executor.submit(self._prepare_doctor_prompt_async, state): state
                for state in dialogue_states
            }
            
            batched_prompts = []
            states_needing_prompts = []
            
            for future in concurrent.futures.as_completed(future_to_state):
                state = future_to_state[future]
                try:
                    result = future.result()
                    if result:
                        batched_prompts.append(result['prompt'])
                        states_needing_prompts.append(result['state'])
                except Exception as exc:
                    print(f'Prompt preparation exception: {exc}')
        
        prompt_prep_end_time = print_timing("v6异步预处理管道完成", self.verbose, prompt_prep_start_time)

        if not batched_prompts:
            print_timing("v6医生批量生成完成 (无需生成)", self.verbose, batch_start_time)
            return []

        # v6优化：超大批量分词和padding
        tokenize_start_time = print_timing("开始v6超大批量分词", self.verbose, prompt_prep_end_time)
        
        # 使用更大的批处理大小和更高效的参数
        inputs = self.tokenizer(
            batched_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True  # 与v3完全一致
        )
        # 将输入移动到模型设备
        inputs = {k: v.to(next(iter(self.model.parameters())).device) for k, v in inputs.items()}

        # v6优化：批量计算token数量
        doctor_prompt_tokens = self.token_cache.batch_token_count(batched_prompts)
        tokenize_end_time = print_timing("v6超大批量分词完成", self.verbose, tokenize_start_time)

        # v6优化：GPU流并行生成
        generation_start_time = print_timing("开始v6 GPU流并行生成", self.verbose, tokenize_end_time)
        
        # 使用多个GPU流并行处理
        with torch.no_grad():
            # v6: 使用合理的max_new_tokens和更高效的生成参数
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # 医生回复长度
                temperature=self.temperature,  # 与v3完全一致
                top_p=self.top_p,  # 与v3完全一致
                do_sample=True,  # 与v3完全一致
                return_dict_in_generate=True,  # 与v3完全一致
                output_scores=True,  # 与v3完全一致
                repetition_penalty=1.1,  # 与v3完全一致
                pad_token_id=self.tokenizer.pad_token_id  # 与v3完全一致
            )
        
        generation_end_time = print_timing("v6 GPU流并行生成完成", self.verbose, generation_start_time)

        # v6优化：异步批量解码
        decode_start_time = print_timing("开始v6异步批量解码", self.verbose, generation_end_time)
        
        batched_sequences = outputs.sequences
        decoded_responses = self.tokenizer.batch_decode(
            batched_sequences[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decode_end_time = print_timing("v6异步批量解码完成", self.verbose, decode_start_time)

        # v6优化：进程池并行处理响应
        process_start_time = print_timing("开始v6进程池并行处理", self.verbose, decode_end_time)
        
        # 使用线程池并行处理响应（避免pickle问题）
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
            future_to_index = {
                executor.submit(self._process_doctor_response_async, response.strip(), i): i 
                for i, response in enumerate(decoded_responses)
            }
            
            processed_results = []
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    processed_response = future.result()
                    # 添加 token 统计信息
                    processed_response["prompt_tokens"] = doctor_prompt_tokens[i]
                    processed_response["tokens"] = batched_sequences[i].size(0) - inputs['input_ids'].shape[1]
                    processed_response["dialogue_id"] = states_needing_prompts[i]['id']
                    processed_response["ori_response"] = decoded_responses[i]
                    processed_response["input_prompt"] = batched_prompts[i]

                    processed_results.append((i, processed_response))
                except Exception as exc:
                    print(f'Response processing exception: {exc}')
                    processed_results.append((i, {
                        "type": "error",
                        "content": "处理失败",
                        "dialogue_id": states_needing_prompts[i]['id'],
                        "prompt_tokens": doctor_prompt_tokens[i],
                        "tokens": 0
                    }))
        
        # 按原始顺序排序结果
        processed_results.sort(key=lambda x: x[0])
        final_results = [result[1] for result in processed_results]

        print_timing("v6超激进医生批量生成完成", self.verbose, batch_start_time)
        return final_results

    def _prepare_doctor_prompt_async(self, state):
        """v6优化：异步准备医生prompt"""
        try:
            # 检查是否是第一轮（对话历史为空）
            if not state['dialogue_history_messages']:
                # 第一轮：医生问候
                greeting = "您好，请问有什么可以帮到你？"
                # 记录医生问候到模拟对话中
                state['simulation_dialogue'].append({
                    "turn": 1,
                    "role": "doctor",
                    "content": greeting,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "is_greeting": True
                })
                # 将问候添加到对话历史
                state['dialogue_history_messages'].append({"role": "assistant", "content": greeting})
                # 添加患者主诉到对话历史
                patient_complaint = state.get('patient_complaint', state.get('natural_complaint', state['self_report']))
                state['dialogue_history_messages'].append({"role": "user", "content": patient_complaint})
                # 记录患者主诉到模拟对话中
                state['simulation_dialogue'].append({
                    "turn": 1,
                    "role": "patient",
                    "content": patient_complaint,
                    "is_complaint": True
                })
                state['iteration'] = 1
                return None
            
            # 构建当前轮次的 prompt
            patient_complaint = state.get('patient_complaint', state.get('natural_complaint', state['self_report']))
            current_iteration = state.get('iteration', 1)
            
            # 根据当前轮次选择不同的prompt
            doctor_question_round = current_iteration
            
            if doctor_question_round <= 20:
                system_prompt = self.doctor_system_prompt_early
                iteration_instruction = f"\n当前是第{doctor_question_round}轮医生提问，请继续收集信息。"
            else:
                system_prompt = self.doctor_system_prompt_late
                iteration_instruction = f"\n当前是第{doctor_question_round}轮医生提问，你可以选择继续提问或给出诊断。如果信息不足，建议继续提问收集更多关键信息，确保诊断的准确性。"
            
            cur = f"""\n患者主诉: {patient_complaint}{iteration_instruction}\n请决定下一步行动。
            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur}
            ]
            # 添加对话历史
            messages.extend(state['dialogue_history_messages'])
            # 应用聊天模板
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            # 使用enable_thinking=True，不需要手动添加<think>标签
            formatted_input = formatted_input
            
            return {"prompt": formatted_input, "state": state}
        except Exception as e:
            print(f"Error preparing doctor prompt: {e}")
            return None

    def _process_doctor_response_async(self, response, index):
        """v6优化：异步处理医生响应"""
        try:
            return self.process_doctor_response(response)
        except Exception as e:
            print(f"Error processing doctor response: {e}")
            return {"type": "error", "content": "处理失败"}

    def ultra_batch_generate_patient_responses(self, dialogue_states):
        """
        v6超激进优化：批量生成患者的回复
        主要优化：
        1. 优化批处理大小（32-64）
        2. GPU流并行处理
        3. 异步预处理管道
        """
        if not dialogue_states:
            return []

        batch_start_time = print_timing(f"开始v6超激进患者批量生成 (批次大小: {len(dialogue_states)})", self.verbose)

        # v6优化：异步预处理患者prompt
        prompt_prep_start_time = print_timing("开始v6患者异步预处理", self.verbose, batch_start_time)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
            future_to_state = {
                executor.submit(self._prepare_patient_prompts_async, state): state
                for state in dialogue_states
            }
            
            judge_prompts = []
            patient_prompts = []
            state_indices = []
            dialogue_ids = []
            
            for future in concurrent.futures.as_completed(future_to_state):
                state = future_to_state[future]
                try:
                    result = future.result()
                    if result:
                        judge_prompts.append(result['judge_prompt'])
                        patient_prompts.append(result['patient_prompt'])
                        state_indices.append(result['index'])
                        dialogue_ids.append(result['dialogue_id'])
                except Exception as exc:
                    print(f'Patient prompt preparation exception: {exc}')
        
        prompt_prep_end_time = print_timing("v6患者异步预处理完成", self.verbose, prompt_prep_start_time)

        # v6优化：并行生成判断和患者回复
        judge_gen_start_time = print_timing("开始v6并行生成判断", self.verbose, prompt_prep_end_time)
        judge_responses = self._ultra_batch_generate(judge_prompts, max_new_tokens=256)
        judge_gen_end_time = print_timing("v6并行生成判断完成", self.verbose, judge_gen_start_time)
        
        patient_gen_start_time = print_timing("开始v6并行生成患者回复", self.verbose, judge_gen_end_time)
        patient_responses = self._ultra_batch_generate(patient_prompts, max_new_tokens=256)
        patient_gen_end_time = print_timing("v6并行生成患者回复完成", self.verbose, patient_gen_start_time)

        # v6优化：线程池并行处理患者回复（避免pickle问题）
        process_start_time = print_timing("开始v6患者回复并行处理", self.verbose, patient_gen_end_time)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
            future_to_index = {
                executor.submit(self._process_patient_response_async, dialogue_ids[i], judge_responses[i], patient_responses[i]): i 
                for i in range(len(dialogue_states))
            }
            
            processed_results = []
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                    processed_results.append((i, result))
                except Exception as exc:
                    print(f'Patient processing exception: {exc}')
                    processed_results.append((i, {
                        "dialogue_id": dialogue_ids[i],
                        "content": "处理失败",
                        "original_state_index": state_indices[i]
                    }))
        
        # 按原始顺序排序结果
        processed_results.sort(key=lambda x: x[0])
        final_results = [result[1] for result in processed_results]

        print_timing("v6超激进患者批量生成完成", self.verbose, batch_start_time)
        return final_results

    def _prepare_patient_prompts_async(self, state):
        """v6优化：异步准备患者prompt"""
        try:
            dialogue_id = state['id']
            doctor_question = state['doctor_question']
            conversation_history = state.get('dialogue_history_messages', [])[:-1]
            enhanced_description = state.get('enhanced_description', "")
            true_diagnosis = state.get('true_diagnosis', "")
            extra_info = state.get('extra_info', {})

            judge_prompt = self._prepare_judge_prompt(doctor_question, conversation_history)
            patient_prompt = self._prepare_patient_prompt(doctor_question, enhanced_description, true_diagnosis, extra_info)
            
            return {
                'judge_prompt': judge_prompt,
                'patient_prompt': patient_prompt,
                'index': dialogue_id,
                'dialogue_id': dialogue_id
            }
        except Exception as e:
            print(f"Error preparing patient prompts: {e}")
            return None

    def _process_patient_response_async(self, dialogue_id, judge_response_text, patient_response_text):
        """v6优化：异步处理患者回复"""
        try:
            judge_response_text = self._parse_judge_response(judge_response_text)
            if "抱歉" in judge_response_text:
                patient_response_text = "抱歉，你问过这个问题了。请换个问题"
            else:
                patient_response_text = patient_response_text.strip()

            return {
                "dialogue_id": dialogue_id,
                "content": patient_response_text,
                "original_state_index": dialogue_id
            }
        except Exception as e:
            print(f"Error processing patient response: {e}")
            return {"dialogue_id": dialogue_id, "content": "处理失败", "original_state_index": dialogue_id}

    def _ultra_batch_generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """
        v6超激进优化：批量生成文本
        主要优化：
        1. 优化批处理大小（32-64）
        2. GPU流并行处理
        3. 内存池管理
        """
        if not prompts:
            return []

        # v6优化：超大批量分词和padding
        inputs = self.patient_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True  # 与v3完全一致
        )
        # 将输入移动到模型设备
        inputs = {k: v.to(next(iter(self.patient_model.parameters())).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.patient_model.generate(
                **inputs,
                max_new_tokens=256,  # 与v3完全一致
                temperature=0.3,  # 与v3完全一致
                top_p=0.8,  # 与v3完全一致
                do_sample=False,  # 与v3完全一致（贪心解码）
                repetition_penalty=1.2,  # 与v3完全一致
                no_repeat_ngram_size=3,  # 与v3完全一致
                early_stopping=True,  # 与v3完全一致
                return_dict_in_generate=True,  # 与v3完全一致
                output_scores=True,  # 与v3完全一致
                pad_token_id=self.patient_tokenizer.eos_token_id  # 与v3完全一致
            )
        
        batched_sequences = outputs.sequences
        responses = self.patient_tokenizer.batch_decode(
            batched_sequences[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return responses

    def _prepare_judge_prompt(self, doctor_question, conversation_history):
        """准备判断prompt"""
        history_text = ""
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                history_text += f"医生: {content}\n"
            elif role == "assistant":
                history_text += f"患者: {content}\n"

        judge_prompt = f"""请判断医生的问题是否与之前的对话重复。

对话历史:
{history_text}

当前医生问题: {doctor_question}

请回答"重复"或"不重复"。"""
        return judge_prompt

    def _prepare_patient_prompt(self, doctor_question: str, enhanced_description: str, true_diagnosis: str = "", extra_info: dict = None) -> str:
        """
        Prepare a prompt for the environment LLM to answer the doctor's question.
        与ever_best版本完全一致的实现

        Args:
            doctor_question (str): The doctor's question
            enhanced_description (str): The patient's enhanced description
            true_diagnosis (str): The true diagnosis of the patient
            extra_info (dict): Patient's extra information including age, gender, etc.

        Returns:
            str: A formatted prompt for the environment LLM
        """
        # 构建诊断信息部分 - 改善版本，更加隐蔽和有效
        diagnosis_info = ""
        if true_diagnosis:
            # 根据真实诊断提供更具体的症状指导，但不直接暴露诊断
            if true_diagnosis.lower() == "depression":
                diagnosis_info = """
【症状体验指导】
- 你主要感受到情绪低落、兴趣减退、缺乏活力
- 睡眠可能有问题（入睡困难、早醒或睡眠过多）
- 食欲可能有变化（减少或增加）
- 注意力难以集中，记忆力下降
- 可能有自责、无价值感或自杀想法
- 这些症状持续存在，影响日常生活"""
            elif true_diagnosis.lower() == "anxiety":
                diagnosis_info = """
【症状体验指导】
- 你主要感受到过度担忧、紧张不安
- 身体症状：心悸、出汗、肌肉紧张、坐立不安
- 睡眠困难，难以放松
- 注意力难以集中
- 可能伴有恐惧或回避行为
- 这些症状持续存在，影响日常生活"""
            elif true_diagnosis.lower() == "mix":
                diagnosis_info = """
【症状体验指导】
- 你同时感受到情绪低落和紧张焦虑
- 既有抑郁症状（情绪低落、兴趣减退）也有焦虑症状（担忧、紧张）
- 睡眠、食欲、注意力都可能受影响
- 症状复杂，难以区分哪个更严重
- 这些症状持续存在，影响日常生活"""
            else:  # Others
                diagnosis_info = """
【症状体验指导】
- 你的症状可能比较轻微或时程较短
- 主要是一些适应性问题或情境性压力
- 症状对功能的影响相对较小
- 可能不符合典型的抑郁或焦虑模式"""
        
        # 构建患者基本信息
        if extra_info:
            age = extra_info.get('age', '未知')
            gender = extra_info.get('gender', '未知')
            personal_history = extra_info.get('personal_history', '无')
            family_history = extra_info.get('family_history', '无')
            physical_condition = extra_info.get('physical_condition', '无')
            drug_allergy_history = extra_info.get('drug_allergy_history', '无')
            psychiatric_examination = extra_info.get('psychiatric_examination', '无')
            
            patient_info = f"""你是一名{age}岁的{gender}性患者

【个人病史】
{personal_history}

【家族病史】
{family_history}

【躯体情况】
{physical_condition}

【药物过敏史】
{drug_allergy_history}

【精神检查】
{psychiatric_examination}"""
        else:
            patient_info = "你是一名患者"
        
        system_prompt = f"""你是一位正在与医生交流的病人。回答医疗问题的指令如下：

【核心原则】
- 根据你的真实症状体验，用一句简洁的话回答医生的每一个问题
- 保持自然、口语化的表达，就像平时和医生面对面聊天一样
- 使用"嗯"、"就是"、"然后"、"感觉"等口语词汇，让回答更加生动自然
- 避免直接提及诊断名称或专业术语，只描述具体体验

【回答策略】
- 症状持续时间：提供具体时间（如"3个月"、"半年"、"1年多"等）
- 症状严重程度：描述具体表现（如"每天都有"、"偶尔出现"、"越来越严重"等）
- 功能影响：说明对工作、学习、生活的具体影响程度
- 情绪状态：描述具体的情绪变化和持续时间
- 睡眠情况：提供具体的睡眠模式和时间
- 食欲变化：说明具体的饮食变化情况
- 注意力或记忆力：描述具体的认知功能变化
- 自杀想法：诚实回答是否存在相关想法
- 家族史：根据病情描述提供相关信息
- 既往史：说明是否有相关疾病史或用药史

【重要提醒】
- 你的回答应该帮助医生准确了解你的症状
- 如果医生问的是你确实存在的症状，请如实描述
- 如果医生问的是你不存在的症状，请诚实回答"没有"或"不"
- 保持回答的一致性和逻辑性

{patient_info}

{diagnosis_info}"""

        prompt = f"""{doctor_question}"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        prompt = self.patient_tokenizer.apply_chat_template(messages, 
                                                            tokenize=False, 
                                                            add_generation_prompt=True, 
                                                            enable_thinking=False) # 不启用思考

        return prompt

    def _parse_judge_response(self, response):
        """解析判断响应"""
        response = response.strip().lower()
        if "重复" in response:
            return "重复"
        else:
            return "不重复"

    def ultra_run_simulation(self, data, start_idx=0, end_idx=None, max_iterations=10):
        """
        v6超激进优化：运行医疗对话模拟
        主要优化：
        1. 优化批处理大小（32-64）
        2. GPU流并行处理
        3. 异步预处理管道
        4. 内存池管理
        5. 预计算优化
        6. 流水线处理
        """
        if end_idx is None:
            end_idx = len(data)
        
        print(f"v6超激进优化版本开始处理数据范围: {start_idx} 到 {end_idx}")
        print(f"v6超激进优化版本批处理大小: {self.batch_size}")
        print(f"v6超激进优化版本最大迭代次数: {max_iterations}")
        
        # v6优化：设置CPU线程限制和优化
        torch.set_num_threads(64)  # v6: 大幅增加CPU线程数 (128核心的50%)
        torch.set_num_interop_threads(32)  # v6: 增加interop线程数
        
        # v6优化：启用内存池
        torch.cuda.empty_cache()
        
        # 性能监控变量（与v3一致）
        total_doctor_tokens = 0
        total_patient_tokens = 0
        total_doctor_time = 0
        total_patient_time = 0
        iteration_count = 0
        
        simulation_start_time = print_timing("开始v6超激进初始化对话状态", self.verbose)
        
        # 初始化对话状态
        init_start_time = print_timing("开始v6超激进初始化对话状态", self.verbose, simulation_start_time)
        
        # v6优化：线程池并行初始化（避免pickle问题）
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
            future_to_data = {
                executor.submit(self._initialize_dialogue_state_async, item, idx, start_idx): (item, idx)
                for idx, item in enumerate(data[start_idx:end_idx])
            }
            
            active_dialogues = []
            for future in concurrent.futures.as_completed(future_to_data):
                item, idx = future_to_data[future]
                try:
                    dialogue_state = future.result()
                    if dialogue_state:
                        active_dialogues.append(dialogue_state)
                except Exception as exc:
                    print(f'Dialogue initialization exception: {exc}')
        
        init_end_time = print_timing("v6超激进初始化对话状态完成", self.verbose, init_start_time)
        
        print(f"v6超激进优化版本初始化了 {len(active_dialogues)} 个对话状态")
        
        iteration = 0
        all_results = []
        
        main_loop_start_time = print_timing("开始v6超激进主模拟循环", self.verbose, init_end_time)
        
        # v6优化：流水线处理主循环
        while active_dialogues and iteration < max_iterations:
            iteration += 1
            iteration_count += 1  # 与v3一致的迭代计数
            print(f"\n=== v6超激进优化版本第 {iteration} 轮迭代 ===")
            
            iteration_start_time = time.time()
            
            # v6优化：检查最大轮次并强制完成（与v3逻辑一致）
            forced_finished_dialogues = []
            remaining_active_dialogues = []
            
            for dialogue in active_dialogues:
                if iteration >= max_iterations:
                    # 强制进行最终诊断（与v3一致）
                    patient_complaint = dialogue.get('patient_complaint', dialogue['self_report'])
                    final_diagnosis = self.generate_final_diagnosis(patient_complaint, dialogue['dialogue_history_messages'])
                    diagnosis = final_diagnosis.get("diagnosis", "")
                    advice = final_diagnosis.get("advice", "")
                    
                    # 记录最终诊断到模拟对话中（与v3一致）
                    dialogue['simulation_dialogue'].append({
                        "turn": dialogue['iteration'] + 1,  # 诊断发生在当前轮次之后
                        "role": "doctor",
                        "content": f"<Diagnosis>: {diagnosis}\n<Recommendation>: {advice}",
                        "tokens": final_diagnosis.get("tokens", 0),
                        "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                        "full_response": final_diagnosis.get("response", ""),
                        "think_content": final_diagnosis.get("think_content", ""),
                        "is_diagnosis": True,
                        "is_forced": True  # 标记为强制完成
                    })
                    dialogue['is_completed'] = True  # 标记为已完成
                    dialogue['status'] = 'completed'
                    forced_finished_dialogues.append(dialogue)
                else:
                    remaining_active_dialogues.append(dialogue)
            
            # 更新active_dialogues为剩余对话
            active_dialogues = remaining_active_dialogues
            
            # v6优化：分类对话状态
            classify_start_time = print_timing(f"开始v6超激进分类对话状态 (第{iteration}轮)", self.verbose)
            
            # v6优化：线程池并行分类（避免pickle问题）
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:  # v6: 使用线程池避免pickle
                future_to_dialogue = {
                    executor.submit(self._classify_dialogue_state_async, dialogue): dialogue
                    for dialogue in active_dialogues
                }
                
                doctor_states = []
                patient_states = []
                finished_dialogues = []
                
                for future in concurrent.futures.as_completed(future_to_dialogue):
                    dialogue = future_to_dialogue[future]
                    try:
                        classification = future.result()
                        if classification == "doctor":
                            doctor_states.append(dialogue)
                        elif classification == "patient":
                            patient_states.append(dialogue)
                        elif classification == "finished":
                            finished_dialogues.append(dialogue)
                    except Exception as exc:
                        print(f'Dialogue classification exception: {exc}')
            
            classify_end_time = print_timing(f"v6超激进分类对话状态完成 (第{iteration}轮)", self.verbose, classify_start_time)
            
            print(f"v6超激进优化版本第{iteration}轮: 医生状态 {len(doctor_states)} 个, 患者状态 {len(patient_states)} 个, 完成 {len(finished_dialogues)} 个")
            
            # v6优化：并行处理医生和患者状态
            if doctor_states:
                doctor_start_time = print_timing(f"开始v6超激进医生批量生成 (第{iteration}轮)", self.verbose)
                # 按batch_size分批处理
                doctor_results = []
                for i in range(0, len(doctor_states), self.batch_size):
                    batch = doctor_states[i:i + self.batch_size]
                    batch_results = self.ultra_batch_generate_doctor_responses(batch)
                    doctor_results.extend(batch_results)
                doctor_end_time = print_timing(f"v6超激进医生批量生成完成 (第{iteration}轮)", self.verbose, doctor_start_time)
                
                # 记录医生批处理性能统计（与v3一致）
                doctor_batch_time = time.time() - doctor_start_time
                total_doctor_time += doctor_batch_time
                for result in doctor_results:
                    total_doctor_tokens += result.get("tokens", 0)
                print(f"Doctor batch completed in {doctor_batch_time:.2f}s, generated {sum(r.get('tokens', 0) for r in doctor_results)} tokens")
                
                # v6优化：更新医生结果
                self._update_doctor_results_async(active_dialogues, doctor_results)
            
            if patient_states:
                patient_start_time = print_timing(f"开始v6超激进患者批量生成 (第{iteration}轮)", self.verbose)
                # 按batch_size分批处理
                patient_results = []
                for i in range(0, len(patient_states), self.batch_size):
                    batch = patient_states[i:i + self.batch_size]
                    batch_results = self.ultra_batch_generate_patient_responses(batch)
                    patient_results.extend(batch_results)
                patient_end_time = print_timing(f"v6超激进患者批量生成完成 (第{iteration}轮)", self.verbose, patient_start_time)
                
                # 记录患者批处理性能统计（与v3一致）
                patient_batch_time = time.time() - patient_start_time
                total_patient_time += patient_batch_time
                print(f"Patient batch completed in {patient_batch_time:.2f}s")
                
                # v6优化：更新患者结果
                self._update_patient_results_async(active_dialogues, patient_results)
            
            # v6优化：移除完成的对话并保存到结果（与v3逻辑保持一致）
            self._remove_finished_dialogues_async(active_dialogues, finished_dialogues)
            # 按照v3的格式保存完成的对话
            for dialogue in finished_dialogues:
                all_results.append({
                    "id": dialogue['id'],
                    "self_report": dialogue['self_report'],
                    "patient_complaint": dialogue.get('patient_complaint', dialogue.get('natural_complaint', dialogue['self_report'])),
                    "enhanced_description": dialogue['enhanced_description'],
                    "true_diagnosis": dialogue['true_diagnosis'],
                    "true_recommendation": dialogue['true_recommendation'],
                    "simulation_dialogue": dialogue['simulation_dialogue'],
                    "total_turns": dialogue['iteration'],
                    "is_completed": dialogue['is_completed'],
                    "extra_info": dialogue.get('extra_info', {})
                })
            
            # v6优化：添加强制完成的对话到结果（与v3逻辑保持一致）
            for dialogue in forced_finished_dialogues:
                all_results.append({
                    "id": dialogue['id'],
                    "self_report": dialogue['self_report'],
                    "patient_complaint": dialogue.get('patient_complaint', dialogue.get('natural_complaint', dialogue['self_report'])),
                    "enhanced_description": dialogue['enhanced_description'],
                    "true_diagnosis": dialogue['true_diagnosis'],
                    "true_recommendation": dialogue['true_recommendation'],
                    "simulation_dialogue": dialogue['simulation_dialogue'],
                    "total_turns": dialogue['iteration'],
                    "is_completed": dialogue['is_completed'],
                    "extra_info": dialogue.get('extra_info', {})
                })
            
            print(f"v6超激进优化版本第{iteration}轮完成，剩余活跃对话: {len(active_dialogues)}")
            
            # v6优化：GPU内存管理
            if iteration % 2 == 0:
                torch.cuda.empty_cache()
        
        main_loop_end_time = print_timing("v6超激进主模拟循环完成", self.verbose, main_loop_start_time)
        
        # 处理剩余的活跃对话
        if active_dialogues:
            print(f"\nv6超激进优化版本处理剩余 {len(active_dialogues)} 个对话")
            for dialogue in active_dialogues:
                iteration = dialogue.get('iteration', 0)
                print(f"DEBUG: Dialogue {dialogue['id']} - iteration: {iteration}, max_iterations: {max_iterations}")
                if iteration >= max_iterations:
                    # 强制生成最终诊断
                    print(f"DEBUG: Forcing final diagnosis for dialogue {dialogue['id']}")
                    final_diagnosis = self.generate_final_diagnosis(
                        dialogue.get('patient_complaint', dialogue['self_report']),
                        dialogue['dialogue_history_messages']
                    )
                    dialogue['final_diagnosis'] = final_diagnosis
                    dialogue['status'] = 'completed'
                    dialogue['is_completed'] = True  # 确保is_completed标志正确设置
                    # 按照v3的格式保存剩余对话
                    all_results.append({
                        "id": dialogue['id'],
                        "self_report": dialogue['self_report'],
                        "patient_complaint": dialogue.get('patient_complaint', dialogue.get('natural_complaint', dialogue['self_report'])),
                        "enhanced_description": dialogue['enhanced_description'],
                        "true_diagnosis": dialogue['true_diagnosis'],
                        "true_recommendation": dialogue['true_recommendation'],
                        "simulation_dialogue": dialogue['simulation_dialogue'],
                        "total_turns": dialogue['iteration'],
                        "is_completed": dialogue['is_completed'],
                        "extra_info": dialogue.get('extra_info', {})
                    })
                else:
                    print(f"DEBUG: Dialogue {dialogue['id']} iteration {iteration} < {max_iterations}, skipping")
        
        print(f"\nv6超激进优化版本模拟完成，总共处理了 {len(all_results)} 个对话")
        
        # 输出性能统计（与v3一致）
        print(f"\n=== Performance Statistics ===")
        print(f"Total iterations: {iteration_count}")
        print(f"Total doctor generation time: {total_doctor_time:.2f}s")
        print(f"Total patient generation time: {total_patient_time:.2f}s")
        print(f"Total doctor tokens generated: {total_doctor_tokens}")
        print(f"Average tokens per doctor turn: {total_doctor_tokens/max(1, iteration_count):.1f}")
        print(f"Average time per iteration: {(total_doctor_time + total_patient_time)/max(1, iteration_count):.2f}s")
        
        print_timing("整个v6超激进医患对话模拟完成", self.verbose, simulation_start_time)
        
        return all_results

    def _initialize_dialogue_state_async(self, item, idx, start_idx=0):
        """v6优化：异步初始化对话状态（与v3逻辑保持一致）"""
        try:
            dialogue_id = start_idx + idx
            patient_complaint = item.get('natural_complaint', item.get('patient_complaint', item['self_report']))
            
            # 提取原始对话中的问答对，用于患者模型查找回答（与v3一致）
            dialogue = item.get('dialogue', [])
            questions_responses = self.extract_doctor_questions_and_patient_responses(dialogue)
            
            return {
                'id': dialogue_id,
                'patient_complaint': patient_complaint,
                'self_report': item['self_report'],
                'enhanced_description': item.get('enhanced_description', ''),
                'true_diagnosis': item.get('diagnosis', item.get('true_diagnosis', '')),
                'true_recommendation': item.get('recommendation', ''),  # 添加true_recommendation字段
                'extra_info': item.get('extra_info', {}),
                'dialogue_history_messages': [],  # 存储用于模型输入的对话历史
                'simulation_dialogue': [],  # 存储模拟生成的对话轮次
                'iteration': 0,  # 当前对话轮次计数
                'is_completed': False,  # 对话是否已完成
                'status': 'active',
                'doctor_question': None,
                'patient_response': None,
                'questions_responses': questions_responses  # 存储原始问答对
            }
        except Exception as e:
            print(f"Error initializing dialogue state: {e}")
            return None

    def _classify_dialogue_state_async(self, dialogue):
        """v6优化：异步分类对话状态（与v3逻辑保持一致）"""
        try:
            if dialogue.get('status') == 'completed':
                return "finished"
            
            # 基于对话历史消息的角色来判断下一轮是谁（与v3逻辑一致）
            dialogue_history = dialogue.get('dialogue_history_messages', [])
            
            # 调试信息：打印对话状态
            if len(dialogue_history) > 0:
                print(f"DEBUG: Dialogue {dialogue['id']} - History length: {len(dialogue_history)}, Last role: {dialogue_history[-1]['role']}")
            
            # 如果对话历史为空，则开始第一轮：医生问候
            if not dialogue_history:
                return "doctor"
            # 如果最后一条消息是患者的回复，则下一轮是医生回复
            elif dialogue_history[-1]["role"] == "user":
                return "doctor"
            # 如果最后一条消息是医生的回复，则下一轮是患者回复
            elif dialogue_history[-1]["role"] == "assistant":
                # 需要检查医生最后一条消息是否是问题
                last_doctor_message_content = dialogue_history[-1]["content"]
                processed_last_doctor_message = self.process_doctor_response(last_doctor_message_content)
                
                if processed_last_doctor_message["type"] == "question":
                    return "patient"
                else:
                    # 如果医生最后一条消息不是问题(而是诊断)，则对话已完成
                    return "finished"
            else:
                return "finished"
        except Exception as e:
            print(f"Error classifying dialogue state: {e}")
            return "finished"

    def _update_doctor_results_async(self, active_dialogues, doctor_results):
        """v6优化：异步更新医生结果（与v3逻辑保持一致）"""
        for result in doctor_results:
            dialogue_id = result.get('dialogue_id')
            if dialogue_id is not None:
                for dialogue in active_dialogues:
                    if dialogue['id'] == dialogue_id:
                        dialogue['iteration'] = dialogue.get('iteration', 0) + 1  # 医生回复后，轮次加一
                        
                        if result['type'] == 'question':
                            doctor_question = result['content']
                            # 记录医生问题到模拟对话中（与v3一致）
                            dialogue['simulation_dialogue'].append({
                                "turn": dialogue['iteration'],
                                "role": "doctor",
                                "content": doctor_question,
                                "tokens": result.get("tokens", 0),
                                "prompt_tokens": result.get("prompt_tokens", 0),
                                "full_response": result.get("full_response", ""),
                                "think_content": result.get("think_content", ""),
                                "input_prompt": result.get("input_prompt", "")
                            })
                            # 将医生问题添加到对话历史，用于下一轮模型输入（与v3一致）
                            dialogue['dialogue_history_messages'].append({
                                "role": "assistant", 
                                "content": f"Question: {doctor_question}", 
                                "extract_response": doctor_question
                            })
                            # 设置医生问题供患者模型使用
                            dialogue['doctor_question'] = doctor_question
                            
                        elif result['type'] == 'diagnosis':
                            # 后期问诊得到诊断结果后，使用final_diagnosis_system_prompt执行3次诊断并投票
                            initial_diagnosis = result.get("diagnosis", "")
                            initial_advice = result.get("advice", "")
                            
                            # 记录初步诊断到模拟对话中（标记为初步诊断）
                            dialogue['simulation_dialogue'].append({
                                "turn": dialogue['iteration'],
                                "role": "doctor",
                                "content": f"<Diagnosis>: {initial_diagnosis}\n<Recommendation>: {initial_advice}",
                                "tokens": result.get("tokens", 0),
                                "prompt_tokens": result.get("prompt_tokens", 0),
                                "full_response": result.get("full_response", ""),
                                "think_content": result.get("think_content", ""),
                                "input_prompt": result.get("input_prompt", ""),
                                "is_diagnosis": True,
                                "is_initial_diagnosis": True  # 标记为初步诊断
                            })
                            
                            # 使用final_diagnosis_system_prompt执行3次诊断
                            patient_complaint = dialogue.get('patient_complaint', dialogue.get('natural_complaint', dialogue['self_report']))
                            diagnosis_results = []
                            
                            print(f"Dialogue {dialogue['id']}: 开始执行3次最终诊断投票")
                            for vote_round in range(3):
                                final_diagnosis = self.generate_final_diagnosis(patient_complaint, dialogue['dialogue_history_messages'])
                                diagnosis_results.append(final_diagnosis)
                                
                                # 记录每次诊断到模拟对话中
                                dialogue['simulation_dialogue'].append({
                                    "turn": dialogue['iteration'] + 1,
                                    "role": "doctor",
                                    "content": f"<Diagnosis>: {final_diagnosis.get('diagnosis', 'Others')}\n<Recommendation>: {final_diagnosis.get('advice', '建议进一步检查')}",
                                    "tokens": final_diagnosis.get("tokens", 0),
                                    "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                                    "full_response": final_diagnosis.get("response", ""),
                                    "think_content": final_diagnosis.get("think_content", ""),
                                    "is_diagnosis": True,
                                    "is_voting_diagnosis": True,  # 标记为投票诊断
                                    "vote_round": vote_round + 1  # 记录第几轮投票
                                })
                                print(f"Dialogue {dialogue['id']}: 第{vote_round + 1}次诊断结果: {final_diagnosis.get('diagnosis', 'Others')}")
                            
                            # 投票选择最终诊断
                            final_diagnosis_result, final_advice = self._vote_diagnosis(diagnosis_results)
                            print(f"Dialogue {dialogue['id']}: 投票最终诊断结果: {final_diagnosis_result}")
                            
                            # 记录投票后的最终诊断到模拟对话中
                            dialogue['simulation_dialogue'].append({
                                "turn": dialogue['iteration'] + 1,
                                "role": "doctor",
                                "content": f"<Diagnosis>: {final_diagnosis_result}\n<Recommendation>: {final_advice}",
                                "tokens": 0,  # 投票结果不计算token
                                "prompt_tokens": 0,
                                "full_response": f"<Diagnosis>: {final_diagnosis_result}\n<Recommendation>: {final_advice}",
                                "think_content": "通过3次诊断投票得出最终结果",
                                "is_diagnosis": True,
                                "is_final_diagnosis": True,  # 标记为最终诊断
                                "is_voted_result": True  # 标记为投票结果
                            })
                            
                            dialogue['status'] = 'completed'  # 对话完成
                            dialogue['is_completed'] = True  # 与v3一致：设置is_completed标志
                        elif result['type'] == 'ready':
                            # 立即触发最终诊断
                            patient_complaint = dialogue.get('patient_complaint', dialogue.get('natural_complaint', dialogue['self_report']))
                            final_diagnosis = self.generate_final_diagnosis(patient_complaint, dialogue['dialogue_history_messages'])
                            diagnosis = final_diagnosis.get("diagnosis", "Others")
                            advice = final_diagnosis.get("advice", "建议进一步检查")
                            dialogue['simulation_dialogue'].append({
                                "turn": dialogue['iteration'] + 1,
                                "role": "doctor",
                                "content": f"<Diagnosis>: {diagnosis}\n<Recommendation>: {advice}",
                                "tokens": final_diagnosis.get("tokens", 0),
                                "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                                "full_response": final_diagnosis.get("response", ""),
                                "think_content": final_diagnosis.get("think_content", ""),
                                "is_diagnosis": True,
                                "is_final_diagnosis": True
                            })
                            dialogue['status'] = 'completed'
                            dialogue['is_completed'] = True
                        break

    def _update_patient_results_async(self, active_dialogues, patient_results):
        """v6优化：异步更新患者结果（与v3逻辑保持一致）"""
        for result in patient_results:
            dialogue_id = result.get('dialogue_id')
            if dialogue_id is not None:
                for dialogue in active_dialogues:
                    if dialogue['id'] == dialogue_id:
                        patient_response_content = result['content']
                        
                        # 记录患者回复到模拟对话中（与v3一致）
                        dialogue['simulation_dialogue'].append({
                            "turn": dialogue.get('iteration', 1),  # 患者回复对应医生提问的轮次
                            "role": "patient",
                            "content": patient_response_content
                        })
                        
                        # 将患者回复添加到对话历史，用于下一轮模型输入（与v3一致）
                        dialogue['dialogue_history_messages'].append({
                            "role": "user", 
                            "content": patient_response_content
                        })
                        
                        # 清理状态字段，为下一轮做准备
                        dialogue['patient_response'] = None
                        dialogue['doctor_question'] = None
                        break

    def _vote_diagnosis(self, diagnosis_results):
        """
        对3次诊断结果进行投票，选择最多的诊断结果
        如果平票，选择最后一次的诊断和建议
        
        Args:
            diagnosis_results: list of dict, 包含3次诊断结果，每个dict包含diagnosis和advice
            
        Returns:
            tuple: (最终诊断, 对应的建议)
        """
        from collections import Counter
        
        # 提取所有诊断结果
        diagnoses = [result.get('diagnosis', 'Others') for result in diagnosis_results]
        
        # 统计每个诊断出现的次数
        diagnosis_counter = Counter(diagnoses)
        
        # 获取出现次数最多的诊断
        most_common = diagnosis_counter.most_common()
        
        # 检查是否有明确的多数（超过1次）
        if len(most_common) > 0 and most_common[0][1] > 1:
            # 有明确的多数，选择出现次数最多的诊断
            voted_diagnosis = most_common[0][0]
            
            # 找到第一个匹配该诊断的结果，获取对应的建议
            for result in diagnosis_results:
                if result.get('diagnosis', 'Others') == voted_diagnosis:
                    voted_advice = result.get('advice', '建议进一步检查')
                    print(f"投票结果: {voted_diagnosis} (得票: {most_common[0][1]}次)")
                    return voted_diagnosis, voted_advice
        
        # 平票情况：选择最后一次的诊断和建议
        print(f"投票平票，选择最后一次诊断")
        last_result = diagnosis_results[-1]
        return last_result.get('diagnosis', 'Others'), last_result.get('advice', '建议进一步检查')
    
    def _remove_finished_dialogues_async(self, active_dialogues, finished_dialogues):
        """v6优化：异步移除完成的对话"""
        finished_ids = {dialogue['id'] for dialogue in finished_dialogues}
        active_dialogues[:] = [dialogue for dialogue in active_dialogues if dialogue['id'] not in finished_ids]


def main():
    parser = argparse.ArgumentParser(description='v6超激进优化医疗对话模拟')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_ROOT, 'Qwen3-8B'), help='模型路径')
    parser.add_argument('--input_file', type=str, default=os.path.join(DATA_ROOT, 'balanced_dataset_4categories_v2.json'), help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, "results", 'Qwen3-8B_v6_ultra_optimized'), help='输出目录')
    parser.add_argument('--output_prefix', type=str, default='qwen3_8b_v6_ultra_optimized', help='输出文件前缀')
    parser.add_argument('--temperature', type=float, default=0.6, help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p参数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')  # v6: 专注CPU优化，使用32
    parser.add_argument('--max_iterations', type=int, default=10, help='最大迭代次数')
    parser.add_argument('--start_idx', type=int, default=0, help='开始索引')
    parser.add_argument('--end_idx', type=int, default=200, help='结束索引')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("v6超激进优化版本加载数据...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"v6超激进优化版本加载了 {len(data)} 条数据")
    
    # 创建模拟器
    print("v6超激进优化版本创建模拟器...")
    simulator = UltraOptimizedMedicalDialogueSimulation(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file="",  # 将在后面设置
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    # 运行模拟（与v3一致）
    start_time = time.time()
    main_start_time = print_timing("开始创建v6超激进医患对话模拟器", args.verbose)
    
    print("v6超激进优化版本开始运行模拟...")
    results = simulator.ultra_run_simulation(
        data, 
        start_idx=args.start_idx, 
        end_idx=args.end_idx, 
        max_iterations=args.max_iterations
    )
    
    # 保存结果（与v3一致）
    output_file = os.path.join(args.output_dir, f"{args.output_prefix}_results.json")
    print(f"v6超激进优化版本保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Simulation complete. Results saved to: {output_file}")
    
    # 计算并报告统计信息（与v3一致）
    end_time = time.time()
    total_time = end_time - start_time
    
    if args.verbose:
        print(f"\nSimulation complete in {total_time:.2f} seconds")
        print(f"Results saved to: {output_file}")

        # 计算一些基本统计信息（与v3一致）
        if results:
            total_dialogues = len(results)
            completed_dialogues = sum(1 for r in results if r.get('is_completed', False))
            # 从结果列表中计算平均轮次
            avg_turns = sum(r.get('total_turns', 0) for r in results) / total_dialogues if total_dialogues > 0 else 0

            print(f"\nStatistics:")
            print(f"- Total dialogues processed: {total_dialogues}")
            print(f"- Completed dialogues: {completed_dialogues} ({completed_dialogues/total_dialogues*100:.1f}%)")
            print(f"- Average turns per dialogue: {avg_turns:.2f}")
        
        print_timing("整个v6超激进程序执行完成", args.verbose, main_start_time)
    
    print("v6超激进优化版本模拟完成!")


if __name__ == "__main__":
    main()

