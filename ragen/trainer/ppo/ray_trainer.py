# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import copy

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.trainer.ppo.core_algos import agg_loss

import re
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    EnvLLM = 7

class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def log_ppo_trajectory(batch: DataProto, global_step: int, batch_idx: int = 0, tokenizer=None):
    """打印完整的PPO更新轨迹（用于调试）- 包含可读的中文对话内容
    
    Args:
        batch: 包含完整轨迹数据的DataProto
        global_step: 当前全局步数
        batch_idx: 批次索引（如果想打印多个样本）
        tokenizer: 用于解码文本的tokenizer
    """
    # 检查是否启用PPO轨迹打印
    if not os.environ.get("RAGEN_LOG_PPO_TRAJECTORY", "0") == "1":
        return
    
    try:
        log_dir = os.environ.get("RAGEN_READABLE_LOG_DIR", "logs/readable")
        log_filename = os.environ.get("RAGEN_DEBUG_LOG_FILENAME", "med_dialogue_debug.log")
        debug_log_path = os.path.join(log_dir, log_filename)
        
        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[PPO_TRAJECTORY] Global Step {global_step}, Batch {batch_idx}\n")
            f.write("=" * 80 + "\n")
            
            # 获取批次大小
            batch_size = batch.batch.get("advantages", torch.tensor([])).shape[0] if "advantages" in batch.batch else 0
            
            if batch_size == 0:
                f.write("[WARNING] No advantages found in batch\n")
                return
            
            f.write(f"Batch Size: {batch_size}\n")
            f.write("-" * 80 + "\n")
            
            # 打印每个样本的关键信息
            num_samples_to_log = min(2, batch_size)  # 最多打印前2个样本
            
            for sample_idx in range(num_samples_to_log):
                f.write(f"\n{'#' * 80}\n")
                f.write(f"[Sample {sample_idx}]\n")
                f.write(f"{'#' * 80}\n\n")
                
                # ==================== 1. 解码完整轨迹（优化：先截断 token 再 decode，避免内存爆炸） ====================
                # 限制最大 decode 的 token 数量（前 1500 + 后 500），避免生成超大字符串
                MAX_HEAD_TOKENS = 1500
                MAX_TAIL_TOKENS = 500
                
                head_text = "N/A"
                tail_text = ""
                total_valid_tokens = 0
                is_truncated = False
                
                if tokenizer is not None:
                    try:
                        # PPO Update 阶段：优先使用 input_ids（包含完整轨迹）
                        # 如果没有 input_ids，再使用 prompts + responses
                        full_ids = None
                        data_source = "unknown"
                        
                        # 打印 batch 中可用的 keys（只打印一次）
                        if sample_idx == 0:
                            batch_keys = list(batch.batch.keys()) if hasattr(batch.batch, 'keys') else []
                            f.write(f"[DEBUG] Batch keys: {batch_keys}\n")
                            # 打印各个 tensor 的 shape
                            for key in ['input_ids', 'prompts', 'responses']:
                                if key in batch.batch:
                                    shape = batch.batch[key].shape if hasattr(batch.batch[key], 'shape') else 'N/A'
                                    f.write(f"[DEBUG] {key}.shape: {shape}\n")
                        
                        # 优先尝试 input_ids（包含完整的 prompts + responses）
                        if 'input_ids' in batch.batch:
                            input_ids = batch.batch['input_ids'][sample_idx]
                            if hasattr(input_ids, 'cpu'):
                                input_ids = input_ids.cpu()
                            full_ids = input_ids
                            data_source = "input_ids"
                        else:
                            # 回退到 prompts + responses（input_ids 不存在时）
                            prompt_ids = None
                            response_ids = None
                            
                            if 'prompts' in batch.batch:
                                prompt_ids = batch.batch['prompts'][sample_idx]
                                if hasattr(prompt_ids, 'cpu'):
                                    prompt_ids = prompt_ids.cpu()
                            
                            if 'responses' in batch.batch:
                                response_ids = batch.batch['responses'][sample_idx]
                                if hasattr(response_ids, 'cpu'):
                                    response_ids = response_ids.cpu()
                            
                            # 拼接 prompts + responses
                            if prompt_ids is not None and response_ids is not None:
                                full_ids = torch.cat([prompt_ids, response_ids], dim=0)
                                data_source = "prompts+responses"
                            elif prompt_ids is not None:
                                full_ids = prompt_ids
                                data_source = "prompts"
                            elif response_ids is not None:
                                full_ids = response_ids
                                data_source = "responses"
                        
                        if full_ids is not None:
                            # 打印数据来源和原始长度
                            if sample_idx == 0:
                                f.write(f"[DEBUG] Data source: {data_source}, raw length: {len(full_ids)}\n")
                            
                            # 过滤 pad token
                            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                                valid_ids = full_ids[full_ids != tokenizer.pad_token_id]
                            else:
                                valid_ids = full_ids
                            
                            total_valid_tokens = len(valid_ids)
                            if sample_idx == 0:
                                f.write(f"[DEBUG] After removing pad tokens: {total_valid_tokens} valid tokens\n")
                            
                            # 优化：先在 token 级别截断，再 decode（避免生成超大字符串）
                            if total_valid_tokens <= MAX_HEAD_TOKENS + MAX_TAIL_TOKENS:
                                # 不需要截断
                                head_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
                            else:
                                # 需要截断：只 decode 前 N 和后 M 个 token
                                is_truncated = True
                                head_ids = valid_ids[:MAX_HEAD_TOKENS]
                                tail_ids = valid_ids[-MAX_TAIL_TOKENS:]
                                head_text = tokenizer.decode(head_ids, skip_special_tokens=False)
                                tail_text = tokenizer.decode(tail_ids, skip_special_tokens=False)
                        
                    except Exception as e:
                        f.write(f"[WARNING] Failed to decode trajectory: {e}\n")
                
                # 打印轨迹内容
                f.write("=" * 80 + "\n")
                f.write("[完整对话轨迹] (PPO Update 使用的数据)\n")
                f.write("=" * 80 + "\n")
                if head_text != "N/A":
                    f.write(f"{head_text}\n")
                    if is_truncated:
                        f.write(f"\n... (轨迹太长，已在 token 级别截断，共 {total_valid_tokens} tokens) ...\n\n")
                        f.write(f"[轨迹末尾 (最后 {MAX_TAIL_TOKENS} tokens)]\n")
                        f.write(f"{tail_text}\n")
                    f.write("-" * 80 + "\n\n")
                else:
                    f.write("[WARNING] 无法解码轨迹\n")
                    f.write("-" * 80 + "\n\n")
                
                # ==================== 2. PPO 数值信息 ====================
                f.write("=" * 80 + "\n")
                f.write("[PPO 数值统计]\n")
                f.write("=" * 80 + "\n")
                
                # Response mask - 用于确定哪些是response tokens
                response_mask = None
                if "response_mask" in batch.batch:
                    mask = batch.batch["response_mask"][sample_idx]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    response_mask = mask
                    num_response_tokens = np.sum(mask)
                    f.write(f"Response Tokens: {int(num_response_tokens)} 个有效token\n")
                
                # Token-level rewards
                token_rewards = None
                if "token_level_rewards" in batch.batch:
                    rewards = batch.batch["token_level_rewards"][sample_idx]
                    if hasattr(rewards, 'cpu'):
                        rewards = rewards.cpu().numpy()
                    token_rewards = rewards
                    
                    # 只统计response部分的reward
                    if response_mask is not None:
                        valid_rewards = rewards[response_mask > 0]
                        if len(valid_rewards) > 0:
                            f.write(f"Token Rewards (response only): sum={np.sum(valid_rewards):.4f}, mean={np.mean(valid_rewards):.4f}, "
                                   f"min={np.min(valid_rewards):.4f}, max={np.max(valid_rewards):.4f}\n")
                        else:
                            f.write(f"Token Rewards (response only): NO VALID TOKENS (response_mask all zero)\n")
                    else:
                        f.write(f"Token Rewards (all): sum={np.sum(rewards):.4f}, mean={np.mean(rewards):.4f}\n")
                
                # Advantages
                token_advantages = None
                if "advantages" in batch.batch:
                    adv = batch.batch["advantages"][sample_idx]
                    if hasattr(adv, 'cpu'):
                        adv = adv.cpu().numpy()
                    token_advantages = adv
                    
                    if response_mask is not None:
                        valid_adv = adv[response_mask > 0]
                        if len(valid_adv) > 0:
                            f.write(f"Advantages (response only): mean={np.mean(valid_adv):.4f}, std={np.std(valid_adv):.4f}, "
                                   f"min={np.min(valid_adv):.4f}, max={np.max(valid_adv):.4f}\n")
                        else:
                            f.write(f"Advantages (response only): NO VALID TOKENS\n")
                    else:
                        f.write(f"Advantages (all): mean={np.mean(adv):.4f}, std={np.std(adv):.4f}\n")
                
                # Returns
                if "returns" in batch.batch:
                    ret = batch.batch["returns"][sample_idx]
                    if hasattr(ret, 'cpu'):
                        ret = ret.cpu().numpy()
                    
                    if response_mask is not None:
                        valid_ret = ret[response_mask > 0]
                        if len(valid_ret) > 0:
                            f.write(f"Returns (response only): mean={np.mean(valid_ret):.4f}, std={np.std(valid_ret):.4f}\n")
                        else:
                            f.write(f"Returns (response only): NO VALID TOKENS\n")
                    else:
                        f.write(f"Returns (all): mean={np.mean(ret):.4f}, std={np.std(ret):.4f}\n")
                
                # Values (如果有critic)
                if "values" in batch.batch:
                    vals = batch.batch["values"][sample_idx]
                    if hasattr(vals, 'cpu'):
                        vals = vals.cpu().numpy()
                    
                    if response_mask is not None:
                        valid_vals = vals[response_mask > 0]
                        if len(valid_vals) > 0:
                            f.write(f"Values (response only): mean={np.mean(valid_vals):.4f}, std={np.std(valid_vals):.4f}\n")
                        else:
                            f.write(f"Values (response only): NO VALID TOKENS\n")
                
                f.write("\n")
            
            # 统计信息
            f.write("\n" + "-" * 80 + "\n")
            f.write("[Batch Statistics]\n")
            
            if "advantages" in batch.batch:
                all_adv = batch.batch["advantages"]
                if hasattr(all_adv, 'cpu'):
                    all_adv = all_adv.cpu().numpy()
                f.write(f"All Advantages: mean={np.mean(all_adv):.4f}, std={np.std(all_adv):.4f}, "
                       f"min={np.min(all_adv):.4f}, max={np.max(all_adv):.4f}\n")
            
            if "returns" in batch.batch:
                all_ret = batch.batch["returns"]
                if hasattr(all_ret, 'cpu'):
                    all_ret = all_ret.cpu().numpy()
                f.write(f"All Returns: mean={np.mean(all_ret):.4f}, std={np.std(all_ret):.4f}, "
                       f"min={np.min(all_ret):.4f}, max={np.max(all_ret):.4f}\n")
            
            if "token_level_rewards" in batch.batch:
                all_rew = batch.batch["token_level_rewards"]
                if hasattr(all_rew, 'cpu'):
                    all_rew = all_rew.cpu().numpy()
                f.write(f"All Token Rewards: sum={np.sum(all_rew):.4f}, mean={np.mean(all_rew):.4f}, "
                       f"min={np.min(all_rew):.4f}, max={np.max(all_rew):.4f}\n")
            
            f.write("=" * 80 + "\n\n")
            
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to log PPO trajectory: {e}")
        traceback.print_exc()


def _check_for_nan(tensor, name):
    """检查张量是否包含 nan/inf 并打印警告"""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        finite_mask = torch.isfinite(tensor)
        finite_vals = tensor[finite_mask] if finite_mask.any() else torch.tensor([0.0])
        print(f"\n[NaN in compute_advantage] {name}:")
        print(f"  Shape: {tensor.shape}, NaN count: {torch.isnan(tensor).sum().item()}, Inf count: {torch.isinf(tensor).sum().item()}")
        if finite_vals.numel() > 0:
            print(f"  Finite range: [{finite_vals.min().item():.4f}, {finite_vals.max().item():.4f}]")
        return True
    return False


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    # [NaN Check] 检查 advantage 计算的输入
    _check_for_nan(data.batch.get("token_level_rewards"), "token_level_rewards (input)")
    _check_for_nan(data.batch.get("response_mask"), "response_mask (input)")

    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        # [NaN Check] 检查 GRPO advantage 输出
        if _check_for_nan(advantages, "advantages (GRPO output)"):
            print(f"  [GRPO Debug] rewards range: [{data.batch['token_level_rewards'].min().item():.4f}, {data.batch['token_level_rewards'].max().item():.4f}]")
        if _check_for_nan(returns, "returns (GRPO output)"):
            pass

        # [诊断] 打印 advantage 统计，帮助判断是否正常
        adv_masked = advantages * data.batch["response_mask"]
        mask_sum = data.batch["response_mask"].sum().item()
        if mask_sum > 0:
            adv_mean = adv_masked.sum().item() / mask_sum
            adv_std = ((adv_masked ** 2).sum().item() / mask_sum - adv_mean ** 2) ** 0.5 if mask_sum > 1 else 0
            adv_max = advantages[data.batch["response_mask"] > 0].max().item() if (data.batch["response_mask"] > 0).any() else 0
            adv_min = advantages[data.batch["response_mask"] > 0].min().item() if (data.batch["response_mask"] > 0).any() else 0
            # 只在 std 异常时打印
            if adv_std < 1e-6 or adv_std > 100:
                print(f"[ADV_STATS] GRPO advantage: mean={adv_mean:.4f}, std={adv_std:.4f}, min={adv_min:.4f}, max={adv_max:.4f}")

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

def normalize_reward(reward, uid, reward_norm_type):
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    for r, u in zip(reward, uid):
        id2score[u].append(r)
    for u in id2score:
        if len(id2score[u]) == 1:
            id2mean[u] = torch.tensor(0.0)
            id2std[u] = torch.tensor(1.0)
        elif len(id2score[u]) > 1:
            id2mean[u] = torch.mean(torch.tensor(id2score[u], dtype=torch.float32))
            id2std[u] = torch.std(torch.tensor([id2score[u]], dtype=torch.float32))
        else:
            raise ValueError(f"no score in prompt index: {u}")
    normalized_reward = [(r - id2mean[u]) / (id2std[u] + 1e-6) for r, u in zip(reward, uid)] # NOTE: +1e-6, maybe +1!
    # transform to the same dtype as reward
    return np.array(normalized_reward, dtype=reward.dtype)

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    if "response_mask" not in batch.batch:
        response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    else:
        response_mask = batch.batch['response_mask'].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
            
        # metrics for actions
        'metric/total_env':
            int(np.array(batch.non_tensor_batch['total_env'], dtype=np.int16).sum()),
        'metric/finished_env':
            int(np.array(batch.non_tensor_batch['finished_env'], dtype=np.int16).sum()),
        'metric/success_env':
            int(np.array(batch.non_tensor_batch['success_env'], dtype=np.int16).sum()),
        'metric/traj_length':
            float(np.array(batch.non_tensor_batch['traj_length'], dtype=np.int16).mean()),
        'metric/valid_action':
            float(np.array(batch.non_tensor_batch['valid_action'], dtype=np.int16).mean()),
        'metric/effective_action':
            float(np.array(batch.non_tensor_batch['effective_action'], dtype=np.int16).mean()),
        'metric/effective_action_ratio':
            float(np.array(batch.non_tensor_batch['effective_action_ratio'], dtype=np.float32).mean()),
        'metric/reward':
            float(np.array(batch.non_tensor_batch['ori_reward'], dtype=np.float32).mean()),
    }

    if 'diagnosis_score' in batch.non_tensor_batch:
        metrics['metric/diagnosis_score'] = float(np.array(batch.non_tensor_batch['diagnosis_score'], dtype=np.float32).mean())
        metrics['metric/recommandation_score'] = float(np.array(batch.non_tensor_batch['recommandation_score'], dtype=np.float32).mean())

    # metric for two-armed bandit
    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
        batch_action = np.array(batch.non_tensor_batch['bandit_metrics'], dtype=np.int16)
        metrics['metric/n_low_arm'] = int(np.sum(batch_action == 1))
        metrics['metric/n_high_arm'] = int(np.sum(batch_action == 2))
        metrics['metric/n_invalid'] = int(np.sum(batch_action == 0))

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 env=None,
                 val_env=None,
                 env_class=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env = env
        self.val_env = env if val_env is None else val_env
        self.env_class = env_class
        
        if val_env is not None:
            print("[INFO] val env is different from train env, it means you are evaluating the model's generalization capabilities.")

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping and not config.algorithm.no_ref_policy
        print(f"use_reference_policy: {self.use_reference_policy}")
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.val_num = 0
        
        # ========== Best checkpoint tracking ==========
        # 只保存一个准确率最高的 checkpoint（不需要稳定性检查）
        self.best_metric_value = float('-inf')
        self.best_metric_name = config.trainer.get('best_metric_name', 'validate_metric/diagnosis_accuracy_overall')
        self.best_checkpoint_step = None
        
        # ========== stable_resume_checkpoint 相关变量 ==========
        # 用于保存"稳定条件满足时验证准确率最高"的 checkpoint
        # 需要满足严格稳定性条件：grad_norm finite、update_skipped=0、KL/clip 正常
        self.stable_resume_best_value = float('-inf')
        self.stable_resume_best_step = None
        self.stable_resume_metric_name = config.trainer.get('stable_resume_metric_name', 'validate_metric/diagnosis_accuracy_overall')
        
        # 稳定性评分配置（可通过 config 覆盖）
        # 注意：min_finish_rate 设为 0 表示不检查完成率，只依赖 KL 和 clip_ratio 来判断稳定性
        self.stability_config = {
            'min_finish_rate': config.trainer.get('best_ckpt_min_finish_rate', 0.0),   # 不检查完成率
            'max_ppo_kl': config.trainer.get('best_ckpt_max_ppo_kl', 0.15),             # KL <= 0.15
            'max_clip_ratio': config.trainer.get('best_ckpt_max_clip_ratio', 0.02),    # clip_ratio <= 0.02
        }

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        self._validate_config()
        self._create_dataloader()
        self._init_logger()
    
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from ragen.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        # use sampler for better ckpt resume
        # 支持均衡类别采样（use_balanced_sampling=True 时启用）
        use_balanced_sampling = self.config.data.get("use_balanced_sampling", False)
        
        if use_balanced_sampling:
            # 从 dataframe 中提取类别标签
            from torch.utils.data import Sampler
            import random
            df = self.train_dataset.dataframe
            
            # 尝试从 reward_model.ground_truth.diagnosis 获取类别
            def get_diagnosis(row):
                try:
                    reward_model = row.get('reward_model', {})
                    if isinstance(reward_model, dict):
                        ground_truth = reward_model.get('ground_truth', {})
                        if isinstance(ground_truth, dict):
                            return ground_truth.get('diagnosis', 'Unknown')
                    return 'Unknown'
                except:
                    return 'Unknown'
            
            categories = df.apply(get_diagnosis, axis=1).tolist()
            
            # 统计各类别数量
            from collections import Counter, defaultdict
            category_counts = Counter(categories)
            print(f"[Stratified Sampling] Category distribution: {dict(category_counts)}")
            
            # 构建每个类别的索引列表
            category_indices = defaultdict(list)
            for idx, cat in enumerate(categories):
                category_indices[cat].append(idx)

            # 获取 batch size
            batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
            num_categories = len(category_counts)

            # 计算每个类别在每个 batch 中应该有多少样本
            samples_per_category = batch_size // num_categories
            remainder = batch_size % num_categories

            print(f"[Stratified Sampling] batch_size={batch_size}, num_categories={num_categories}")
            print(f"[Stratified Sampling] samples_per_category={samples_per_category}, remainder={remainder}")

            if samples_per_category == 0:
                print(f"[WARNING] batch_size ({batch_size}) < num_categories ({num_categories}), falling back to WeightedRandomSampler")
                # Fallback to weighted sampling if batch too small
                from torch.utils.data import WeightedRandomSampler
                total_samples = len(categories)
                weights = []
                for cat in categories:
                    cat_count = category_counts[cat]
                    weight = total_samples / (num_categories * cat_count)
                    weights.append(weight)
                weights_tensor = torch.DoubleTensor(weights)
                train_dataloader_generator = torch.Generator()
                train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
                sampler = WeightedRandomSampler(
                    weights=weights_tensor,
                    num_samples=len(weights),
                    replacement=True,
                    generator=train_dataloader_generator
                )
            else:
                # 实现严格的 Stratified Batch Sampler
                class StratifiedBatchSampler(Sampler):
                    def __init__(self, category_indices, samples_per_category, batch_size, seed=1):
                        self.category_indices = category_indices
                        self.samples_per_category = samples_per_category
                        self.batch_size = batch_size
                        self.seed = seed

                    def __iter__(self):
                        return iter([])

                    def __len__(self):
                        return 0

                # 使用 StratifiedBatchSampler
                sampler = StratifiedBatchSampler(category_indices, samples_per_category, batch_size)

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        # 【修复】当 val_batch_size 为 None 时，使用验证数据集的实际大小（一次性处理所有数据）
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
            print(f"[VAL] val_batch_size is None, using dataset size: {val_batch_size}")

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,  # 【修复】验证集不丢弃最后的不完整 batch，确保所有数据都被使用
            collate_fn=collate_fn,
        )

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # create env_llm worker
        if self.config.env.use_env_llm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.EnvLLM)
            env_llm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.EnvLLM], 
                                            config=self.config.env.env_llm,
                                            role='env_llm')
            self.resource_pool_to_cls[resource_pool]['env_llm'] = env_llm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # initialize env_llm worker
        if self.config.env.use_env_llm:
            self.env_llm_wg = all_wg['env_llm']
            self.env_llm_wg.init_model()
            self.env.env_llm_worker = self.env_llm_wg
            self.val_env.env_llm_worker = self.env_llm_wg

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _check_stability(self, metrics: dict) -> tuple[bool, str]:
        """
        检查当前 step 是否满足稳定性条件。
        
        稳定性条件（三项都需满足）：
        1. 完成率 F = finished_env / total_env >= 0.90
        2. KL散度 actor/ppo_kl <= 0.15
        3. 响应截断率 response_length/clip_ratio <= 0.02
        
        Returns:
            (is_stable, reason): 是否稳定，以及不稳定的原因
        """
        # 获取 metrics
        total_env = metrics.get('metric/total_env', 1)
        finished_env = metrics.get('metric/finished_env', 0)
        ppo_kl = metrics.get('actor/ppo_kl', 0)
        clip_ratio = metrics.get('response_length/clip_ratio', 0)
        
        # 计算完成率
        finish_rate = finished_env / max(total_env, 1)
        
        # 检查稳定性条件
        reasons = []
        if finish_rate < self.stability_config['min_finish_rate']:
            reasons.append(f"F={finish_rate:.3f} < {self.stability_config['min_finish_rate']}")
        if ppo_kl > self.stability_config['max_ppo_kl']:
            reasons.append(f"KL={ppo_kl:.4f} > {self.stability_config['max_ppo_kl']}")
        if clip_ratio > self.stability_config['max_clip_ratio']:
            reasons.append(f"clip={clip_ratio:.4f} > {self.stability_config['max_clip_ratio']}")
        
        is_stable = len(reasons) == 0
        reason_str = "; ".join(reasons) if reasons else "stable"
        
        return is_stable, reason_str

    def _compute_best_checkpoint_score(self, diag_metrics: dict, metrics: dict) -> float:
        """
        计算 best checkpoint 的综合评分（越大越好）。
        
        评分公式：
        1. 几何平均（防偏科）: G = (ad * aa * am)^(1/3)
        2. 完成率: F = finished_env / total_env
        3. 最终评分: Score = 0.7*A + 0.3*G - 0.2*(1-F)
        
        其中:
        - A = train/diagnosis_accuracy_cumulative (总体准确率)
        - ad = train/diagnosis_accuracy_depression_cumulative
        - aa = train/diagnosis_accuracy_anxiety_cumulative
        - am = train/diagnosis_accuracy_mix_cumulative
        
        Returns:
            综合评分 (float)
        """
        # 总体准确率 A
        A = diag_metrics.get('train/diagnosis_accuracy_cumulative', 0)
        
        # 各分类准确率
        ad = diag_metrics.get('train/diagnosis_accuracy_depression_cumulative', 0)
        aa = diag_metrics.get('train/diagnosis_accuracy_anxiety_cumulative', 0)
        am = diag_metrics.get('train/diagnosis_accuracy_mix_cumulative', 0)
        
        # 完成率 F
        total_env = metrics.get('metric/total_env', 1)
        finished_env = metrics.get('metric/finished_env', 0)
        F = finished_env / max(total_env, 1)
        
        # 几何平均（防止某一类别为0导致整体为0，加一个小的 epsilon）
        eps = 1e-6
        G = (max(ad, eps) * max(aa, eps) * max(am, eps)) ** (1/3)
        
        # 最终评分
        score = 0.7 * A + 0.3 * G - 0.2 * (1 - F)
        
        return score

    def _save_best_checkpoint_v2(self, diag_metrics: dict, metrics: dict):
        """
        简化版 best checkpoint 保存逻辑：只保存一个准确率最高的 checkpoint。
        【不需要稳定性检查】：只看 validate_metric/diagnosis_accuracy_overall 是否创新高。
        
        stable_resume_checkpoint 才需要稳定性检查。
        
        Args:
            diag_metrics: 诊断相关 metrics（来自 env_class.get_diagnosis_metrics()）
            metrics: 其他训练 metrics（包含验证指标）
        """
        # 从 metrics 中获取验证集诊断准确率（优先使用 overall）
        metric_value = metrics.get('validate_metric/diagnosis_accuracy_overall', None)
        metric_name = 'validate_metric/diagnosis_accuracy_overall'
        
        # 如果 overall 不存在，回退到 diagnosis_accuracy
        if metric_value is None:
            metric_value = metrics.get('validate_metric/diagnosis_accuracy', None)
            metric_name = 'validate_metric/diagnosis_accuracy'
        
        # 如果没有验证指标，跳过保存
        if metric_value is None:
            print(f"[Best Checkpoint] Step {self.global_steps}: No validation metrics available, skipping.")
            return
        
        metric_value = float(metric_value)
        
        # 获取其他辅助信息用于日志
        val_finished = metrics.get('validate_metric/finished_total', 0)
        val_total = metrics.get('validate_metric/total_env', 0)
        completion_rate = val_finished / max(val_total, 1)
        ppo_kl = metrics.get('actor/ppo_kl', 0)
        
        print(f"[Best Checkpoint] Step {self.global_steps}: {metric_name}={metric_value:.4f}, "
              f"completion={completion_rate:.2%}, KL={ppo_kl:.4f}")
        
        # 检查是否超过之前的最佳值
        if metric_value <= self.best_metric_value:
            print(f"[Best Checkpoint] Step {self.global_steps}: {metric_value:.4f} <= "
                  f"best={self.best_metric_value:.4f}, skipping.")
            return
        
        # 更新最佳记录
        old_best = self.best_metric_value
        self.best_metric_value = metric_value
        self.best_checkpoint_step = self.global_steps
        
        print(f"[Best Checkpoint] NEW BEST! {old_best:.4f} -> {metric_value:.4f}")
        
        # 保存到 best_checkpoint 目录
        self._save_single_best_checkpoint(
            folder_name="best_checkpoint",
            metric_name=metric_name,
            metric_value=metric_value,
            extra_info={
                'completion_rate': completion_rate,
                'ppo_kl': ppo_kl,
            }
        )
    
    def _save_single_best_checkpoint(self, folder_name: str, metric_name: str, metric_value: float, extra_info: dict = None):
        """
        保存 best/stable checkpoint 到指定目录。
        【关键特性】直接复制完整的 global_step_XXX 目录，而不是调用 save_checkpoint。
        这样 checkpoint 是独立的完整副本，不会被 max_ckpt_to_keep=2 清理掉。
        
        用于：
        - best_checkpoint: 准确率最高的 checkpoint（不需要稳定性）
        - stable_resume_checkpoint: 稳定 + 准确率最高的 checkpoint
        
        Args:
            folder_name: 保存目录名（如 'best_checkpoint', 'stable_resume_checkpoint'）
            metric_name: 指标名称
            metric_value: 指标值
            extra_info: 额外信息字典（写入 best_info.txt）
        """
        import shutil
        
        best_folder = os.path.join(self.config.trainer.default_local_dir, folder_name)
        source_ckpt_path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        
        print(f"[Best Checkpoint] Saving {folder_name} at step {self.global_steps}")
        print(f"[Best Checkpoint] Source: {source_ckpt_path}")
        
        # 检查源 checkpoint 是否存在
        if os.path.exists(source_ckpt_path):
            # 确保目标目录存在
            os.makedirs(best_folder, exist_ok=True)
            
            # 清理目标目录中的旧内容（避免残留文件）
            for item in os.listdir(best_folder):
                item_path = os.path.join(best_folder, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            # 复制完整的 global_step 目录内容到 best_folder
            shutil.copytree(source_ckpt_path, best_folder, dirs_exist_ok=True)
            print(f"[Best Checkpoint] Copied full checkpoint from {source_ckpt_path} to {best_folder}")
        else:
            # 如果源目录不存在（可能是非 save_freq 步骤），回退到原始方法
            print(f"[Best Checkpoint] WARNING: Source {source_ckpt_path} not found, falling back to save_checkpoint")
        actor_local_path = os.path.join(best_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, folder_name, "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None
        )
        
        if self.use_critic:
            critic_local_path = os.path.join(best_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(self.config.trainer.default_hdfs_dir, folder_name, "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None
            )
        
        # 保存 dataloader 状态
        dataloader_local_path = os.path.join(best_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        
        # 保存 best checkpoint 元信息（无论哪种方式都需要写入）
        best_info_path = os.path.join(best_folder, "best_info.txt")
        with open(best_info_path, "w") as f:
            f.write(f"global_step: {self.global_steps}\n")
            f.write(f"metric_name: {metric_name}\n")
            f.write(f"metric_value: {metric_value:.6f}\n")
            if extra_info:
                for k, v in extra_info.items():
                    if isinstance(v, float):
                        f.write(f"{k}: {v:.6f}\n")
                    else:
                        f.write(f"{k}: {v}\n")
        
        print(f"[Best Checkpoint] Saved to: {best_folder}")

    def _check_stable_resume(self, metrics: dict) -> bool:
        """
        检查当前 step 是否满足"稳定可续训"的条件。
        比 _check_stability 更严格，额外要求：
        1. grad_norm 必须是 finite（不是 NaN/Inf）
        2. update_skipped == 0（本 step 没有因为 NaN 梯度而跳过更新）
        
        Returns:
            bool: 是否满足稳定条件
        """
        import math
        
        # 先检查基础稳定性（注意 _check_stability 返回 (bool, reason) 元组）
        is_stable, reason = self._check_stability(metrics)
        if not is_stable:
            print(f"[Stable Resume] Step {self.global_steps}: 基础稳定性不满足: {reason}", flush=True)
            return False
        
        # 额外检查：grad_norm 必须是 finite
        grad_norm = metrics.get('actor/grad_norm', None)
        if grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)):
            print(f"[Stable Resume] Step {self.global_steps}: grad_norm={grad_norm} 不是 finite，不满足稳定条件", flush=True)
            return False
        
        # 额外检查：update_skipped 必须为 0
        update_skipped = metrics.get('actor/update_skipped', 0)
        if update_skipped > 0:
            print(f"[Stable Resume] Step {self.global_steps}: update_skipped={update_skipped} > 0，不满足稳定条件", flush=True)
            return False
        
        return True
    
    def _save_stable_resume_checkpoint(self, metrics: dict):
        """
        保存"稳定可续训"的 checkpoint。
        只有当满足稳定条件且验证准确率超过之前的最佳值时才保存。
        保存位置固定为 default_local_dir/stable_resume_checkpoint（始终覆盖，永远只有一个）。
        """
        import shutil
        
        # 检查是否满足稳定条件
        if not self._check_stable_resume(metrics):
            return
        
        # 获取验证准确率指标
        metric_name = self.stable_resume_metric_name
        metric_value = metrics.get(metric_name, None)
        
        # 如果主指标不存在，尝试回退到 diagnosis_accuracy
        if metric_value is None:
            fallback_name = 'validate_metric/diagnosis_accuracy'
            metric_value = metrics.get(fallback_name, None)
            if metric_value is not None:
                metric_name = fallback_name
        
        if metric_value is None:
            print(f"[Stable Resume] Step {self.global_steps}: 没有找到指标 {self.stable_resume_metric_name}，跳过保存", flush=True)
            return
        
        metric_value = float(metric_value)
        
        # 比较是否超过之前的最佳值
        if metric_value <= self.stable_resume_best_value:
            print(f"[Stable Resume] Step {self.global_steps}: {metric_name}={metric_value:.4f} <= "
                  f"best={self.stable_resume_best_value:.4f}，跳过保存", flush=True)
            return
        
        # 更新最佳记录
        old_best = self.stable_resume_best_value
        self.stable_resume_best_value = metric_value
        self.stable_resume_best_step = self.global_steps
        
        print(f"[Stable Resume] 新的最佳稳定 checkpoint! {metric_name}: {old_best:.4f} -> {metric_value:.4f}", flush=True)
        print(f"[Stable Resume] 保存 stable_resume_checkpoint at step {self.global_steps}", flush=True)
        
        # 保存位置
        stable_folder = os.path.join(self.config.trainer.default_local_dir, "stable_resume_checkpoint")
        source_ckpt_path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        
        if os.path.exists(source_ckpt_path):
            # 确保目标目录存在
            os.makedirs(stable_folder, exist_ok=True)
            
            # 清理目标目录中的旧内容
            for item in os.listdir(stable_folder):
                item_path = os.path.join(stable_folder, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            # 复制完整的 global_step 目录内容
            shutil.copytree(source_ckpt_path, stable_folder, dirs_exist_ok=True)
            print(f"[Stable Resume] 已复制完整 checkpoint 从 {source_ckpt_path} 到 {stable_folder}", flush=True)
        else:
            print(f"[Stable Resume] WARNING: 源目录 {source_ckpt_path} 不存在（可能非 save_freq 步骤），跳过保存", flush=True)
            return
        
        # 保存 stable checkpoint 元信息
        info_path = os.path.join(stable_folder, "stable_info.txt")
        with open(info_path, "w") as f:
            f.write(f"global_step: {self.global_steps}\n")
            f.write(f"metric_name: {metric_name}\n")
            f.write(f"metric_value: {metric_value:.6f}\n")
            f.write(f"is_stable: True\n")
            # 记录稳定性相关指标
            for key in ['actor/grad_norm', 'actor/update_skipped', 'actor/ppo_kl', 'actor/pg_clipfrac']:
                if key in metrics:
                    f.write(f"{key}: {metrics[key]}\n")
        
        print(f"[Stable Resume] 保存完成: {stable_folder}", flush=True)

    def _save_best_checkpoint(self, current_metric_value: float):
        """
        保存 best checkpoint，只保留唯一一个最佳模型。
        Args:
            current_metric_value: 当前验证指标值
        """
        if current_metric_value <= self.best_metric_value:
            print(f"[Best Checkpoint] Current {self.best_metric_name}={current_metric_value:.4f} <= "
                  f"Best={self.best_metric_value:.4f}, skipping save.")
            return
        
        # 更新最佳记录
        old_best = self.best_metric_value
        self.best_metric_value = current_metric_value
        self.best_checkpoint_step = self.global_steps
        
        print(f"[Best Checkpoint] New best! {self.best_metric_name}: {old_best:.4f} -> {current_metric_value:.4f}")
        print(f"[Best Checkpoint] Saving best checkpoint at step {self.global_steps}")
        
        # 保存到 best_checkpoint 目录（始终覆盖）
        best_folder = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint")
        
        actor_local_path = os.path.join(best_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "best_checkpoint", "actor")
        )
        
        # Best checkpoint 保存到固定路径，文件会自动覆盖，不需要 max_ckpt_to_keep 清理逻辑
        # 使用 max_ckpt_to_keep=None 避免误删正常 checkpoint（它们共享同一个 previous_saved_paths 列表）
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None
        )
        
        if self.use_critic:
            critic_local_path = os.path.join(best_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, "best_checkpoint", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None
            )
        
        # 保存 dataloader 状态
        dataloader_local_path = os.path.join(best_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        
        # 保存 best checkpoint 元信息
        best_info_path = os.path.join(best_folder, "best_info.txt")
        with open(best_info_path, "w") as f:
            f.write(f"global_step: {self.global_steps}\n")
            f.write(f"metric_name: {self.best_metric_name}\n")
            f.write(f"metric_value: {current_metric_value}\n")
        
        print(f"[Best Checkpoint] Saved to: {best_folder}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # 【强制检查】如果指定了 resume_from_path，必须从那里加载，否则报错
        resume_from_path = self.config.trainer.get('resume_from_path', None)
        if resume_from_path is not None and isinstance(resume_from_path, str) and resume_from_path.strip():
            # 用户明确指定了续训路径，必须使用它
            global_step_folder = resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)

            # 检查路径是否存在
            if not os.path.exists(global_step_folder):
                raise FileNotFoundError(
                    f"[FATAL] 续训路径不存在: {global_step_folder}\n"
                    f"请检查 resume_from_path 配置是否正确！"
                )

            actor_path = os.path.join(global_step_folder, "actor")
            if not os.path.exists(actor_path):
                raise FileNotFoundError(
                    f"[FATAL] 续训路径下没有 actor 目录: {actor_path}\n"
                    f"请确认 checkpoint 完整性！"
                )

            print(f"[RESUME] 强制使用指定的续训路径: {global_step_folder}")
        else:
            # 没有指定 resume_from_path，使用 default_local_dir 自动找最新 ckpt
            if self.config.trainer.default_hdfs_dir is not None:
                raise NotImplementedError("load from hdfs is not implemented yet")

            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0

        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        # 优先使用配置中的 resume_step，否则从路径解析
        resume_step_from_config = self.config.trainer.get('resume_step', None)
        if resume_step_from_config is not None:
            self.global_steps = int(resume_step_from_config)
            print(f"[RESUME] Using resume_step from config: {self.global_steps}")
        elif "global_step_" in global_step_folder:
            # 从路径提取数字，支持 global_step_600_val_70.0_backup_xxx 格式
            import re
            step_part = global_step_folder.split("global_step_")[-1]
            match = re.match(r'^(\d+)', step_part)
            if match:
                self.global_steps = int(match.group(1))
                print(f"[RESUME] Parsed global_step from path: {self.global_steps}")
            else:
                raise ValueError(f"[FATAL] 无法从路径解析 step 数字: {global_step_folder}")
        else:
            # 尝试从配置中读取 resume_step
            resume_step = self.config.trainer.get('resume_step', None)
            if resume_step is not None:
                self.global_steps = int(resume_step)
                print(f"[RESUME] Using resume_step from config: {self.global_steps}")
            else:
                # 尝试从 best_info.txt 读取
                best_info_path = os.path.join(global_step_folder, "best_info.txt")
                if os.path.exists(best_info_path):
                    with open(best_info_path, "r") as f:
                        for line in f:
                            if line.startswith("global_step:"):
                                self.global_steps = int(line.split(":")[1].strip())
                                print(f"[RESUME] Read global_step from best_info.txt: {self.global_steps}")
                                break
                        else:
                            raise ValueError(
                                f"[FATAL] 无法确定 resume step！\n"
                                f"路径 {global_step_folder} 不包含 'global_step_'，\n"
                                f"且未设置 trainer.resume_step，\n"
                                f"且 best_info.txt 中未找到 global_step。"
                            )
                else:
                    raise ValueError(
                        f"[FATAL] 无法确定 resume step！\n"
                        f"路径 {global_step_folder} 不包含 'global_step_'，\n"
                        f"且未设置 trainer.resume_step，\n"
                        f"且不存在 best_info.txt。"
                    )

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # 同上：dataloader 状态同样以 weights_only=True 加载，避免任意代码执行。
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=True)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix) 
        metrics.update(global_balance_stats)
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1


        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger = logger,
        )

        envs = [self.env.copy() for _ in range(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n_agent)] 



        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                # update ref_policy_wg (only if using reference policy)
                if self.use_reference_policy and self.config.trainer.ref_update_steps is not None and self.global_steps % self.config.trainer.ref_update_steps == 0:
                    self.actor_rollout_wg.save_checkpoint(
                        local_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        hdfs_path=None
                    )
                    self.ref_policy_wg.load_model_parameters(
                        source_model_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        strict=True
                    )
                    print(f"load parameters from ./log/temp/actor_rollout_wg_global_step_{self.global_steps} to ref_policy_wg")

                metrics = {}
                timing_raw = {}
                
                # 重置当前 step 的诊断统计
                try:
                    if hasattr(self.env_class, 'reset_step_diagnosis_stats'):
                        self.env_class.reset_step_diagnosis_stats()
                except Exception as e:
                    print(f"[WARNING] Failed to reset diagnosis stats: {e}")

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                env_seeds = [int(i['index']) if i.get('index') is not None else 0 for i in batch.non_tensor_batch['extra_info']]
                print("env_seeds:", env_seeds)
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)


                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                # with _timer('gen', timing_raw):
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                              dtype=object)
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)

                #     # output batch to file
                #     self._record_batch(batch, path=f'.log/{self.config.trainer.experiment_name}/gen_batch.txt')

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################

                with _timer('step', timing_raw):
                    """
                    keep rolling to generate K turns of responses.
                    when doing this, update the "original right side" when new responses are generated.
                    finally, concatenate the "original left side" and "original right side" to get the final thing to feed to train the model.

                    Left-pad prompts, right-gen flow, Tensors dance like stardust glow.
                    Errors swarm? Stay calm, don't fret- Code with coffee, debug sunset.
                    """

                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    output_dir = (f"{self.config.logging.log_image_dir}/"
                                 f"{self.config.trainer.experiment_name}/"
                                 f"train/"
                                 f"step_{self.global_steps}")

                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            envs=envs,
                            initial_input_ids=first_input_ids,
                            output_dir=output_dir,
                            global_steps=self.global_steps,
                        )

                    # with torch.no_grad():
                    #     output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    #     final_gen_batch_output = final_gen_batch_output.union(output)
                    
                    if self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.reward_norm_type == 'grpo': # NOTE we currently use seed to group, better use prompt (hash) to group
                        batch.non_tensor_batch['uid'] = np.array([str(i) for i in env_seeds], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'brpo' or self.config.algorithm.reward_norm_type == 'brpo':
                        batch.non_tensor_batch['uid'] = np.array(["" for _ in range(len(batch.batch))], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'arpo' or self.config.algorithm.reward_norm_type == 'arpo':
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object) # No Relative normalization

                    # reward
                    batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['reward'][idx] = env.reward
                    batch.non_tensor_batch['ori_reward'] = copy.deepcopy(batch.non_tensor_batch['reward'])

                    # 判断diagnosis_score是不是envs[0]的一个属性
                    if hasattr(envs[0], 'diagnosis_score'):
                        batch.non_tensor_batch['diagnosis_score'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        batch.non_tensor_batch['recommandation_score'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['diagnosis_score'][idx] = env.diagnosis_score
                            batch.non_tensor_batch['recommandation_score'][idx] = env.recommandation_score

                    # normalize reward
                    if self.config.algorithm.reward_norm_type is not None:
                        batch.non_tensor_batch['reward'] = normalize_reward(batch.non_tensor_batch['reward'], batch.non_tensor_batch['uid'], self.config.algorithm.reward_norm_type)

                    # ========== [新增] 收集 reward_list 并应用 GRPO 归一化 ==========
                    # reward_list 是每轮的奖励列表，用于多轮奖励分配
                    batch.non_tensor_batch['reward_list'] = [getattr(env, '_reward_list', [env.reward]) for env in envs]

                    # 对 reward_list 中的每个元素应用 GRPO 归一化（使用相同的 mean/std）
                    if self.config.algorithm.reward_norm_type is not None:
                        try:
                            # 计算所有 ori_reward 的 mean 和 std（按 uid 分组）
                            ori_rewards = batch.non_tensor_batch['ori_reward']
                            uids = batch.non_tensor_batch['uid']

                            # 按 uid 分组计算统计量
                            uid_to_rewards = {}
                            for uid, r in zip(uids, ori_rewards):
                                if uid not in uid_to_rewards:
                                    uid_to_rewards[uid] = []
                                uid_to_rewards[uid].append(float(r))

                            uid_to_stats = {}
                            # 【优化】给 std 加下限，避免组内方差太小导致归一化后饱和到极端值
                            std_floor = 0.5  # 下限，防止 std 太小放大噪声
                            for uid, rs in uid_to_rewards.items():
                                mean_r = np.mean(rs)
                                std_r = max(np.std(rs), std_floor) + 1e-8
                                uid_to_stats[uid] = (mean_r, std_r)

                            # 对 reward_list 中的每个值应用相同的归一化
                            # 【优化】缩小 clip 范围，减少极端梯度（5.0 -> 2.0）
                            clip_v = 2.0
                            normalized_reward_lists = []
                            for idx, (uid, rl) in enumerate(zip(uids, batch.non_tensor_batch['reward_list'])):
                                mean_r, std_r = uid_to_stats.get(uid, (0, 1))
                                normalized_rl = []
                                for r in rl:
                                    norm_r = (float(r) - mean_r) / std_r
                                    # clip 归一化后的奖励（缩小范围）
                                    norm_r = max(min(norm_r, clip_v), -clip_v)
                                    normalized_rl.append(norm_r)
                                normalized_reward_lists.append(normalized_rl)
                            batch.non_tensor_batch['reward_list'] = normalized_reward_lists
                        except Exception as e:
                            print(f"[WARNING] Failed to normalize reward_list: {e}")

                    # 打印 multi_turn_reward 样本数用于调试
                    multi_turn_count = sum(1 for rl in batch.non_tensor_batch['reward_list'] if len(rl) > 1)
                    print(f"[DEBUG] multi_turn_reward_samples: {multi_turn_count}/{len(envs)}")

                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                        batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()

                    # metrics for actions
                    batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                        batch.non_tensor_batch['success_env'][idx] = int(env.success())
                        tracking_vars = env.get_tracking_variables()
                        batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                        batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                        batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                        # 避免除以零：如果 actions 为空，ratio 设为 0
                        actions_len = len(tracking_vars['actions'])
                        if actions_len > 0:
                            batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / actions_len
                        else:
                            batch.non_tensor_batch['effective_action_ratio'][idx] = 0.0

                    # 【方案 A】从 final_gen_batch_output.meta_info 取 action_end_positions，
                    # 存到 non_tensor_batch（这样后续 repeat/reorder 时会跟着样本走）
                    _action_end_positions = getattr(final_gen_batch_output, "meta_info", {}).get("action_end_positions", None)
                    if _action_end_positions is not None:
                        batch.non_tensor_batch["action_end_positions"] = np.array(_action_end_positions, dtype=object)
                        print(f"[方案A] 已将 action_end_positions 从 meta_info 转移到 non_tensor_batch，长度={len(_action_end_positions)}")

                    # 【修复】reward_list 和 action_end_positions 都是不规则数组（每个样本长度不同），numpy.repeat 无法处理
                    # 先临时移除，repeat 后手动处理
                    _reward_list_backup = batch.non_tensor_batch.pop('reward_list', None)
                    _action_end_positions_backup = batch.non_tensor_batch.pop('action_end_positions', None)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # 手动 repeat reward_list 并放回（使用 list(rl) 避免引用同一对象）
                    # 直接存为 numpy object array，避免后续 reorder 时报错
                    if _reward_list_backup is not None:
                        repeat_n = self.config.actor_rollout_ref.rollout.n
                        repeated_reward_list = []
                        for rl in _reward_list_backup:
                            for _ in range(repeat_n):
                                repeated_reward_list.append(list(rl))  # 复制而非引用
                        # 转成 numpy object array
                        reward_list_np = np.empty(len(repeated_reward_list), dtype=object)
                        for idx, rl in enumerate(repeated_reward_list):
                            reward_list_np[idx] = rl
                        batch.non_tensor_batch['reward_list'] = reward_list_np
                    
                    # 【方案 A】手动 repeat action_end_positions 并放回
                    if _action_end_positions_backup is not None:
                        repeat_n = self.config.actor_rollout_ref.rollout.n
                        repeated_aep = []
                        for aep in _action_end_positions_backup:
                            for _ in range(repeat_n):
                                repeated_aep.append(list(aep))  # 复制而非引用
                        # 转成 numpy object array
                        aep_np = np.empty(len(repeated_aep), dtype=object)
                        for idx, aep in enumerate(repeated_aep):
                            aep_np[idx] = aep
                        batch.non_tensor_batch['action_end_positions'] = aep_np
                        print(f"[方案A] action_end_positions 已 repeat {repeat_n}x，最终长度={len(aep_np)}")
                    
                    # 【关键修复】删除 batch 里已有的 input_ids/attention_mask/position_ids/responses/prompts
                    # 否则 union 不会用 final_gen_batch_output 里的新值覆盖它们
                    # 导致 global_seqlen 仍然使用旧的（未截断的）数据
                    keys_to_remove = ['input_ids', 'attention_mask', 'position_ids', 'responses', 'prompts', 'response_mask']
                    
                    # DEBUG: 打印删除前的 batch keys 和 shapes
                    print(f"[DEBUG] Before removing keys from batch:")
                    print(f"  batch.batch.keys(): {list(batch.batch.keys())}")
                    if 'input_ids' in batch.batch.keys():
                        print(f"  batch.batch['input_ids'].shape: {batch.batch['input_ids'].shape}")
                    if 'attention_mask' in batch.batch.keys():
                        print(f"  batch.batch['attention_mask'].shape: {batch.batch['attention_mask'].shape}")
                    
                    # 使用 TensorDict 的正确方式删除 keys
                    for key in keys_to_remove:
                        if key in batch.batch.keys():
                            batch.batch.pop(key, None)  # 使用 pop 而不是 del
                    
                    # DEBUG: 打印删除后的 batch keys
                    print(f"[DEBUG] After removing keys from batch:")
                    print(f"  batch.batch.keys(): {list(batch.batch.keys())}")
                    
                    # DEBUG: 打印 final_gen_batch_output 的 shapes
                    print(f"[DEBUG] final_gen_batch_output keys and shapes:")
                    print(f"  final_gen_batch_output.batch.keys(): {list(final_gen_batch_output.batch.keys())}")
                    if 'input_ids' in final_gen_batch_output.batch.keys():
                        print(f"  final_gen_batch_output.batch['input_ids'].shape: {final_gen_batch_output.batch['input_ids'].shape}")
                    
                    batch = batch.union(final_gen_batch_output)
                    
                    # DEBUG: 打印 union 后的 batch shapes
                    print(f"[DEBUG] After union:")
                    if 'input_ids' in batch.batch.keys():
                        print(f"  batch.batch['input_ids'].shape: {batch.batch['input_ids'].shape}")
                    if 'attention_mask' in batch.batch.keys():
                        print(f"  batch.batch['attention_mask'].shape: {batch.batch['attention_mask'].shape}")

                    # [DEBUG] 打印一个样本的完整轨迹（PPO更新前）
                    try:
                        log_dir = os.environ.get("RAGEN_READABLE_LOG_DIR", "logs/readable")
                        log_filename = os.environ.get("RAGEN_DEBUG_LOG_FILENAME", "med_dialogue_debug.log")
                        debug_log_path = os.path.join(log_dir, log_filename)
                        
                        # 尝试分别获取 prompts 和 responses（如果存在）
                        prompt_text = "N/A"
                        response_text = "N/A"
                        full_text = "N/A"
                        
                        if 'prompts' in batch.batch and 'responses' in batch.batch:
                            # Prompt
                            p_ids = batch.batch['prompts'][0]
                            p_valid = p_ids[p_ids != self.tokenizer.pad_token_id]
                            prompt_text = self.tokenizer.decode(p_valid, skip_special_tokens=False)
                            
                            # Response
                            r_ids = batch.batch['responses'][0]
                            r_valid = r_ids[r_ids != self.tokenizer.pad_token_id]
                            response_text = self.tokenizer.decode(r_valid, skip_special_tokens=False)
                            
                            full_text = prompt_text + response_text
                        else:
                            # Fallback to full input_ids
                            input_ids_sample = batch.batch['input_ids'][0]
                            non_pad_mask = input_ids_sample != self.tokenizer.pad_token_id
                            valid_ids = input_ids_sample[non_pad_mask]
                            full_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                        
                        with open(debug_log_path, "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"[DEBUG][RayTrainer] PPO更新前完整轨迹样本\n")
                            f.write(f"{'='*80}\n")
                            if prompt_text != 'N/A':
                                f.write(f"[PROMPT]\n{prompt_text}\n")
                                f.write(f"{'-'*40}\n")
                                f.write(f"[RESPONSE]\n{response_text}\n")
                            else:
                                f.write(f"[FULL INPUT_IDS]\n{full_text}\n")
                            f.write(f"{'='*80}\n")
                    except Exception as e:
                        print(f"[WARNING] Failed to log full trajectory: {e}")

                    ####################
                    ####################

                    if self.config.actor_rollout_ref.actor.state_masking:
                        batch, metrics = self._create_loss_mask(batch, metrics)
                    else:
                        # 【关键修复】保留 generation.py 传入的 response_mask (= action_mask)
                        # action_mask 只标记医生输出的 token，排除 system/user/assistant 等格式 token
                        # 如果已经有 response_mask，就不要用 attention_mask 覆盖
                        if "response_mask" not in batch.batch:
                            print(f"[WARNING] response_mask 不在 batch 中，回退到 attention_mask 计算", flush=True)
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        else:
                            # 验证 response_mask 的有效性
                            resp_mask = batch.batch["response_mask"]
                            resp_mask_sum = resp_mask.sum().item()
                            attn_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            attn_resp_sum = attn_mask[:, -responses.shape[1]:].sum().item()
                            print(f"[RESPONSE_MASK_CHECK] 保留 generation.py 传入的 response_mask (只含医生 token)", flush=True)
                            print(f"  response_mask.sum()={resp_mask_sum:.0f}, attention_mask[responses].sum()={attn_resp_sum:.0f}", flush=True)
                            print(f"  医生 token 占比: {resp_mask_sum/max(attn_resp_sum,1)*100:.1f}%", flush=True)
                    
                    # DEBUG: 打印 _balance_batch 之前的 attention_mask shape
                    print(f"[DEBUG] Before _balance_batch:")
                    print(f"  batch.batch['attention_mask'].shape: {batch.batch['attention_mask'].shape}")
                    attn_sum = batch.batch['attention_mask'].sum(dim=-1)
                    print(f"  attention_mask sum per sample (first 4): {attn_sum[:4].tolist()}")
                    print(f"  attention_mask sum max: {attn_sum.max().item()}")
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        # 【修复】reward_list 是 Python list，需要转成 numpy object array 才能被索引
                        if 'reward_list' in batch.non_tensor_batch:
                            reward_list_py = batch.non_tensor_batch['reward_list']
                            # 转成 numpy object array（每个元素是一个 list）
                            reward_list_np = np.empty(len(reward_list_py), dtype=object)
                            for idx, rl in enumerate(reward_list_py):
                                reward_list_np[idx] = rl
                            batch.non_tensor_batch['reward_list'] = reward_list_np
                        
                        # 【方案 A】action_end_positions 同样需要转成 numpy object array
                        if 'action_end_positions' in batch.non_tensor_batch:
                            aep_py = batch.non_tensor_batch['action_end_positions']
                            aep_np = np.empty(len(aep_py), dtype=object)
                            for idx, aep in enumerate(aep_py):
                                aep_np[idx] = aep
                            batch.non_tensor_batch['action_end_positions'] = aep_np

                        self._balance_batch(batch, metrics=metrics)

                        # reorder 后 reward_list 和 action_end_positions 已经是 numpy object array，保持这个格式

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        
                        # [DEBUG] 打印完整PPO轨迹
                        log_ppo_trajectory(batch, global_step=self.global_steps, batch_idx=0, tokenizer=self.tokenizer)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            # if self.config.actor_rollout_ref.actor.state_masking:
                            #     batch,metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                # collect metrics (先收集，以便 best checkpoint 评分使用)
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # 获取诊断准确率指标（从 env_class 的类变量中）
                diag_metrics = {}
                try:
                    if hasattr(self.env_class, 'get_diagnosis_metrics'):
                        diag_metrics = self.env_class.get_diagnosis_metrics()
                        metrics.update(diag_metrics)
                except Exception as e:
                    print(f"[WARNING] Failed to get diagnosis metrics: {e}")

                # 保存 checkpoint 和 best checkpoint
                if self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    
                    # 检查并保存 best checkpoint（只看 validate_metric/diagnosis_accuracy_overall 是否创新高）
                    if self.config.trainer.get('save_best_checkpoint', True):
                        try:
                            # diag_metrics 可为空，函数内部只看 validate_metric/*
                                self._save_best_checkpoint_v2(diag_metrics, metrics)
                        except Exception as e:
                            print(f"[WARNING] Failed to save best checkpoint: {e}")
                    
                    # 检查并保存 stable_resume_checkpoint（稳定条件 + 准确率最高）
                    try:
                        self._save_stable_resume_checkpoint(metrics)
                    except Exception as e:
                        print(f"[WARNING] Failed to save stable_resume_checkpoint: {e}")

                # 【新增】打印关键 env 指标到 training.log（方便本地排查，不用翻 wandb）
                try:
                    env_keys = [
                        'train/diagnosis_accuracy', 'train/diagnosis_total', 
                        'train/diagnosis_accuracy_cumulative',
                        'train/fallback_per_episode', 'train/pollution_sanitized_per_episode',
                        'train/fallback_count', 'train/pollution_sanitized_count', 'train/episode_count',
                        'train/diagnosis_accuracy_depression', 'train/diagnosis_accuracy_anxiety',
                        'train/diagnosis_accuracy_mix', 'train/diagnosis_accuracy_others',
                        # Analysis LLM 解析失败率（监控分析模型输出质量）
                        'train/analysis_fail_rate', 'train/analysis_fail_count',
                        'train/severe_repair_per_episode',  # severe repair 比例
                    ]
                    # 修改：始终包含 analysis_fail_rate 和 severe_repair，即使值为 N/A
                    env_stats = {}
                    for k in env_keys:
                        if k in metrics:
                            env_stats[k] = metrics[k]
                        elif 'analysis_fail' in k or 'severe_repair' in k:
                            # 对于关键监控指标，即使不存在也输出 N/A
                            env_stats[k] = 'N/A'
                    if env_stats:
                        env_stats_str = ' - '.join([f"{k.split('/')[-1]}:{v:.4f}" if isinstance(v, float) else f"{k.split('/')[-1]}:{v}" for k, v in env_stats.items()])
                        print(f"[ENV_METRICS] step:{self.global_steps} - {env_stats_str}")
                except Exception as e:
                    print(f"[WARNING] Failed to print env metrics: {e}")
                
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        metrics.update(val_metrics)
                    return
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        if "response_mask" in batch.batch.keys():
            response_mask = batch.batch['response_mask']
        else:
            response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # Initialize state mask
        state_mask = torch.ones_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]
    
        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            start_marker = self.config.algorithm.state_masking.start_state_marker
            end_marker = self.config.algorithm.state_masking.end_state_marker   
            
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            
            prev_end = 0
            prev_end_token_pos = 0
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[prev_end:start]
                state_section = response[start:end]
                prev_end = end
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)

                start_token_pos = len(start_tokens) + prev_end_token_pos
                end_token_pos = start_token_pos + len(state_tokens)
                prev_end_token_pos = end_token_pos
                
                state_mask[i, start_token_pos:end_token_pos] = 0
        
        loss_mask = state_mask * response_mask # 1 for valid tokens, 0 for masked tokens
        batch.batch['loss_mask'] = loss_mask
        batch.batch['response_mask'] = loss_mask
        batch.batch['critic_response_mask'] = state_mask * batch.batch['attention_mask'][:, -response_length-1:-1]
        
        # Debug print
        print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch.batch['responses'][0]))
        response_ids = batch.batch['responses'][0]
        unmasked_ids = response_ids[loss_mask[0] == 1]
        print("\nUnmasked batch[0] (after masking):\n", self.tokenizer.decode(unmasked_ids))
        
        masked_ids = response_ids[loss_mask[0] == 0]
        print("\nMasked batch[0] (masked parts):\n", self.tokenizer.decode(masked_ids))
        
        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        import os
        import json
        # Initialize global metric storage
        global_token_scores = []
        global_metrics = {}
        metrics = defaultdict(list)
        
        # 【新增】诊断准确率和联合分析指标统计
        # diagnosed_only 口径：只统计产出诊断的环境
        val_diag_total = 0
        val_diag_correct = 0
        val_class_diag_total = {"Depression": 0, "Anxiety": 0, "Mix": 0, "Others": 0}
        val_class_diag_correct = {"Depression": 0, "Anxiety": 0, "Mix": 0, "Others": 0}
        # overall 口径：统计所有环境（没诊断的也算错）
        val_diag_total_overall = 0
        val_diag_correct_overall = 0
        val_class_diag_total_overall = {"Depression": 0, "Anxiety": 0, "Mix": 0, "Others": 0}
        val_class_diag_correct_overall = {"Depression": 0, "Anxiety": 0, "Mix": 0, "Others": 0}
        # finished_only 口径：只统计 finished 的环境（correct/finished）
        val_finished_total = 0
        val_finished_correct = 0
        val_empathy_scores = []
        val_naturalness_scores = []

        self.val_num += 1

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger = self.logger,
            is_validation = True,
        )

        # 【修复】当 val_batch_size 为 None 时，使用验证数据集的实际大小
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
            print(f"[VAL] val_batch_size is None, using dataset size: {val_batch_size}")
        envs = [self.val_env.copy() for _ in range(val_batch_size)] # do not repeat
        # envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        # 用于给验证阶段的输出目录做唯一标识，避免多个 batch 覆盖同一个 step_1 目录
        val_global_steps = 0

        def _safe_json_dump(obj, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        def _dump_validation_dialogues_to_json(envs, env_seeds, output_dir: str):
            """
            保存验证阶段生成的完整对话轨迹到 JSON 文件。
            每个 env 一个 json，放在 output_dir/val_dialogues_json/ 下。
            """
            # 可用 config 控制开关（默认开启）
            if not self.config.trainer.get("save_val_dialogues_json", True):
                return

            dump_dir = os.path.join(output_dir, "val_dialogues_json")
            os.makedirs(dump_dir, exist_ok=True)

            for idx, env in enumerate(envs):
                try:
                    seed = env_seeds[idx] if idx < len(env_seeds) else None
                    tracking_vars = env.get_tracking_variables() if hasattr(env, "get_tracking_variables") else {}
                    actions = (tracking_vars.get("actions", []) or [])
                    actions_valid = (tracking_vars.get("actions_valid", []) or [])
                    actions_effective = (tracking_vars.get("actions_effective", []) or [])

                    patient_responses = getattr(env, "_trajectory_patient_responses", None)
                    if patient_responses is None:
                        patient_responses = []

                    # 兼容：如果长度不一致，尽量对齐到最短（避免写出错位）
                    n = min(len(actions), len(patient_responses)) if patient_responses else len(actions)

                    turns = []
                    for t in range(n):
                        turns.append(
                            {
                                "turn": t,
                                "doctor_action": actions[t],
                                "patient_response": patient_responses[t] if t < len(patient_responses) else "",
                                "action_valid": actions_valid[t] if t < len(actions_valid) else None,
                                "action_effective": actions_effective[t] if t < len(actions_effective) else None,
                            }
                        )

                    # GT / Pred（与上面的统计逻辑保持一致，尽量用 env._data + canonicalize）
                    gt_raw = None
                    pred_raw = getattr(env, "_predicted_diagnosis", None)
                    try:
                        gt_raw = env._data[env.index]["target"]["diagnosis"]
                    except Exception:
                        try:
                            gt_raw = (getattr(env, "reward_model", {}) or {}).get("ground_truth", {}).get("diagnosis", None)
                        except Exception:
                            gt_raw = None

                    # Pred 兜底：尝试从最后 action 解析
                    if not pred_raw:
                        try:
                            if actions:
                                is_diag, parsed_diag, parsed_rec, parse_mode = env._parse_diagnosis_and_recommendation(
                                    actions[-1], return_parse_mode=True
                                )
                                if is_diag and parsed_diag:
                                    pred_raw = parsed_diag
                        except Exception as e:
                            print(f"[VAL_PRED_PARSE_WARNING] Env {idx}: parse from last action failed: {e}", flush=True)

                    gt_label = env._canonicalize_diag_label(gt_raw) if hasattr(env, "_canonicalize_diag_label") else gt_raw
                    pred_label = env._canonicalize_diag_label(pred_raw) if (pred_raw and hasattr(env, "_canonicalize_diag_label")) else pred_raw

                    # 生成一个便于肉眼看的完整文本
                    full_text_lines = []
                    for t in turns:
                        full_text_lines.append(f"医生: {t['doctor_action']}")
                        full_text_lines.append(f"患者: {t['patient_response']}")
                    full_dialogue_text = "\n".join(full_text_lines)

                    payload = {
                        "val_num": int(self.val_num),
                        "val_batch_step": int(val_global_steps),
                        "env_idx_in_batch": int(idx),
                        "seed": seed,
                        "finished": int(env.finished()) if hasattr(env, "finished") else None,
                        "success": int(env.success()) if hasattr(env, "success") else None,
                        "reward": getattr(env, "reward", None),
                        "gt_raw": gt_raw,
                        "gt_label": gt_label,
                        "pred_raw": pred_raw,
                        "pred_label": pred_label,
                        "turns": turns,
                        "full_dialogue_text": full_dialogue_text,
                    }

                    fname = f"env_{idx}_seed_{seed}.json" if seed is not None else f"env_{idx}.json"
                    _safe_json_dump(payload, os.path.join(dump_dir, fname))
                except Exception as e:
                    # 不因为 dump 失败影响验证流程
                    print(f"[VAL_DIALOGUE_DUMP_WARNING] Env {idx} dump failed: {e}", flush=True)

        for batch_dict in self.val_dataloader:
            val_global_steps += 1
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

            env_seeds = [i['index'] for i in test_batch.non_tensor_batch['extra_info']]
            print("env_seeds:", env_seeds)
            
            # 【修复】根据实际 batch 大小调整 envs 数量，避免 envs 与 predictions 数量不匹配
            actual_batch_size = len(env_seeds)
            if len(envs) > actual_batch_size:
                envs = envs[:actual_batch_size]
            elif len(envs) < actual_batch_size:
                # 需要更多 envs，创建新的
                while len(envs) < actual_batch_size:
                    envs.append(self.val_env.copy())
            
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                output_dir = (f"{self.config.logging.log_image_dir}/"
                                f"{self.config.trainer.experiment_name}/"
                                f"validation_{self.val_num}/"
                                f"step_{val_global_steps}")
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        envs=envs,
                        initial_input_ids=first_input_ids,
                        output_dir=output_dir,
                        global_steps=val_global_steps,
                    )
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)

                test_batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['reward'][idx] = env.reward

                # 【新增】保存验证阶段生成的完整对话轨迹（JSON）
                _dump_validation_dialogues_to_json(envs=envs, env_seeds=env_seeds, output_dir=output_dir)

                if test_batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    test_batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        test_batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()
                    metrics['bandit_metrics'].append(test_batch.non_tensor_batch['bandit_metrics'])
                
                test_batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                    test_batch.non_tensor_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])
                    
                    # 【修复】统一口径：使用 env._canonicalize_diag_label 和 env._data 的 GT
                    # 这与环境内部判定逻辑完全一致
                    try:
                        gt_raw = None
                        pred_raw = getattr(env, "_predicted_diagnosis", None)
                        
                        # GT：优先用 env._data（与 readable/环境内部一致）
                        try:
                            gt_raw = env._data[env.index]["target"]["diagnosis"]
                        except Exception:
                            # 兜底：如果 env._data 不可用，再尝试 reward_model
                            try:
                                gt_raw = (getattr(env, "reward_model", {}) or {}).get("ground_truth", {}).get("diagnosis", None)
                            except Exception:
                                gt_raw = None
                        
                        # Pred：如果 env 没保存，就从最后 action 解析一次
                        if not pred_raw:
                            try:
                                actions = tracking_vars.get("actions", []) or []
                                if actions:
                                    is_diag, parsed_diag, parsed_rec, parse_mode = env._parse_diagnosis_and_recommendation(
                                        actions[-1], return_parse_mode=True
                                    )
                                    if is_diag and parsed_diag:
                                        pred_raw = parsed_diag
                            except Exception as e:
                                print(f"[VAL_PRED_PARSE_WARNING] Env {idx}: parse from last action failed: {e}", flush=True)
                        
                        # 使用 env._canonicalize_diag_label 归一化到四类
                        gt_label = env._canonicalize_diag_label(gt_raw)
                        pred_label = env._canonicalize_diag_label(pred_raw) if pred_raw else None
                        
                        # overall 口径：所有环境都计入（没诊断算错）
                        val_diag_total_overall += 1
                        val_class_diag_total_overall[gt_label] = val_class_diag_total_overall.get(gt_label, 0) + 1
                        if pred_raw and pred_label == gt_label:
                            val_diag_correct_overall += 1
                            val_class_diag_correct_overall[gt_label] = val_class_diag_correct_overall.get(gt_label, 0) + 1
                        
                        # 【修改】diagnosed_only 口径：只统计 finished 且产出诊断的环境（与训练一致）
                        # 训练时只有诊断格式正确才会进入统计分支，所以验证也要求 finished
                        is_finished = env.finished() if hasattr(env, 'finished') else False
                        if is_finished and pred_raw:
                            val_diag_total += 1
                            val_class_diag_total[gt_label] = val_class_diag_total.get(gt_label, 0) + 1
                            if pred_label == gt_label:
                                val_diag_correct += 1
                                val_class_diag_correct[gt_label] = val_class_diag_correct.get(gt_label, 0) + 1
                        
                        # finished_only 口径：只统计 finished 的环境（不管有没有诊断）
                        if is_finished:
                            val_finished_total += 1
                            if pred_raw and pred_label == gt_label:
                                val_finished_correct += 1
                        
                        # 打印详细诊断信息
                        print(f"[VAL_DIAG_DEBUG] Env {idx}: gt_raw='{gt_raw}' -> '{gt_label}', "
                            f"pred_raw='{pred_raw}' -> '{pred_label}', "
                            f"match={pred_label == gt_label if pred_raw else 'N/A'}, finished={is_finished}", flush=True)
                    except Exception as e:
                        print(f"[VAL_DIAG_COLLECT_ERROR] Env {idx}: {e}", flush=True)
                    
                    # 【新增】收集共情和自然性分数
                    if hasattr(env, 'empathy_scores') and env.empathy_scores:
                        val_empathy_scores.extend(env.empathy_scores)
                    if hasattr(env, 'naturalness_scores') and env.naturalness_scores:
                        val_naturalness_scores.extend(env.naturalness_scores)

                # action metrics
                metrics['total_env'].append(test_batch.non_tensor_batch['total_env'])
                metrics['finished_env'].append(test_batch.non_tensor_batch['finished_env'])
                metrics['success_env'].append(test_batch.non_tensor_batch['success_env'])
                metrics['traj_length'].append(test_batch.non_tensor_batch['traj_length'])
                metrics['valid_action'].append(test_batch.non_tensor_batch['valid_action'])
                metrics['effective_action'].append(test_batch.non_tensor_batch['effective_action'])
                metrics['effective_action_ratio'].append(test_batch.non_tensor_batch['effective_action_ratio'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch.non_tensor_batch['reward'])


        global_scores = np.concatenate(global_token_scores, axis=0)
        
        # 辅助函数：安全地将可能包含嵌套结构的列表转换为扁平数组
        def safe_flatten_sum(lst, dtype=np.int32):
            """将可能包含嵌套列表的结构展平后求和"""
            flat = []
            for item in lst:
                if isinstance(item, (list, np.ndarray)):
                    flat.extend(np.array(item).flatten().tolist())
                else:
                    flat.append(item)
            return int(np.array(flat, dtype=dtype).sum())
        
        def safe_flatten_mean(lst, dtype=np.float32):
            """将可能包含嵌套列表的结构展平后求均值"""
            flat = []
            for item in lst:
                if isinstance(item, (list, np.ndarray)):
                    flat.extend(np.array(item).flatten().tolist())
                else:
                    flat.append(item)
            arr = np.array(flat, dtype=dtype)
            return float(arr.mean()) if len(arr) > 0 else 0.0
        
        global_metrics = {
            'global_score/mean': float(global_scores.mean()),
            'global_score/max': float(global_scores.max()),
            'global_score/min': float(global_scores.min()),
            'global_score/std': float(global_scores.std()),
            'validate_metric/total_env': safe_flatten_sum(metrics['total_env']),
            'validate_metric/finished_env': safe_flatten_sum(metrics['finished_env']),
            'validate_metric/success_env': safe_flatten_sum(metrics['success_env']),
            'validate_metric/traj_length': safe_flatten_mean(metrics['traj_length']),
            'validate_metric/valid_action': safe_flatten_mean(metrics['valid_action']),
            'validate_metric/effective_action': safe_flatten_mean(metrics['effective_action']),
            'validate_metric/effective_action_ratio': safe_flatten_mean(metrics['effective_action_ratio']),
        }
        if 'bandit_metrics' in metrics: # NOTE hard code for two-armed bandit
            batch_action = np.array(metrics['bandit_metrics'], dtype=np.int16)
            global_metrics['validate_metric/n_low_arm'] = int(np.sum(batch_action == 1))
            global_metrics['validate_metric/n_high_arm'] = int(np.sum(batch_action == 2))
            global_metrics['validate_metric/n_invalid'] = int(np.sum(batch_action == 0))
        
        # 【修复】诊断准确率指标 - 两种口径
        # diagnosed_only 口径：只统计产出诊断的环境
        global_metrics['validate_metric/diagnosis_total'] = val_diag_total
        global_metrics['validate_metric/diagnosis_correct'] = val_diag_correct
        global_metrics['validate_metric/diagnosis_accuracy'] = val_diag_correct / max(val_diag_total, 1)
        
        # overall 口径：统计所有环境（没诊断的也算错）- 这是更严格的指标
        global_metrics['validate_metric/diagnosis_total_overall'] = val_diag_total_overall
        global_metrics['validate_metric/diagnosis_correct_overall'] = val_diag_correct_overall
        global_metrics['validate_metric/diagnosis_accuracy_overall'] = val_diag_correct_overall / max(val_diag_total_overall, 1)
        
        # finished_only 口径：只统计 finished 的环境（correct/finished）- 完成率加权准确率
        global_metrics['validate_metric/finished_total'] = val_finished_total
        global_metrics['validate_metric/finished_correct'] = val_finished_correct
        global_metrics['validate_metric/diagnosis_accuracy_finished'] = val_finished_correct / max(val_finished_total, 1)
        # 完成率 = finished / total_env
        global_metrics['validate_metric/completion_rate'] = val_finished_total / max(val_diag_total_overall, 1)
        
        # 各类别诊断准确率 - diagnosed_only 口径
        for label in ['Depression', 'Anxiety', 'Mix', 'Others']:
            total = val_class_diag_total.get(label, 0)
            correct = val_class_diag_correct.get(label, 0)
            global_metrics[f'validate_metric/diagnosis_accuracy_{label.lower()}'] = correct / max(total, 1)
            global_metrics[f'validate_metric/diagnosis_total_{label.lower()}'] = total
            global_metrics[f'validate_metric/diagnosis_correct_{label.lower()}'] = correct
        
        # 各类别诊断准确率 - overall 口径
        for label in ['Depression', 'Anxiety', 'Mix', 'Others']:
            total = val_class_diag_total_overall.get(label, 0)
            correct = val_class_diag_correct_overall.get(label, 0)
            global_metrics[f'validate_metric/diagnosis_accuracy_{label.lower()}_overall'] = correct / max(total, 1)
            global_metrics[f'validate_metric/diagnosis_total_{label.lower()}_overall'] = total
            global_metrics[f'validate_metric/diagnosis_correct_{label.lower()}_overall'] = correct
        
        # 【新增】共情和自然性指标
        if val_empathy_scores:
            global_metrics['validate_metric/empathy_mean'] = float(np.mean(val_empathy_scores))
            global_metrics['validate_metric/empathy_std'] = float(np.std(val_empathy_scores))
            global_metrics['validate_metric/empathy_max'] = float(np.max(val_empathy_scores))
            global_metrics['validate_metric/empathy_min'] = float(np.min(val_empathy_scores))
        else:
            global_metrics['validate_metric/empathy_mean'] = 0.0
            global_metrics['validate_metric/empathy_std'] = 0.0
            global_metrics['validate_metric/empathy_max'] = 0.0
            global_metrics['validate_metric/empathy_min'] = 0.0
            
        if val_naturalness_scores:
            global_metrics['validate_metric/naturalness_mean'] = float(np.mean(val_naturalness_scores))
            global_metrics['validate_metric/naturalness_std'] = float(np.std(val_naturalness_scores))
            global_metrics['validate_metric/naturalness_max'] = float(np.max(val_naturalness_scores))
            global_metrics['validate_metric/naturalness_min'] = float(np.min(val_naturalness_scores))
        else:
            global_metrics['validate_metric/naturalness_mean'] = 0.0
            global_metrics['validate_metric/naturalness_std'] = 0.0
            global_metrics['validate_metric/naturalness_max'] = 0.0
            global_metrics['validate_metric/naturalness_min'] = 0.0
        
        # 打印更详细的验证结果
        print("=" * 80)
        print("[VALIDATION RESULTS]")
        print(f"  【Overall口径】所有环境: {val_diag_total_overall}个, 正确: {val_diag_correct_overall}个, 准确率: {val_diag_correct_overall / max(val_diag_total_overall, 1):.2%}")
        for label in ['Depression', 'Anxiety', 'Mix', 'Others']:
            total = val_class_diag_total_overall.get(label, 0)
            correct = val_class_diag_correct_overall.get(label, 0)
            acc = correct / max(total, 1)
            print(f"    {label}: {correct}/{total} = {acc:.2%}")
        print(f"  【Diagnosed口径(=训练口径)】finished且有诊断: {val_diag_total}个, 正确: {val_diag_correct}个, 准确率: {val_diag_correct / max(val_diag_total, 1):.2%}")
        for label in ['Depression', 'Anxiety', 'Mix', 'Others']:
            total = val_class_diag_total.get(label, 0)
            correct = val_class_diag_correct.get(label, 0)
            acc = correct / max(total, 1)
            print(f"    {label}: {correct}/{total} = {acc:.2%}")
        print(f"  【Finished口径】完成的环境: {val_finished_total}个, 正确: {val_finished_correct}个, 准确率: {val_finished_correct / max(val_finished_total, 1):.2%}")
        print(f"  【完成率】{val_finished_total}/{val_diag_total_overall} = {val_finished_total / max(val_diag_total_overall, 1):.2%}")
        unfinished_count = val_diag_total_overall - val_finished_total
        print(f"  【未完成数】{unfinished_count}个环境未 finished（可能是补考超时）")
        if val_empathy_scores:
            print(f"  共情分数: mean={np.mean(val_empathy_scores):.2f}, std={np.std(val_empathy_scores):.2f}")
        if val_naturalness_scores:
            print(f"  自然性分数: mean={np.mean(val_naturalness_scores):.2f}, std={np.std(val_naturalness_scores):.2f}")
        print("=" * 80)
        
        print("global_metrics", global_metrics)
        self.logger.log(data=global_metrics, step=self.val_num)
        return global_metrics
    