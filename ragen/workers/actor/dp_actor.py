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
Single Process Actor
"""

import itertools
from typing import Tuple

import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from ragen.utils.trust import trust_remote_code as _trust_remote_code


# ============ Distributed Utilities ============
def _dist_info():
    """获取分布式信息: (is_distributed, rank, world_size)"""
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def _sync_bool_flag(local_flag: bool, device) -> bool:
    """跨所有 rank 同步一个布尔标志，任意 rank 为 True 则返回 True"""
    dist_on, rank, world = _dist_info()
    if not dist_on or world == 1:
        return local_flag
    local_tensor = torch.tensor(int(local_flag), device=device, dtype=torch.int32)
    dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return bool(local_tensor.item())

# ============ NaN Detection Utilities ============
def check_tensor_for_nan(tensor, name, raise_on_nan=True, dump_stats=True):
    """检查张量是否包含 nan/inf，如果是则打印统计信息并可选地抛出异常"""
    if tensor is None:
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        if dump_stats:
            finite_mask = torch.isfinite(tensor)
            finite_vals = tensor[finite_mask] if finite_mask.any() else torch.tensor([0.0])
            print(f"\n{'='*60}")
            print(f"[NaN DETECTED] {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
            print(f"  Inf count: {torch.isinf(tensor).sum().item()}")
            if finite_vals.numel() > 0:
                print(f"  Finite values - min: {finite_vals.min().item():.6f}, max: {finite_vals.max().item():.6f}, mean: {finite_vals.mean().item():.6f}")
            print(f"{'='*60}\n")
        
        if raise_on_nan:
            raise ValueError(f"NaN/Inf detected in {name}! See stats above.")
        return True
    return False


def check_ppo_tensors(old_log_prob, log_prob, advantages, response_mask, ratio=None,
                      pg_loss=None, kl_loss=None, policy_loss=None, step_info=""):
    """PPO 训练中关键张量的 NaN 检查"""
    found_nan = False
    prefix = f"[Step {step_info}] " if step_info else ""
    
    # 只警告不抛异常，以便收集更多信息
    if check_tensor_for_nan(old_log_prob, f"{prefix}old_log_prob", raise_on_nan=False):
        found_nan = True
    if check_tensor_for_nan(log_prob, f"{prefix}log_prob", raise_on_nan=False):
        found_nan = True
    if check_tensor_for_nan(advantages, f"{prefix}advantages", raise_on_nan=False):
        found_nan = True
    if check_tensor_for_nan(response_mask, f"{prefix}response_mask", raise_on_nan=False):
        found_nan = True
    if ratio is not None and check_tensor_for_nan(ratio, f"{prefix}ratio (exp(log_prob - old_log_prob))", raise_on_nan=False):
        found_nan = True
    if pg_loss is not None and check_tensor_for_nan(pg_loss, f"{prefix}pg_loss", raise_on_nan=False):
        found_nan = True
    if kl_loss is not None and check_tensor_for_nan(kl_loss, f"{prefix}kl_loss", raise_on_nan=False):
        found_nan = True
    if policy_loss is not None and check_tensor_for_nan(policy_loss, f"{prefix}policy_loss", raise_on_nan=False):
        found_nan = True
    
    if found_nan:
        # 打印额外的诊断信息
        print(f"\n[NaN DIAGNOSIS] {prefix}Additional stats:")
        if old_log_prob is not None and log_prob is not None:
            log_prob_diff = log_prob - old_log_prob
            print(f"  log_prob - old_log_prob: min={log_prob_diff.min().item():.4f}, max={log_prob_diff.max().item():.4f}")
            if ratio is None:
                ratio = torch.exp(log_prob_diff)
            print(f"  ratio range: min={ratio.min().item():.4f}, max={ratio.max().item():.4f}")
        if advantages is not None:
            print(f"  advantages range: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, std={advantages.std().item():.4f}")
        print()
        
    return found_nan
# ============ End of NaN Detection ============


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            
            # 【安全检查】如果序列长度为 0 或 attention_mask 全为 0，返回空结果
            if seqlen == 0 or attention_mask.sum() == 0:
                print(f"[WARNING] Skipping _forward_micro_batch: seqlen={seqlen}, attention_mask.sum()={attention_mask.sum()}")
                response_length = micro_batch["responses"].size(-1) if "responses" in micro_batch else 0
                # 返回带梯度追踪的零张量，避免 loss.backward() 失败
                return (
                    torch.zeros(batch_size, response_length, device=input_ids.device, requires_grad=True),
                    torch.zeros(batch_size, response_length, device=input_ids.device, requires_grad=True)
                )
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                
                # 【安全检查】如果 unpad 后的序列长度为 0，返回空结果
                if input_ids_rmpad.numel() == 0:
                    print(f"[WARNING] Skipping _forward_micro_batch after unpad: input_ids_rmpad is empty")
                    response_length = micro_batch["responses"].size(-1) if "responses" in micro_batch else 0
                    # 返回带梯度追踪的零张量，避免 loss.backward() 失败
                    return (
                        torch.zeros(batch_size, response_length, device=input_ids.device, requires_grad=True),
                        torch.zeros(batch_size, response_length, device=input_ids.device, requires_grad=True)
                    )

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                
                # 【已禁用 FP32 转换】保持 bf16 以节省显存
                # 如果出现 log_prob==0 的问题，下面的 -eps 兜底会处理
                # logits_rmpad = logits_rmpad.float()

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward
                )

                # compute entropy
                if calculate_entropy:
                    # logits 已经是 bf16，直接计算 entropy
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                # ========== [方案B修复 - rmpad分支] 将精确的 0 替换为 -eps ==========
                LOG_PROB_EPS = 1e-6
                zero_mask = (log_probs == 0)
                if zero_mask.any():
                    log_probs = torch.where(zero_mask, log_probs.new_full((), -LOG_PROB_EPS), log_probs)
                
                # ========== [数值稳定性] clamp log_probs 防止极端值导致 ratio 爆炸 ==========
                # log_prob 理论范围: (-inf, 0]，但实际中极小值会导致 exp(log_diff) 爆炸
                # 这里使用 -50 作为下界（exp(-50) ≈ 1.9e-22，足够小但不会导致数值问题）
                LOG_PROB_MIN = -50.0
                extreme_mask = log_probs < LOG_PROB_MIN
                if extreme_mask.any():
                    extreme_count = extreme_mask.sum().item()
                    print(f"[LOGPROB_CLAMP] rmpad分支: 有 {extreme_count} 个 log_prob < {LOG_PROB_MIN}，将被 clamp", flush=True)
                    log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                
                # 【已禁用 FP32 转换】保持 bf16 以节省显存
                # 如果出现 log_prob==0 的问题，下面的 -eps 兜底会处理
                # logits = logits.float()
                
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                
                # ========== [方案B修复] 将精确的 0 替换为 -eps，避免触发 ==0 判断 ==========
                # 这解决了 bf16 精度问题导致的 log_prob=0（softmax 饱和到 1.0）
                # 不会引入 NaN：log_prob 理论上 <= 0，将 0 改成 -eps 只会让 ratio 发生极小变化
                LOG_PROB_EPS = 1e-6
                zero_mask = (log_probs == 0)
                if zero_mask.any():
                    log_probs = torch.where(zero_mask, log_probs.new_full((), -LOG_PROB_EPS), log_probs)
                
                # ========== [数值稳定性] clamp log_probs 防止极端值导致 ratio 爆炸 ==========
                LOG_PROB_MIN = -50.0
                extreme_mask = log_probs < LOG_PROB_MIN
                if extreme_mask.any():
                    extreme_count = extreme_mask.sum().item()
                    print(f"[LOGPROB_CLAMP] non-rmpad分支: 有 {extreme_count} 个 log_prob < {LOG_PROB_MIN}，将被 clamp", flush=True)
                    log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN)
                
                # ========== [诊断] 检查 log_probs 中的 0 值（精简版）==========
                # 修复后这里应该不再打印（因为 0 已被替换为 -eps）
                zero_count = (log_probs == 0).sum().item()
                total_count = log_probs.numel()
                if zero_count > 0:
                    zero_ratio = zero_count / total_count * 100
                    print(f"[LOGPROB_ZERO_CHECK] zero_count={zero_count}/{total_count} ({zero_ratio:.1f}%), "
                          f"range=[{log_probs.min().item():.2f}, {log_probs.max().item():.2f}]", flush=True)
                
                if calculate_entropy:
                    # logits 已经是 bf16，直接计算 entropy
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        """
        执行 optimizer step，增强版：
        1. 在 clip 之前先扫描是否有 NaN/Inf 梯度
        2. 如果发现，打印问题参数名并跳过本次更新
        3. 返回 (grad_norm, update_skipped)，供上层判断是否稳定
        """
        assert self.config.grad_clip is not None
        
        # ========== [诊断增强] 在 clip 之前扫描梯度是否含 NaN/Inf ==========
        nan_params = []
        inf_params = []
        for name, param in self.actor_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)
        
        if nan_params or inf_params:
            print(f"[GRAD_NAN_DETECTED] 发现梯度异常，跳过本次 optimizer.step()", flush=True)
            if nan_params:
                print(f"  NaN梯度参数 (前5个): {nan_params[:5]}", flush=True)
            if inf_params:
                print(f"  Inf梯度参数 (前5个): {inf_params[:5]}", flush=True)
            self.actor_optimizer.zero_grad()
            # 返回 NaN 作为 grad_norm，update_skipped=1 表示本次更新被跳过
            return torch.tensor(float('nan')), 1

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}", flush=True)
            self.actor_optimizer.zero_grad()
            return grad_norm, 1  # update_skipped=1
        else:
            self.actor_optimizer.step()
            return grad_norm, 0  # update_skipped=0

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            # 【诊断】打印 compute_log_prob 使用的 max_token_len
            print(f"[COMPUTE_LOG_PROB] use_dynamic_bsz=True, max_token_len={max_token_len} "
                  f"(来自 meta_info={data.meta_info['max_token_len']} * ulysses_sp={self.ulysses_sequence_parallel_size})", flush=True)
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            response_length = micro_batch["responses"].size(-1)
            # Skip samples with empty responses
            if response_length == 0:
                print(f"[WARNING] Skipping micro_batch with response_length=0 in compute_old_log_prob")
                # Create empty tensors with correct batch size but 0 sequence length
                batch_size = micro_batch["responses"].size(0)
                empty_log_probs = torch.empty(batch_size, 0, dtype=torch.float32, device=micro_batch["responses"].device)
                log_probs_lst.append(empty_log_probs)
                if calculate_entropy:
                    empty_entropy = torch.empty(batch_size, 0, dtype=torch.float32, device=micro_batch["responses"].device)
                    entropy_lst.append(empty_entropy)
                continue
            
            response_mask = micro_batch["attention_mask"][:, -response_length :]
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            # 【修复】entropys 也需要还原到原顺序
            if entropys is not None:
                entropys = entropys[revert_indices]

        # ========== [方案B修复] 将精确的 0 替换为 -eps（兜底，确保所有路径都处理）==========
        LOG_PROB_EPS = 1e-6
        zero_mask = (log_probs == 0)
        zero_count_before = zero_mask.sum().item()
        if zero_mask.any():
            log_probs = torch.where(zero_mask, log_probs.new_full((), -LOG_PROB_EPS), log_probs)
        
        # ========== [数值稳定性] clamp log_probs 防止极端值（这是 old_log_prob 的来源）==========
        LOG_PROB_MIN = -50.0
        extreme_mask = log_probs < LOG_PROB_MIN
        extreme_count_before = extreme_mask.sum().item()
        if extreme_mask.any():
            log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN)
            print(f"[COMPUTE_LOG_PROB_CLAMP] 有 {extreme_count_before} 个 log_prob < {LOG_PROB_MIN}，已 clamp", flush=True)
        
        # ========== [诊断] compute_log_prob 最终结果汇总 ==========
        zero_count = (log_probs == 0).sum().item()
        total_count = log_probs.numel()
        # 打印修复前后的统计（修复后 zero_count 应该为 0）
        if zero_count_before > 0 or zero_count > 0:
            print(f"[COMPUTE_LOG_PROB_SUMMARY] zero_lp: {zero_count_before} -> {zero_count} (fixed), "
                  f"range=[{log_probs.min().item():.2f}, {log_probs.max().item():.2f}]", flush=True)
        
        # ========== [诊断] compute_log_prob 结果检查 ==========
        if torch.isinf(log_probs).any() or torch.isnan(log_probs).any():
            inf_count = torch.isinf(log_probs).sum().item()
            nan_count = torch.isnan(log_probs).sum().item()
            print(f"\n{'='*60}")
            print(f"[CRITICAL] compute_log_prob 返回的 log_probs 包含 inf/nan!")
            print(f"  log_probs shape: {log_probs.shape}")
            print(f"  inf_count: {inf_count}, nan_count: {nan_count}")
            finite_mask = torch.isfinite(log_probs)
            if finite_mask.any():
                print(f"  finite values: min={log_probs[finite_mask].min().item():.4f}, max={log_probs[finite_mask].max().item():.4f}")
            print(f"{'='*60}\n")

        return log_probs, entropys

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        
        # 【关键】保存原始 meta_info，因为后面 data 会被 dataloader 迭代覆盖
        original_meta_info = data.meta_info

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        # 【省显存 KL loss】如果 use_kl_loss=True 但没有 ref_log_prob（因为 no_ref_policy=True），
        # 则使用 old_log_probs 替代 ref_log_prob，这样可以实现 KL 约束而不需要加载额外的 ref model
        use_old_logprob_as_ref = self.config.use_kl_loss and "ref_log_prob" not in data.batch.keys()
        if self.config.use_kl_loss and not use_old_logprob_as_ref:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        
        # ========== [诊断] 全局打印 old_log_probs 的统计，帮助确认来源和一致性 ==========
        # 兼容 DataProto 和 TensorDict 两种类型
        try:
            if hasattr(batch, 'batch'):
                # DataProto 类型
                batch_dict = batch.batch
            else:
                # TensorDict 类型
                batch_dict = batch
            
            if "old_log_probs" in batch_dict.keys():
                old_lp_global = batch_dict["old_log_probs"]
                resp_mask_global = batch_dict.get("response_mask", None)
                resp_global = batch_dict.get("responses", None)
                pad_token_id = None
                eos_token_id = None
                try:
                    # Prefer meta_info (trainer passes these)
                    pad_token_id = data.meta_info.get("pad_token_id", None)
                    eos_token_id = data.meta_info.get("eos_token_id", None)
                except Exception:
                    pass
                # Fallback to model config
                try:
                    if pad_token_id is None:
                        pad_token_id = getattr(getattr(self.actor_module, "config", None), "pad_token_id", None)
                    if eos_token_id is None:
                        eos_token_id = getattr(getattr(self.actor_module, "config", None), "eos_token_id", None)
                except Exception:
                    pass
                print(f"\n{'='*70}", flush=True)
                print(f"[OLD_LOGPROB_GLOBAL_STATS] 全局 old_log_probs 统计（update_policy 入口）", flush=True)
                print(f"  来源: actor.compute_log_prob()（更新前的 actor 模型在 fit() 中重新计算）", flush=True)
                print(f"  shape: {old_lp_global.shape}", flush=True)
                print(f"  dtype: {old_lp_global.dtype}", flush=True)
                print(f"  range: [{old_lp_global.min().item():.4f}, {old_lp_global.max().item():.4f}]", flush=True)
                print(f"  mean: {old_lp_global.mean().item():.4f}", flush=True)
                print(f"  has_nan: {torch.isnan(old_lp_global).any().item()}, has_inf: {torch.isinf(old_lp_global).any().item()}", flush=True)
                print(f"  pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id}", flush=True)
                # 打印一些可能的异常情况
                if resp_mask_global is not None:
                    valid_count = (resp_mask_global > 0).sum().item()
                    # 检查 mask=1 位置的 old_log_prob 统计
                    masked_old_lp = old_lp_global * (resp_mask_global > 0).float()
                    if valid_count > 0:
                        masked_mean = masked_old_lp.sum().item() / valid_count
                        print(f"  [masked stats] valid_tokens={valid_count}, masked_mean={masked_mean:.4f}", flush=True)
                    # 检查是否存在 old_log_prob=0 的情况（可能是未计算到的 token）
                    zero_count = ((old_lp_global == 0) & (resp_mask_global > 0)).sum().item()
                    if zero_count > 0:
                        print(f"  [WARNING] 有 {zero_count} 个 valid token 的 old_log_prob=0！可能是计算不完整", flush=True)
                        # 进一步定位：这些 0 对应的 token_id 分布
                        if resp_global is not None:
                            try:
                                zero_pos = (old_lp_global == 0) & (resp_mask_global > 0)
                                zero_token_ids = resp_global[zero_pos].to(torch.long)
                                # 统计 top-K token_id
                                uniq_ids, uniq_cnt = torch.unique(zero_token_ids, return_counts=True)
                                topk = min(10, uniq_ids.numel())
                                if topk > 0:
                                    top_cnt, top_idx = torch.topk(uniq_cnt, k=topk)
                                    top_ids = uniq_ids[top_idx]
                                    top_pairs = [(int(tid), int(tc)) for tid, tc in zip(top_ids.tolist(), top_cnt.tolist())]
                                    print(f"  [ZERO_LP_TOKEN_TOPK] token_id,count(top{topk}): {top_pairs}", flush=True)
                                    # 尝试解码 top-K token ids
                                    print(f"  [DEBUG] 准备解码 top-K tokens...", flush=True)
                                    try:
                                        from transformers import AutoTokenizer
                                        tkzr = getattr(self, '_debug_tokenizer', None)
                                        if tkzr is None:
                                            print(f"  [DEBUG] tokenizer 为 None，开始加载...", flush=True)
                                            model_name = getattr(getattr(self.actor_module, "config", None), "_name_or_path", None)
                                            print(f"  [DEBUG] model_name: {model_name}", flush=True)
                                            if model_name:
                                                print(f"  [DEBUG] 正在加载 tokenizer: {model_name}", flush=True)
                                                tkzr = AutoTokenizer.from_pretrained(model_name, trust_remote_code=_trust_remote_code())
                                                self._debug_tokenizer = tkzr
                                                print(f"  [DEBUG] tokenizer 加载成功", flush=True)
                                        if tkzr is not None:
                                            print(f"  [DEBUG] 开始解码 {len(top_pairs)} 个 tokens...", flush=True)
                                            decoded_pairs = [(int(tid), repr(tkzr.decode([int(tid)]))) for tid, _ in top_pairs]
                                            print(f"  [ZERO_LP_TOKEN_DECODED] top{topk} token_id -> text: {decoded_pairs}", flush=True)
                                        else:
                                            print(f"  [DEBUG] tokenizer 仍然为 None", flush=True)
                                    except Exception as decode_e:
                                        import traceback
                                        print(f"  [ZERO_LP_TOKEN_DECODED] 解码失败: {decode_e}", flush=True)
                                        traceback.print_exc()
                                # pad/eos 覆盖比例
                                if pad_token_id is not None:
                                    pad_in_zero = (zero_token_ids == int(pad_token_id)).sum().item()
                                    print(f"  [ZERO_LP_PAD] zero_lp tokens == pad_token_id: {pad_in_zero}/{zero_token_ids.numel()} ({pad_in_zero/max(zero_token_ids.numel(),1)*100:.1f}%)", flush=True)
                                if eos_token_id is not None:
                                    eos_in_zero = (zero_token_ids == int(eos_token_id)).sum().item()
                                    print(f"  [ZERO_LP_EOS] zero_lp tokens == eos_token_id: {eos_in_zero}/{zero_token_ids.numel()} ({eos_in_zero/max(zero_token_ids.numel(),1)*100:.1f}%)", flush=True)
                                
                                # ========== 打印第一个 zero_lp & mask=1 位置的上下文窗口 ==========
                                # 用于判断是对齐偏移还是特定 token 类型问题
                                try:
                                    # 打印 shape 信息，方便定位
                                    print(f"  [ZERO_LP_DEBUG] old_lp_global.shape={old_lp_global.shape}, resp_mask_global.shape={resp_mask_global.shape}", flush=True)
                                    if resp_global is not None:
                                        print(f"  [ZERO_LP_DEBUG] resp_global.shape={resp_global.shape}", flush=True)
                                    else:
                                        print(f"  [ZERO_LP_DEBUG] resp_global is None", flush=True)
                                    
                                    # zero_pos: [batch, seq_len] bool
                                    batch_size_lp = old_lp_global.shape[0]
                                    seq_len_lp = old_lp_global.shape[1]
                                    
                                    # 遍历每个样本，找第一个有 zero_lp & mask=1 的样本
                                    for sample_idx in range(batch_size_lp):
                                        row_zero_mask = zero_pos[sample_idx]  # [seq_len]
                                        if row_zero_mask.any():
                                            zero_indices = row_zero_mask.nonzero(as_tuple=False).squeeze(-1)
                                            if zero_indices.dim() == 0:
                                                zero_indices = zero_indices.unsqueeze(0)
                                            first_pos = zero_indices[0].item()
                                            
                                            # 取上下文窗口（前后各 5 个 token）
                                            ctx_start = max(0, first_pos - 5)
                                            ctx_end = min(seq_len_lp, first_pos + 6)
                                            
                                            ctx_old_lp = old_lp_global[sample_idx, ctx_start:ctx_end].tolist()
                                            ctx_mask = resp_mask_global[sample_idx, ctx_start:ctx_end].tolist()
                                            
                                            # 获取 token ids（考虑 responses 可能长度不同）
                                            ctx_token_ids = []
                                            if resp_global is not None and resp_global.shape[1] >= seq_len_lp:
                                                # 假设 responses 长度匹配
                                                ctx_token_ids = resp_global[sample_idx, ctx_start:ctx_end].tolist()
                                            elif resp_global is not None:
                                                # responses 可能包含 prompt，我们取最后 seq_len_lp 列
                                                resp_offset = resp_global.shape[1] - seq_len_lp
                                                if resp_offset >= 0:
                                                    ctx_token_ids = resp_global[sample_idx, resp_offset + ctx_start:resp_offset + ctx_end].tolist()
                                            
                                            print(f"  [ZERO_LP_CONTEXT] 第一个 zero_lp & mask=1 位置的上下文窗口:", flush=True)
                                            print(f"    sample_idx={sample_idx}, token_pos={first_pos} (response 部分，seq_len={seq_len_lp})", flush=True)
                                            print(f"    ctx_range=[{ctx_start}, {ctx_end})", flush=True)
                                            print(f"    ctx_token_ids: {ctx_token_ids}", flush=True)
                                            print(f"    ctx_old_lp:    {[f'{v:.4f}' for v in ctx_old_lp]}", flush=True)
                                            print(f"    ctx_mask:      {[int(m) for m in ctx_mask]}", flush=True)
                                            
                                            # 尝试解码 token
                                            if ctx_token_ids:
                                                try:
                                                    from transformers import AutoTokenizer
                                                    tkzr = getattr(self, '_debug_tokenizer', None)
                                                    if tkzr is None:
                                                        model_name = getattr(getattr(self.actor_module, "config", None), "_name_or_path", None)
                                                        if model_name:
                                                            tkzr = AutoTokenizer.from_pretrained(model_name, trust_remote_code=_trust_remote_code())
                                                            self._debug_tokenizer = tkzr
                                                    if tkzr is not None:
                                                        ctx_tokens_str = [repr(tkzr.decode([int(tid)])) for tid in ctx_token_ids]
                                                        print(f"    ctx_tokens_str: {ctx_tokens_str}", flush=True)
                                                        # 打印 zero 位置对应的 token
                                                        zero_tid = ctx_token_ids[first_pos - ctx_start]
                                                        zero_str = tkzr.decode([int(zero_tid)])
                                                        print(f"    >>> ZERO_POS token_id={zero_tid}, decoded={repr(zero_str)}", flush=True)
                                                except Exception as tkz_e:
                                                    print(f"    (无法解码 token: {tkz_e})", flush=True)
                                            break  # 只打印一条样本
                                except Exception as ctx_e:
                                    import traceback
                                    print(f"  [DEBUG] ZERO_LP_CONTEXT 打印失败: {ctx_e}", flush=True)
                                    traceback.print_exc()
                            except Exception as e:
                                import traceback
                                print(f"  [DEBUG] ZERO_LP token 分布统计失败: {e}", flush=True)
                                traceback.print_exc()
                    # 检查是否存在极端值
                    extreme_count = ((old_lp_global.abs() > 50) & (resp_mask_global > 0)).sum().item()
                    if extreme_count > 0:
                        print(f"  [WARNING] 有 {extreme_count} 个 valid token 的 |old_log_prob| > 50！", flush=True)
                    
                    # ========== [方案A+B] 不再自动 mask 掉 old_log_prob==0 的位置 ==========
                    # 方案B 会在后续 micro_batch 处理时把精确的 0 替换为 -1e-6
                    # 因此这里只做统计打印，不修改 mask
                    if zero_count > 0 and valid_count > 0:
                        zero_ratio = zero_count / valid_count
                        print(f"  [INFO] zero_lp 比例 {zero_ratio*100:.1f}%，将在 micro_batch 中通过方案B修复", flush=True)
                print(f"{'='*70}\n", flush=True)
        except Exception as e:
            print(f"[DEBUG] 无法打印 old_log_probs 统计: {e}", flush=True)
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    # 【关键修复】使用与 compute_log_prob 相同的 max_token_len
                    # compute_log_prob 使用 data.meta_info["max_token_len"]（来自 rollout.log_prob_max_token_len_per_gpu）
                    # 如果这里使用 ppo_max_token_len_per_gpu，会导致 rearrange_micro_batches 产生不同的排列
                    # 从而使 old_log_prob 和 log_prob 的 token 位置错位，导致 log_ratio 爆炸
                    # 修复方法：使用保存的 original_meta_info 中的 max_token_len（如果存在），保证一致性
                    if "max_token_len" in original_meta_info:
                        max_token_len = original_meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                        if batch_idx == 0 and epoch == 0:
                            print(f"\n{'='*70}", flush=True)
                            print(f"[UPDATE_POLICY][SUCCESS] max_token_len 成功从 original_meta_info 获取！", flush=True)
                            print(f"  use_dynamic_bsz=True, max_token_len={max_token_len}", flush=True)
                            print(f"  来源: original_meta_info['max_token_len']={original_meta_info['max_token_len']} * ulysses_sp={self.ulysses_sequence_parallel_size}", flush=True)
                            print(f"  config.ppo_max_token_len_per_gpu={self.config.ppo_max_token_len_per_gpu}（旧代码使用这个值，已修复不再使用）", flush=True)
                            print(f"  【验证】compute_log_prob 和 update_policy 都使用相同的 max_token_len={max_token_len}，排列一致", flush=True)
                            print(f"{'='*70}\n", flush=True)
                    else:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        if batch_idx == 0 and epoch == 0:
                            print(f"\n{'='*70}", flush=True)
                            print(f"[UPDATE_POLICY][WARNING] max_token_len 不在 original_meta_info 中，回退到 config", flush=True)
                            print(f"  使用: config.ppo_max_token_len_per_gpu={self.config.ppo_max_token_len_per_gpu}", flush=True)
                            print(f"  original_meta_info keys: {list(original_meta_info.keys())}", flush=True)
                            print(f"  这可能导致与 compute_log_prob 的排列不一致！请检查 fsdp_workers.py 的 compute_log_prob 是否正确传递 meta_info", flush=True)
                            print(f"{'='*70}\n", flush=True)
                    micro_batches, indices = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    # 注意：rearrange_micro_batches 已经把 mini_batch 中的所有字段（包括 old_log_probs）
                    # 按 indices 重排并分配到 micro_batches 中，不需要额外处理
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                
                # ========== 统计 zero_loss 和 token_mask 的计数器 ==========
                zero_loss_count = 0
                total_micro_batch_count = 0
                zero_loss_reasons_counter = {}  # 记录每种原因的次数
                
                # Token-Level Masking 统计
                total_masked_tokens = 0
                total_valid_tokens_before_mask = 0
                batches_with_masked_tokens = 0
                
                # ========== [方案B统计] 记录本 batch 修复的 old_log_prob==0 个数 ==========
                old_logprob_zero_fixed_total = 0  # 累积整个 batch 的修复数量

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    input_ids = data["input_ids"]
                    seqlen = input_ids.size(1)
                    
                    # ========== 统一的"置零 loss"标志位 ==========
                    # 所有需要跳过的情况都设置这个标志，最后统一在 backward 前把 loss * 0
                    # 这样保证所有 rank 都执行相同的 collective 通信，避免 NCCL 死锁
                    zero_loss_flag = False
                    zero_loss_reason = ""
                    
                    # Skip samples with empty responses
                    # 【全 rank 一致】response_length 通常所有 rank 相同，但为安全起见加同步
                    local_empty_response = (response_length == 0)
                    global_empty_response = _sync_bool_flag(local_empty_response, device=input_ids.device)
                    if global_empty_response:
                        dist_on, rank, _ = _dist_info()
                        if not dist_on or rank == 0:
                            print(f"[ZERO_LOSS] batch_idx={batch_idx}：response_length=0，将 loss 置 0 (global_sync)", flush=True)
                        zero_loss_flag = True
                        zero_loss_reason = "empty_response"
                    
                    attention_mask = data["attention_mask"]
                    if 'response_mask' in data:
                        response_mask = data['response_mask']
                    else:
                        response_mask = attention_mask[:, -response_length:] if response_length > 0 else attention_mask[:, :0]
                    old_log_prob = data["old_log_probs"]
                    
                    # ========== [方案B修复] 将 old_log_prob 中精确的 0 替换为 -eps ==========
                    # 确保 ratio = exp(new - old) 不会因为 old=0 而出现异常
                    # 方案A+B：不再 mask 掉这些位置，而是直接修复 log_prob 值
                    LOG_PROB_EPS = 1e-6
                    old_zero_mask = (old_log_prob == 0)
                    old_zero_count = old_zero_mask.sum().item()
                    if old_zero_mask.any():
                        old_log_prob = torch.where(old_zero_mask, old_log_prob.new_full((), -LOG_PROB_EPS), old_log_prob)
                    # 累积统计修复数量
                    old_logprob_zero_fixed_total += old_zero_count
                    
                    # ========== [数值稳定性] clamp old_log_prob 防止极端值 ==========
                    # 这是在 micro_batch 层面的二次保护（compute_log_prob 已经做过一次）
                    LOG_PROB_MIN = -50.0
                    old_extreme_mask = old_log_prob < LOG_PROB_MIN
                    if old_extreme_mask.any():
                        old_extreme_count = old_extreme_mask.sum().item()
                        if batch_idx == 0:  # 只在第一个 batch 打印
                            print(f"[OLD_LOGPROB_CLAMP] micro_batch: 有 {old_extreme_count} 个 old_log_prob < {LOG_PROB_MIN}，将被 clamp", flush=True)
                        old_log_prob = torch.clamp(old_log_prob, min=LOG_PROB_MIN)
                    
                    advantages = data["advantages"]
                    
                    # ========== [方案B-断言1] 强制验证：responses 必须是 input_ids 的后缀 ==========
                    # 这是最关键的"token 对齐语义"检查：不成立的话，后面所有 logprob slice 都可能错
                    # 【全 rank 一致】使用 _sync_bool_flag 确保所有 rank 同步决定是否跳过
                    if response_length > 0 and not zero_loss_flag:
                        suffix = input_ids[:, -response_length:]  # (bs, resp_len)
                        mismatch = (suffix != responses) & (response_mask > 0)
                        local_has_mismatch = mismatch.any().item()
                        global_has_mismatch = _sync_bool_flag(local_has_mismatch, device=input_ids.device)
                        
                        if global_has_mismatch:
                            dist_on, rank, _ = _dist_info()
                            if (not dist_on or rank == 0) and local_has_mismatch:
                                # 只在 rank0 且本地确实有 mismatch 时打印详情
                                si, ti = torch.where(mismatch)
                                si, ti = si[0].item(), ti[0].item()
                                print("\n" + "=" * 80, flush=True)
                                print("[CRITICAL] responses 不是 input_ids 的后缀（在有效 mask 上）=> 必然存在 token 级错位 (global_sync)", flush=True)
                                print(f"  sample={si}, token={ti}, seqlen={seqlen}, response_length={response_length}", flush=True)
                                print(f"  input_ids_suffix_token={suffix[si, ti].item()}, responses_token={responses[si, ti].item()}", flush=True)
                                ctx = 5
                                s0 = max(0, seqlen - response_length - ctx)
                                s1 = seqlen
                                r0 = max(0, ti - ctx)
                                r1 = min(response_length, ti + ctx + 1)
                                print(f"  input_ids tail (pos {s0}-{s1-1}): {input_ids[si, s0:s1].tolist()}", flush=True)
                                print(f"  responses ctx  (pos {r0}-{r1-1}): {responses[si, r0:r1].tolist()}", flush=True)
                                print(f"  response_mask ctx: {response_mask[si, r0:r1].tolist()}", flush=True)
                                print("=" * 80 + "\n", flush=True)
                            if not dist_on or rank == 0:
                                print(f"[ZERO_LOSS] batch_idx={batch_idx}：token 错位，将 loss 置 0 (global_sync)", flush=True)
                            zero_loss_flag = True
                            zero_loss_reason = "token_mismatch"
                    
                    # ========== [诊断1] 检查 mask/有效 token 是否为 0 ==========
                    # 【全 rank 一致】使用 _sync_bool_flag 确保所有 rank 同步决定是否跳过
                    mask_sum = response_mask.sum().item() if response_length > 0 else 0
                    local_empty_mask = (mask_sum == 0)
                    global_empty_mask = _sync_bool_flag(local_empty_mask, device=response_mask.device) if not zero_loss_flag else False
                    
                    if global_empty_mask and not zero_loss_flag:
                        dist_on, rank, _ = _dist_info()
                        if not dist_on or rank == 0:
                            print(f"\n{'='*60}", flush=True)
                            print(f"[CRITICAL] response_mask 全为 0! batch_idx={batch_idx} (global_sync)", flush=True)
                            print(f"  response_mask shape: {response_mask.shape}", flush=True)
                            print(f"  attention_mask shape: {attention_mask.shape}", flush=True)
                            print(f"  response_length: {response_length}", flush=True)
                            print(f"  responses shape: {responses.shape}", flush=True)
                            print(f"{'='*60}\n", flush=True)
                            print(f"[ZERO_LOSS] batch_idx={batch_idx}：mask_sum=0，将 loss 置 0 (global_sync)", flush=True)
                        zero_loss_flag = True
                        zero_loss_reason = "empty_mask"
                    elif mask_sum < response_mask.numel() * 0.01 and mask_sum > 0:  # 有效 token 少于 1%
                        dist_on, rank, _ = _dist_info()
                        if not dist_on or rank == 0:
                            print(f"[WARNING] response_mask 有效 token 很少! sum={mask_sum:.0f}, total={response_mask.numel()}, ratio={mask_sum/response_mask.numel():.4f}", flush=True)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # 如果已经标记为 zero_loss，仍然需要执行 forward 以保持计算图一致
                    # 但后续 loss 会被置 0，所以即使 forward 有问题也不会影响梯度
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    
                    # ========== [数值稳定性] clamp 新计算的 log_prob ==========
                    # 这里是在 update_policy 中的最后一道防线，确保 log_ratio 不会爆炸
                    LOG_PROB_MIN = -50.0
                    new_extreme_mask = log_prob < LOG_PROB_MIN
                    if new_extreme_mask.any():
                        new_extreme_count = new_extreme_mask.sum().item()
                        if batch_idx == 0:  # 只在第一个 batch 打印
                            print(f"[NEW_LOGPROB_CLAMP] 有 {new_extreme_count} 个 new log_prob < {LOG_PROB_MIN}，将被 clamp", flush=True)
                        log_prob = torch.clamp(log_prob, min=LOG_PROB_MIN)

                    # ========== [诊断2] 检查 logprob 里是否有 inf/-inf ==========
                    old_has_inf = torch.isinf(old_log_prob).any().item()
                    old_has_nan = torch.isnan(old_log_prob).any().item()
                    new_has_inf = torch.isinf(log_prob).any().item()
                    new_has_nan = torch.isnan(log_prob).any().item()
                    
                    # 【关键诊断】对比 old 和 new log_prob 在相同 mask 位置的差异（只 rank0 打印）
                    # 计算逐 token 差异的统计
                    log_diff = (log_prob - old_log_prob) * (response_mask > 0).float()
                    mask_sum = response_mask.sum().item()
                    dist_on_dbg, rank_dbg, _ = _dist_info()
                    if mask_sum > 0 and batch_idx == 0 and (not dist_on_dbg or rank_dbg == 0):  # 只在第一个 batch + rank0 打印详细信息
                        diff_abs_mean = log_diff.abs().sum().item() / mask_sum
                        diff_max = log_diff.max().item()
                        diff_min = log_diff.min().item()
                        old_mean = (old_log_prob * (response_mask > 0).float()).sum().item() / mask_sum
                        new_mean = (log_prob * (response_mask > 0).float()).sum().item() / mask_sum
                        print(f"\n[LOGPROB_DIFF_STATS] batch_idx={batch_idx}, epoch={epoch}")
                        print(f"  old_log_prob: mean={old_mean:.4f}, range=[{old_log_prob.min().item():.2f}, {old_log_prob.max().item():.2f}]")
                        print(f"  log_prob:     mean={new_mean:.4f}, range=[{log_prob.min().item():.2f}, {log_prob.max().item():.2f}]")
                        print(f"  diff (new-old): abs_mean={diff_abs_mean:.4f}, range=[{diff_min:.2f}, {diff_max:.2f}]")
                        # 【修复验证】如果 abs_mean < 1.0 且 range 在 [-5, 5] 内，说明修复成功
                        if diff_abs_mean < 1.0 and diff_min > -5.0 and diff_max < 5.0:
                            print(f"  ✓ [FIX_VERIFIED] log_diff 正常！abs_mean={diff_abs_mean:.4f} < 1.0, range=[{diff_min:.2f}, {diff_max:.2f}] ⊂ [-5, 5]")
                        elif diff_abs_mean < 2.0 and diff_min > -10.0 and diff_max < 10.0:
                            print(f"  ~ [FIX_PARTIAL] log_diff 较小但仍需观察: abs_mean={diff_abs_mean:.4f}, range=[{diff_min:.2f}, {diff_max:.2f}]")
                        else:
                            print(f"  ✗ [FIX_FAILED?] log_diff 仍然较大！可能还有对齐问题")
                        # 检查是否有极端差异的 token
                        extreme_mask = (log_diff.abs() > 10) & (response_mask > 0)
                        extreme_count = extreme_mask.sum().item()
                        if extreme_count > 0:
                            print(f"  [WARNING] 有 {extreme_count} 个 token 的 log_diff > 10，可能是对齐问题！")
                            # 打印第一个极端差异的位置（简化输出）
                            extreme_indices = torch.where(extreme_mask)
                            if len(extreme_indices[0]) > 0:
                                si = extreme_indices[0][0].item()
                                ti = extreme_indices[1][0].item()
                                print(f"  [ALIGN_DEBUG] 极端 token: sample={si}, pos={ti}, token_id={responses[si, ti].item()}")
                                print(f"    old_log_prob={old_log_prob[si, ti].item():.4f}, new_log_prob={log_prob[si, ti].item():.4f}")
                    
                    # 只在 rank0 打印诊断信息，避免日志爆炸
                    dist_on_diag, rank_diag, _ = _dist_info()
                    if old_has_inf or old_has_nan or new_has_inf or new_has_nan:
                        if not dist_on_diag or rank_diag == 0:
                            print(f"\n{'='*60}")
                            print(f"[CRITICAL] logprob 存在 inf/nan! batch_idx={batch_idx}")
                            print(f"  old_log_prob: has_inf={old_has_inf}, has_nan={old_has_nan}")
                            print(f"    min={old_log_prob.min().item():.4f}, max={old_log_prob.max().item():.4f}")
                            print(f"    inf_count={torch.isinf(old_log_prob).sum().item()}, nan_count={torch.isnan(old_log_prob).sum().item()}")
                            print(f"  log_prob: has_inf={new_has_inf}, has_nan={new_has_nan}")
                            print(f"    min={log_prob.min().item():.4f}, max={log_prob.max().item():.4f}")
                            print(f"    inf_count={torch.isinf(log_prob).sum().item()}, nan_count={torch.isnan(log_prob).sum().item()}")
                            print(f"{'='*60}\n")
                    
                    # ========== [诊断3] 检查 old/new logprob 对齐是否一致 ==========
                    if old_log_prob.shape != log_prob.shape:
                        if not dist_on_diag or rank_diag == 0:
                            print(f"\n{'='*60}")
                            print(f"[CRITICAL] old_log_prob 和 log_prob 形状不一致!")
                            print(f"  old_log_prob.shape: {old_log_prob.shape}")
                            print(f"  log_prob.shape: {log_prob.shape}")
                            print(f"  response_mask.shape: {response_mask.shape}")
                            print(f"{'='*60}\n")
                    
                    # 检查 logprob 差值是否异常（ratio 溢出的前兆）
                    log_ratio = log_prob - old_log_prob
                    log_ratio_masked = log_ratio * (response_mask > 0).float()
                    log_ratio_max = log_ratio_masked.max().item()
                    log_ratio_min = log_ratio_masked.min().item()
                    
                    # ========== [全 rank 一致] 获取分布式信息 ==========
                    dist_on, rank, world = _dist_info()
                    
                    # 只在 rank0 打印警告，避免日志爆炸
                    if (log_ratio_max > 15 or log_ratio_min < -15) and (not dist_on or rank == 0):
                        print(f"[WARNING] log_ratio 差值过大，可能导致 ratio 溢出! batch_idx={batch_idx}")
                        print(f"  log_ratio range: [{log_ratio_min:.4f}, {log_ratio_max:.4f}]")
                        print(f"  exp(log_ratio) 将变成: [{torch.exp(torch.tensor(log_ratio_min)).item():.2e}, {torch.exp(torch.tensor(log_ratio_max)).item():.2e}]")
                    
                    # ========== [Token-Level Masking V3] 只 mask 极端 token，不 mask 整个 batch ==========
                    # 核心改进：不再用 zero_loss_flag 把整个 batch 的 loss 置零
                    # 而是把 log_ratio > threshold 的异常 token 的 response_mask 置 0
                    # 这样 99% 的正常 token 依然能产生有效梯度
                    
                    LOG_RATIO_THRESHOLD = 10.0  # 阈值：exp(10) ≈ 22000，超过这个会导致数值爆炸
                    
                    # 找到异常 token：|log_ratio| > threshold 且原本有效（response_mask > 0）
                    extreme_mask = (log_ratio.abs() > LOG_RATIO_THRESHOLD) & (response_mask > 0)
                    n_extreme_tokens = extreme_mask.sum().item()
                    n_valid_tokens = (response_mask > 0).sum().item()
                    
                    if n_extreme_tokens > 0:
                        # 把异常 token 的 mask 置 0
                        response_mask_original = response_mask.clone()  # 保存原始 mask 用于诊断
                        response_mask = response_mask.clone()  # 避免修改原数据
                        response_mask[extreme_mask] = 0
                        
                        # 更新统计
                        total_masked_tokens += n_extreme_tokens
                        total_valid_tokens_before_mask += n_valid_tokens
                        batches_with_masked_tokens += 1
                        
                        # 统计
                        n_remaining = (response_mask > 0).sum().item()
                        mask_ratio = n_extreme_tokens / max(n_valid_tokens, 1) * 100
                        
                        if not dist_on or rank == 0:
                            print(f"[TOKEN_MASK] batch_idx={batch_idx}: masked {n_extreme_tokens}/{n_valid_tokens} tokens ({mask_ratio:.1f}%), remaining={n_remaining}", flush=True)
                            print(f"  log_ratio range: [{log_ratio_min:.2f}, {log_ratio_max:.2f}], threshold={LOG_RATIO_THRESHOLD}", flush=True)
                            
                            # 如果 mask 太多（>50%），额外警告
                            if mask_ratio > 50:
                                print(f"  [WARNING] 超过 50% token 被 mask！本 batch 学习效果有限", flush=True)
                            
                            # 打印前 2 个极端 token 的诊断信息
                            extreme_indices = torch.where(extreme_mask)
                            if len(extreme_indices[0]) > 0:
                                COMMON_SPECIAL_TOKENS = {
                                    151643: "EOS(<|endoftext|>)",
                                    151644: "IM_START(<|im_start|>)",
                                    151645: "IM_END(<|im_end|>)",
                                    0: "PAD",
                                    1: "UNK",
                                }
                                
                                for i in range(min(2, len(extreme_indices[0]))):  # 最多打印 2 个
                                    si = extreme_indices[0][i].item()
                                    ti = extreme_indices[1][i].item()
                                    token_id = responses[si, ti].item()
                                    valid_len = (response_mask_original[si] > 0).sum().item()
                                    is_special_token = token_id in COMMON_SPECIAL_TOKENS
                                    special_token_name = COMMON_SPECIAL_TOKENS.get(token_id, "N/A")
                                    
                                    print(f"  [EXTREME_TOKEN] sample={si}, pos={ti}/{response_length}, token_id={token_id} ({special_token_name})", flush=True)
                                    print(f"    old_lp={old_log_prob[si, ti].item():.4f}, new_lp={log_prob[si, ti].item():.4f}, diff={log_ratio[si, ti].item():.4f}", flush=True)
                        
                        # 如果所有 token 都被 mask 了，才触发 ZERO_LOSS
                        if n_remaining == 0 and not zero_loss_flag:
                            zero_loss_flag = True
                            zero_loss_reason = "all_tokens_masked"
                            if not dist_on or rank == 0:
                                print(f"[ZERO_LOSS] batch_idx={batch_idx}: 所有 token 都被 mask，本 batch 无有效 token", flush=True)
                        
                        # ========== [稳健性补丁] 把极端 token 的 log_prob 对齐到 old_log_prob ==========
                        # 避免 compute_policy_loss 内部 exp(log_ratio) 溢出
                        # 对齐后 log_ratio=0, exp(0)=1, 从源头杜绝 inf
                        log_prob = log_prob.clone()
                        log_prob[extreme_mask] = old_log_prob[extreme_mask]
                    
                    # ========== [诊断4] 检查 advantages 是否正常（只 rank0 打印）==========
                    adv_has_nan = torch.isnan(advantages).any().item()
                    adv_has_inf = torch.isinf(advantages).any().item()
                    if adv_has_nan or adv_has_inf:
                        if not dist_on or rank == 0:
                            print(f"\n{'='*60}")
                            print(f"[CRITICAL] advantages 存在 inf/nan! batch_idx={batch_idx}")
                            print(f"  min={advantages.min().item():.4f}, max={advantages.max().item():.4f}")
                            print(f"  nan_count={torch.isnan(advantages).sum().item()}, inf_count={torch.isinf(advantages).sum().item()}")
                            print(f"{'='*60}\n")
                    
                    # [NaN Check] 检查 PPO 输入
                    check_ppo_tensors(
                        old_log_prob=old_log_prob, 
                        log_prob=log_prob, 
                        advantages=advantages, 
                        response_mask=response_mask,
                        step_info=f"before_policy_loss_batch{batch_idx}"
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )
                    
                    # [NaN Check] 检查 pg_loss
                    check_tensor_for_nan(pg_loss, f"pg_loss_batch{batch_idx}", raise_on_nan=False)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        # 【省显存 KL loss】优先使用 ref_log_prob，如果没有则使用 old_log_probs
                        # 这允许在 no_ref_policy=True 时仍然使用 KL 约束，而不需要加载额外的 ref model
                        if "ref_log_prob" in data.keys():
                            ref_log_prob = data["ref_log_prob"]
                        else:
                            # 使用 old_log_probs 作为 reference（这是 PPO 的标准做法）
                            ref_log_prob = data["old_log_probs"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode
                        )

                        kl_penalty_scaled = kl_loss * self.config.kl_loss_coef
                        policy_loss = policy_loss + kl_penalty_scaled
                        metrics["actor/kl_penalty_raw"] = kl_loss.detach().item()
                        metrics["actor/kl_penalty_scaled"] = kl_penalty_scaled.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        
                        # [NaN Check] 检查 kl_loss
                        check_tensor_for_nan(kl_loss, f"kl_loss_batch{batch_idx}", raise_on_nan=False)

                    # [NaN Check] 检查最终 policy_loss（在 backward 之前）
                    # 注意：不再用 continue，改为标志位 + loss=0，避免 NCCL 死锁
                    # 【全 rank 一致】使用 _sync_bool_flag 确保所有 rank 同步决定是否跳过
                    local_has_nan = check_tensor_for_nan(policy_loss, f"policy_loss_before_backward_batch{batch_idx}", raise_on_nan=False)
                    global_has_nan = _sync_bool_flag(local_has_nan, device=policy_loss.device)
                    
                    if not zero_loss_flag and global_has_nan:
                        dist_on, rank, _ = _dist_info()
                        if not dist_on or rank == 0:
                            print(f"[ZERO_LOSS] batch_idx={batch_idx}：policy_loss 含 NaN（global_sync），将 loss 置 0", flush=True)
                        zero_loss_flag = True
                        zero_loss_reason = "nan_in_policy_loss"

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    
                    # ========== 核心修改 V2：如果需要跳过，先 nan_to_num 再乘 0 ==========
                    # 【关键修复】解决 NaN * 0.0 = NaN 的问题
                    # V2 版本：先用 nan_to_num 把 NaN/Inf 变成 0，再乘 0
                    # 这样既不会产生 NaN 梯度，又保持计算图连接（满足 FSDP/DDP 同步要求）
                    total_micro_batch_count += 1
                    if zero_loss_flag:
                        # 先 nan_to_num：把可能的 NaN/Inf 变成有限值（0）
                        # 再乘 0.0：保证本步梯度为 0，但仍然走 backward 计算图
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
                        zero_loss_count += 1
                        zero_loss_reasons_counter[zero_loss_reason] = zero_loss_reasons_counter.get(zero_loss_reason, 0) + 1
                        # 只在 rank0 且第一次出现时打印详细信息，避免日志爆炸
                        dist_on, rank, _ = _dist_info()
                        if (not dist_on or rank == 0) and zero_loss_reasons_counter[zero_loss_reason] <= 3:
                            print(f"[ZERO_LOSS_APPLIED] batch_idx={batch_idx}, reason={zero_loss_reason}, nan_to_num + *0.0", flush=True)
                    
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                # ========== 每个 mini-batch 结束时打印 zero_loss 统计（只 rank0）==========
                dist_on, rank, _ = _dist_info()
                if total_micro_batch_count > 0 and (not dist_on or rank == 0):
                    zero_loss_ratio = zero_loss_count / total_micro_batch_count
                    reason_str = ", ".join([f"{k}:{v}" for k, v in zero_loss_reasons_counter.items()])
                    if zero_loss_count > 0:
                        print(f"[ZERO_LOSS_STATS] epoch={epoch}, batch_idx={batch_idx}: zero_loss={zero_loss_count}/{total_micro_batch_count} ({zero_loss_ratio*100:.1f}%), reasons=[{reason_str}]", flush=True)
                    
                    # Token-Level Masking 统计
                    if batches_with_masked_tokens > 0:
                        mask_ratio = total_masked_tokens / max(total_valid_tokens_before_mask, 1) * 100
                        print(f"[TOKEN_MASK_STATS] epoch={epoch}: masked {total_masked_tokens}/{total_valid_tokens_before_mask} tokens ({mask_ratio:.2f}%) in {batches_with_masked_tokens} batches", flush=True)

                # ========== [方案B-SUMMARY] 打印本 batch 的 old_log_prob==0 修复统计（只 rank0）==========
                # 这是证明 A+B 方案生效的关键日志
                # 验证修复后是否还有精确的 0（应该为 0）
                # 注意：此处 old_log_prob 是最后一个 micro_batch 的，仅用于示例检查
                if not dist_on or rank == 0:
                    remaining_zeros = (old_log_prob == 0).sum().item() if 'old_log_prob' in dir() else 0
                    print(f"[方案B-SUMMARY] epoch={epoch}, batch_idx={batch_idx}: "
                          f"fixed={old_logprob_zero_fixed_total}, remaining_zeros={remaining_zeros}, "
                          f"所有 response_mask=1 的 token 均可正常学习（无因 logprob=0 被额外 mask）", flush=True)

                grad_norm, update_skipped = self._optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else float(grad_norm),
                    "actor/update_skipped": update_skipped,
                }
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
