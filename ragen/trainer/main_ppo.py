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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
import re
import numpy as np

import ragen.utils.reward_score.countdown as countdown

from ragen.trainer.ppo.ray_trainer import RayPPOTrainer
from ragen.utils.env import get_train_val_env
from ragen.env import (
    SokobanEnv, 
    FrozenLakeEnv, 
    BanditEnv, 
    TwoArmedBanditEnv, 
    CountdownEnv,
    MedicalConsultationEnv,
    MedicalConsultationEnvWithPatientLLM,
    MedicalConsultationEnvWithPatientLLMandRM,
    MedicalConsultationEnvWithPatientLLMCategory,
)

ENV_CLASS_MAPPING = {
    'sokoban': SokobanEnv,
    'frozenlake': FrozenLakeEnv,
    'bandit': BanditEnv,
    'two_armed_bandit': TwoArmedBanditEnv,
    'countdown': CountdownEnv,
    'medical_consultation': MedicalConsultationEnv,
    'medical_consultation_patient_llm': MedicalConsultationEnvWithPatientLLM,
    'medical_consultation_patient_llm_rm': MedicalConsultationEnvWithPatientLLMandRM,
    'medical_consultation_patient_llm_category': MedicalConsultationEnvWithPatientLLMCategory,
    # 注意：当前使用 env_patient_llm_category.py（有 RAG 功能 + 完整监控指标）
    # 如需使用 simple 版本，需添加: 'medical_consultation_patient_llm_category_simple': MedicalConsultationEnvWithPatientLLMCategorySimple
}

def _select_rm_score_fn(data_source):
    if "countdown" in data_source:
        return countdown.compute_score
    elif "sokoban" in data_source or "frozenlake" in data_source or "bandit" in data_source:
        def judge_fn(*args, **kwargs):
            solution = kwargs['solution_str']
            # 1. reward based on the game:
            # find all patterns like reward: -0.1\n, add them together as reward.
            # 1. game reward
            pattern = r'reward: (-?\d+\.\d+)\ndone: (True|False)'
            matches = re.findall(pattern, solution)
            reward = sum(float(match[0]) for match in matches)
            # print(f"reward: {reward}")
            # 2. format reward, find "action is invalid", add -0.1 to reward
            pattern = r'Action is invalid. You stay in the same position.'
            matches = re.findall(pattern, solution)
            reward -= len(matches) * 1
            if reward > 15:
                print(f"[REWARD TOO MUCH]. solution: \n{solution}")
            return reward

            # # 2. reward based on success:
            # # if there is done: True, it is 1, otherwise 0.
            # if "done: True" in solution:
            #     reward = 1.0
            # else:
            #     reward = 0.0
            # return reward

        return judge_fn
    else:
        return lambda *args, **kwargs: 0.0 # the reward is implemented in the env class


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def _find_answer_end_positions(self, response_str: str, response_ids: torch.Tensor) -> list:
        """
        找到 response 中所有 </answer> 标签对应的 token 位置

        Args:
            response_str: 解码后的 response 字符串
            response_ids: response 的 token ids

        Returns:
            list of int: 每个 </answer> 对应的 token 位置索引
        """
        positions = []
        try:
            # 找到所有 </answer> 的位置
            end_tag = "</answer>"
            start = 0
            while True:
                pos = response_str.find(end_tag, start)
                if pos == -1:
                    break

                # 将字符位置转换为 token 位置
                # 通过编码前缀来确定 token 位置
                prefix = response_str[:pos + len(end_tag)]
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                token_pos = len(prefix_tokens) - 1  # 最后一个 token 的位置

                # 确保位置在有效范围内
                if 0 <= token_pos < len(response_ids):
                    positions.append(token_pos)

                start = pos + len(end_tag)
        except Exception:
            pass

        return positions

    def _find_all_tag_end_positions(self, response_str: str, response_ids: torch.Tensor) -> list:
        """
        找到 response 中所有 </answer> 和 </rag_query> 标签对应的 token 位置，按出现顺序排序。
        
        用于支持两步流程：问诊步骤 (</answer>) 和 RAG query 步骤 (</rag_query>)。
        
        Args:
            response_str: 解码后的 response 字符串
            response_ids: response 的 token ids

        Returns:
            list of tuple: [(tag_type, token_pos), ...] 按出现顺序排列
                tag_type: "answer" 或 "rag_query"
                token_pos: token 位置索引
        """
        positions = []
        try:
            # 同时查找 </answer> 和 </rag_query>
            # 注：降级反馈现在放在 <rag_query> 内部（[RAG_FAILED:原因]），不再使用独立 tag
            tags = [("answer", "</answer>"), ("rag_query", "</rag_query>")]
            
            for tag_type, end_tag in tags:
                start = 0
                while True:
                    pos = response_str.find(end_tag, start)
                    if pos == -1:
                        break

                    # 将字符位置转换为 token 位置
                    prefix = response_str[:pos + len(end_tag)]
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                    token_pos = len(prefix_tokens) - 1

                    # 确保位置在有效范围内
                    if 0 <= token_pos < len(response_ids):
                        # 存储 (字符位置, tag类型, token位置) 用于排序
                        positions.append((pos, tag_type, token_pos))

                    start = pos + len(end_tag)
            
            # 按字符位置排序，确保按出现顺序
            positions.sort(key=lambda x: x[0])
            # 返回 (tag_type, token_pos) 列表
            positions = [(p[1], p[2]) for p in positions]
            
        except Exception:
            pass

        return positions

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 检查 valid_response_length 是否为0，如果为0则跳过或给予惩罚
            if valid_response_length == 0:
                print(f"[WARNING] Sample {i} has empty response (valid_response_length=0), skipping reward assignment")
                score = -1.0  # 给予惩罚分数
                all_scores.append(score)
                continue

            if data_source in ENV_CLASS_MAPPING.keys():
                if 'reward' not in data_item.non_tensor_batch.keys():
                    # TODO: currently validate is not implemented
                    score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
                    # print("[WARNING] reward is not in data_item.non_tensor_batch.keys(), probably because validate is not implemented")
                else:
                    score = data_item.non_tensor_batch['reward']
                score = float(score)
                # print(f"reward: {score}")
                if score > 20:
                    print(f"[REWARD TOO MUCH]. solution: \n{sequences_str}")
                # score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            # ========== [新增] 多轮奖励分配逻辑 ==========
            # 如果存在 reward_list，将其分配到对应的动作边界位置
            reward_list = data_item.non_tensor_batch.get('reward_list', None)
            action_end_positions = data_item.non_tensor_batch.get('action_end_positions', None)

            # 【方案 A 优先】使用 action_end_positions（真实动作边界）分配 reward
            # 修复：对所有 action 数都生效（包括 len==1 的情况），并检查 token_pos >= 0
            if reward_list is not None and action_end_positions is not None and len(action_end_positions) > 0:
                # 【方案 A】使用精确的动作边界位置
                if i < 2:  # 只打印前2个样本
                    print(f"[PPO对齐-方案A] 样本{i}: 使用 action_end_positions")
                    print(f"[PPO对齐-方案A] 样本{i}: reward_list长度={len(reward_list)}, action_end_positions长度={len(action_end_positions)}")
                    print(f"[PPO对齐-方案A] 样本{i}: reward_list={[f'{r:.3f}' for r in reward_list]}")
                    print(f"[PPO对齐-方案A] 样本{i}: action_end_positions={action_end_positions}")
                
                # 分配奖励：按顺序匹配 reward_list 和 action_end_positions
                # 【修复】增加 token_pos >= 0 检查，避免负索引（如果某个 doctor 段 token 数为 0）
                for pos_idx, token_pos in enumerate(action_end_positions):
                    if pos_idx < len(reward_list) and 0 <= token_pos < valid_response_length:
                        reward_tensor[i, token_pos] = float(reward_list[pos_idx])
                
                # 如果 reward_list 比动作数多，剩余的奖励加到最后一个 token
                if len(reward_list) > len(action_end_positions):
                    remaining_rewards = reward_list[len(action_end_positions):]
                    remaining_sum = sum(float(r) for r in remaining_rewards)
                    reward_tensor[i, valid_response_length - 1] += remaining_sum
                    if i < 2:
                        print(f"[PPO对齐-方案A警告] 样本{i}: reward_list比action多 {len(remaining_rewards)} 个，剩余奖励={remaining_sum:.3f} 加到末尾")
                
                # 使用整段轨迹的总回报作为 score 记录（用于 debug 日志）
                score = sum(float(r) for r in reward_list)
            elif reward_list is not None and len(reward_list) > 0:
                # 【回退】使用 tag 扫描方式（当 action_end_positions 不可用时）
                response_str = self.tokenizer.decode(valid_response_ids)
                tag_positions = self._find_all_tag_end_positions(response_str, valid_response_ids)

                if i < 2:  # 只打印前2个样本
                    rag_query_count = sum(1 for t, _ in tag_positions if t == 'rag_query')
                    answer_count = sum(1 for t, _ in tag_positions if t == 'answer')
                    print(f"[PPO对齐-回退] 样本{i}: 使用 tag 扫描（action_end_positions 不可用）")
                    print(f"[PPO对齐-回退] 样本{i}: reward_list长度={len(reward_list)}, tag数={len(tag_positions)} (rag_query={rag_query_count}, answer={answer_count})")
                    print(f"[PPO对齐-回退] 样本{i}: reward_list={[f'{r:.3f}' for r in reward_list]}")
                    print(f"[PPO对齐-回退] 样本{i}: tag_positions={[(t, pos) for t, pos in tag_positions]}")

                # 分配奖励：按出现顺序匹配 reward_list
                for pos_idx, (tag_type, token_pos) in enumerate(tag_positions):
                    if pos_idx < len(reward_list) and 0 <= token_pos < valid_response_length:
                        reward_tensor[i, token_pos] = float(reward_list[pos_idx])

                # 如果 reward_list 比找到的位置多，剩余的奖励加到最后一个 token
                if len(reward_list) > len(tag_positions):
                    remaining_rewards = reward_list[len(tag_positions):]
                    remaining_sum = sum(float(r) for r in remaining_rewards)
                    reward_tensor[i, valid_response_length - 1] += remaining_sum
                    if i < 2:
                        print(f"[PPO对齐-回退警告] 样本{i}: reward_list比tag多 {len(remaining_rewards)} 个，剩余奖励={remaining_sum:.3f} 加到末尾")

                # 使用整段轨迹的总回报作为 score 记录（用于 debug 日志）
                score = sum(float(r) for r in reward_list)
            else:
                # 单一奖励（无 reward_list 或为空）：放在最后一个 token
                reward_tensor[i, valid_response_length - 1] = score

            all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        print(f"[DEBUG] all_scores: {all_scores}")
        print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        # 统计多轮奖励样本数
        multi_turn_count = sum(1 for i in range(len(data))
                               if data[i].non_tensor_batch.get('reward_list') is not None
                               and len(data[i].non_tensor_batch.get('reward_list', [])) > 1)
        print(f"[DEBUG] multi_turn_reward_samples: {multi_turn_count}/{len(data)}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from ragen.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from ragen.workers.env_llm_worker import EnvironmentLLMWorker

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    ref_pool_id = 'ref_pool'
    
    # Check if we should use separate ref pool (fewer ref workers)
    use_separate_ref_pool = config.trainer.get('use_separate_ref_pool', False)
    ref_gpus = config.trainer.get('ref_n_gpus', 1)  # Default to 1 GPU for ref
    
    if use_separate_ref_pool:
        # Separate resource pools: main pool for actor/rollout, small pool for ref
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            ref_pool_id: [ref_gpus] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: ref_pool_id,  # Use separate small pool for ref
        }
    else:
        # Original: all roles share the same pool
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

    if config.env.use_env_llm:
        role_worker_mapping[Role.EnvLLM] = ray.remote(EnvironmentLLMWorker)
        mapping[Role.EnvLLM] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_env, val_env = get_train_val_env(env_class, config)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            env=train_env,
                            val_env=val_env,
                            env_class=env_class)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()