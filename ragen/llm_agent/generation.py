import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from ragen.utils import set_seed
from ragen.utils.plot import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
LOG_ROOT = os.environ.get("MIND_LOG_ROOT", "logs/readable")



class AllPadBatchError(Exception):
    """当整个 batch 的 input_ids 全是 pad token 时抛出，用于优雅终止 rollout 循环。"""
    pass

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    logging: dict
    num_gpus: int
    no_think_rl: bool=False
    state_masking: bool=False
    start_state_marker: str="<start-state>"
    end_state_marker: str="<end-state>"
    use_env_llm: bool=False
    batch_size: int=1
    max_num_batched_tokens: int=7800  # 硬上限，默认值为 7800

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        env_class,
        config: GenerationConfig,
        logger: Tracking,
        env_llm_wg=None,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.env_class = env_class
        self.config = config
        self.logger = logger
        self.is_validation = is_validation
        self.env_llm_wg = env_llm_wg
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
            max_response_length=config.max_response_length
        ))
        
        # 轨迹构建时的截断上限（用于控制显存）
        self.MAX_SYSTEM_PROMPT_TOKENS = 300  # 后期/诊断系统 prompt 的最大 token 数
        self.MAX_PATIENT_RESPONSE_TOKENS = 128  # 病人回复在轨迹中的最大 token 数
        self.HARD_LIMIT_TOTAL_SEQLEN = config.max_num_batched_tokens  # 整个序列（prompts + responses）的硬上限，从配置中读取
        
        # 【关键】每轮医生输出的 token 硬截断上限，防止长垃圾文本顶爆 grad_norm
        # 这是防止 PPO 梯度爆炸的核心配置
        # 【二步流程】rag_query 和 answer 分开截断，各自最多 256 tokens
        self.MAX_RAG_QUERY_TOKENS = 256          # RAG query 步骤：最多 256 tokens
        self.MAX_ANSWER_TOKENS = 256             # 问诊 answer 步骤：最多 256 tokens
        self.MAX_INQUIRY_TOKENS_PER_TURN = 512   # 兜底：未知形态时最多 512 tokens
        self.MAX_DIAGNOSIS_TOKENS = 2048         # 诊断轮：最多 2048 tokens（保留完整 <think> + <answer>）
        self.MAX_ERROR_TOKENS = 128              # 错误轮：最多 128 tokens（足够学"别这么输出"）

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """将文本截断到指定的 token 数量上限。
        
        Args:
            text: 原始文本
            max_tokens: 最大 token 数
            
        Returns:
            截断后的文本
        """
        if not text or max_tokens <= 0:
            return text
        
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        
        # 截断并解码回文本
        truncated_ids = token_ids[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=False)
        return truncated_text

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    @staticmethod
    def _process_answer_tag(responses_str):
        """
        Process a list of response strings to keep only the first <answer></answer> tag pair
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing answer tags
            
        Returns:
            List[str]: Processed responses with only first answer tag pair preserved
        """
        def process_single_response(resp):
            # If no answer tags present, return original string
            if '<answer>' not in resp or '</answer>' not in resp:
                return resp
                
            # Find the first complete <answer> tag pair
            pattern = r'<answer>.*?</answer>'
            match = re.search(pattern, resp, re.DOTALL)
            
            if not match:
                return resp
                
            # Get the matched answer tag content
            answer_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            rest_of_string = resp[match.end():]
            cleaned_rest = re.sub(r'<answer>(.*?)</answer>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp[:match.start()] + answer_content + cleaned_rest
        
        # Process each response string
        return [process_single_response(resp) for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor, envs: List[Any]) -> torch.Tensor:
        """Process responses to remove 1. multiple answers or 2. reward hacking attempts."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # responses_str = [resp.split('</answer>')[0] + '</answer>' 
        #             if '</answer>' in resp else resp 
        #             for resp in responses_str]
        responses_str = self._process_answer_tag(responses_str)
        
        if self.config.state_masking:
            # Escape special characters in markers for regex
            start_marker = re.escape(self.config.start_state_marker)
            end_marker = re.escape(self.config.end_state_marker)
            hack_pattern = f'{start_marker}[\\s\\S]*?{end_marker}'
            
            hacked = [resp for resp in responses_str if re.search(hack_pattern, resp, re.DOTALL)]
            if hacked:
                print(f"[WARNING] HACKED RESPONSES: {hacked}")
            responses_str = [re.sub(hack_pattern, '', resp, re.DOTALL) for resp in responses_str]

        # NOTE: no_think_rl 功能已禁用，避免覆盖医生模型的原始输出格式
        # if self.config.no_think_rl:
        #     # if no_think_rl is enabled, only keep action in the str
        #     actions, _ = self.env_class.postprocess_predictions(envs, responses_str)
        #     responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
        #     print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str



    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        # 【关键修复】使用文本系统提示作为 fallback，而不是 pad_token
        # 目的：
        # 1. 避免 tokenize 后变成全 pad 序列，触发 AllPadBatchError
        # 2. 用系统提示引导模型"继续输出"，而不是学会"看到异常就停"
        # 3. 保持对话流程完整，让 PPO 能正常计算 reward
        fallback_obs = "<|im_start|>user\n[系统提示] 上轮响应处理异常，请根据已有信息继续问诊或给出诊断。\n<|im_end|>\n<|im_start|>assistant\n"
        
        for i, obs in enumerate(next_obs):
            if not obs or len(obs.strip()) == 0:
                print(f"[EMPTY_OBS_FIX] next_obs[{i}] is empty! Replacing with safe text fallback.", flush=True)
                next_obs[i] = fallback_obs
        
        if self.config.state_masking:
            start_marker = self.config.start_state_marker
            end_marker = self.config.end_state_marker
            
            # Create inner versions by adding 'inner_' prefix
            inner_start = f"<inner_{start_marker[1:]}"
            inner_end = f"<inner_{end_marker[1:]}"
            
            # Replace any existing markers with inner versions
            next_obs = [re.sub(re.escape(start_marker), inner_start, obs) for obs in next_obs]
            next_obs = [re.sub(re.escape(end_marker), inner_end, obs) for obs in next_obs]
            
            # Wrap with state markers
            def wrap_obs(obs):
                marker = "<|im_end|>\n<|im_start|>assistant\n"  # 移除 <think>
                b_marker = "<|im_start|>user\n"
                if marker in obs and b_marker in obs:
                    start_index = obs.index(b_marker) + len(b_marker)
                    end_index = obs.index(marker)
                    return f"{obs[:start_index]}{start_marker}{obs[start_index:end_index]}{end_marker}{obs[end_index:]}"
                else:
                    return f"{start_marker}{obs}{end_marker}"
            next_obs = [wrap_obs(obs) for obs in next_obs]
        
        # 使用和第一轮相同的 tokenizer 调用方式：add_special_tokens=False
        # 第一轮: tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False  # 和第一轮保持一致
        )['input_ids']
        
        # 【二次兜底】tokenize 后仍全 pad → 强制置为 eos_token_id
        # 这是最后一道防线，理论上前面的 fallback_obs 已经避免了这种情况
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            all_pad_mask = (next_obs_ids == pad_id).all(dim=1)
            if all_pad_mask.any():
                num_all_pad = all_pad_mask.sum().item()
                eos_id = self.tokenizer.eos_token_id or pad_id
                print(f"[ALL_PAD_FIX_POST_TOKENIZE] {num_all_pad} sequences are all-pad after tokenize; forcing eos_id={eos_id}", flush=True)
                # 把全 pad 的序列替换为 eos_id（至少有一个有效 token）
                next_obs_ids[all_pad_mask, 0] = eos_id
        
        # 如果 obs 过长被截断，在可读日志里打标记，方便和医生输入对齐排查
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            try:
                debug_log_path = os.path.join(LOG_ROOT, "med_dialogue_debug.log")
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("[WARNING] OBS TRUNCATED BEFORE FEEDING DOCTOR MODEL\n")
                    f.write(f"原始 token 长度: {next_obs_ids.shape[1]}, 截断到: {self.config.max_obs_length}\n")
                    # 打印第一条样本截断后的内容预览，方便对比第一轮 prompt
                    sample_ids = next_obs_ids[0, -self.config.max_obs_length:]
                    decoded_preview = self.tokenizer.decode(sample_ids, skip_special_tokens=False)
                    f.write("截断后第一个样本预览:\n")
                    f.write(decoded_preview + "\n")
                    f.write("=" * 80 + "\n")
            except Exception as e:
                print(f"[WARNING] Failed to log obs truncation: {e}")
            
            next_obs_ids = next_obs_ids[:, -self.config.max_obs_length:]
            
        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new observations only.

        对于当前医疗对话环境，我们已经在 env 内部通过 `_prepare_doctor_prompt`
        显式构造了完整的下一轮 prompt（包含系统提示 + 历史对话）。
        这里不再额外拼接上一轮的 `input_ids` 和 `cur_responses`，否则会让
        模型在 KV cache 里看到「自己上轮的长输出」，干扰我们在 prompt 中
        精心设计的格式与稳定性。

        因此，新的 rolling state 直接等于 `next_obs_ids`，只做长度截断。
        """
        # 直接使用 env 提供的 next_obs_ids 作为下一轮的输入
        new_input_ids = next_obs_ids
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor) -> Dict:
        """Update right side state.
        
        NOTE: 此函数已被弃用，保留仅为兼容性。
        新的轨迹构造在 _build_trajectory_from_history 中实现。
        """
        # 只累积医生输出，不再累积 next_obs_ids（因为它包含重复的历史）
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.tensor_fn.config.max_response_length, effective_len)
        
        return {'responses': responses[:, :max_len]}

    def _build_trajectory_from_history(self, envs: List[Any], 
                                       initial_input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从 env.conversation_history 重建完整轨迹的 responses 部分。
        
        轨迹格式（包含多阶段系统 prompt）：
        [S_early] + D1 + P1 + ... + D5 + P5 + [S_late] + D6 + P6 + ... + [S_final] + D_final
        
        其中：
        - S_early/S_late/S_final 是不同阶段的系统 prompt
        - D_t 是第 t 轮医生输出（从 env._actions 获取完整输出，包含 <think> 和 <answer>）
        - P_t 是第 t 轮病人回复（从 conversation_history 的 user 消息获取）
        
        返回：
        - responses: 完整轨迹的 token ids
        - action_mask: 标记哪些是医生输出（1=医生，0=系统prompt/病人）
        
        优化：
        - 限制处理的最大轮次数，避免过长轨迹导致卡顿
        - 带超时保护，超时返回空轨迹
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        import time
        
        # 超时时间（秒）：可通过环境变量配置
        timeout_seconds = int(os.environ.get("RAGEN_TRAJECTORY_BUILD_TIMEOUT", "60"))
        # 最大处理轮次：避免过长轨迹
        max_process_turns = int(os.environ.get("RAGEN_MAX_TRAJECTORY_TURNS", "20"))
        
        batch_size = len(envs)
        all_responses = []
        all_action_masks = []
        all_action_end_positions = []  # 【方案 A】记录每个 env 的动作结束位置列表
        
        start_time = time.time()
        
        for env_idx, env in enumerate(envs):
            # 超时检查：如果已经超过时间限制，跳过剩余环境
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"[WARNING] _build_trajectory_from_history 超时 ({elapsed:.1f}s > {timeout_seconds}s)，跳过剩余 {len(envs) - env_idx} 个环境")
                # 为剩余环境填充空轨迹
                for _ in range(len(envs) - env_idx):
                    all_responses.append(torch.tensor([], dtype=torch.long))
                    all_action_masks.append(torch.tensor([], dtype=torch.long))
                    all_action_end_positions.append([])  # 【方案 A】空动作边界
                break
            
            # 【简化版轨迹构建】只使用 _actions（已在 env 侧过滤，只包含正常轮次）
            # _actions 中的每一条都是有效的医生输出（包含 <think> 和 <answer>）
            doctor_outputs = getattr(env, '_actions', []) or []
            valid_turns = getattr(env, '_valid_turns', list(range(len(doctor_outputs))))
            
            # 【修复】先获取 max_turns，后面的 debug print 会用到
            max_turns = getattr(env, 'max_turns', 10)
            
            # 限制历史长度
            if len(doctor_outputs) > max_process_turns:
                doctor_outputs = doctor_outputs[-max_process_turns:]
                valid_turns = valid_turns[-max_process_turns:]
            
            # DEBUG: 打印第一个 env 的信息
            if env_idx == 0:
                print(f"[DEBUG _build_trajectory] env._actions count: {len(doctor_outputs)} (clean, no error turns)")
                print(f"[DEBUG _build_trajectory] max_turns={max_turns}, RAG_START_TURN={max(1, max_turns - 3)}")
                
                # 【新增】打印 _trajectory_patient_responses 长度，对比 _actions
                raw_patient_responses = getattr(env, '_trajectory_patient_responses', []) or []
                print(f"[DEBUG _build_trajectory] _trajectory_patient_responses count: {len(raw_patient_responses)}")
                
                # 【新增】统计 actions 中 RAG query 和 answer 的数量
                rag_count = sum(1 for a in doctor_outputs if a and '<rag_query>' in a and '</rag_query>' in a and '<answer>' not in a)
                ans_count = sum(1 for a in doctor_outputs if a and '<answer>' in a)
                rag_and_ans_count = sum(1 for a in doctor_outputs if a and '<rag_query>' in a and '<answer>' in a)
                print(f"[DEBUG _build_trajectory] Actions breakdown: RAG_only={rag_count}, ANS_only={ans_count-rag_and_ans_count}, RAG+ANS={rag_and_ans_count}")
                print(f"[DEBUG _build_trajectory] 期望患者回复数量={ans_count} (每个<answer>对应一个患者回复，RAG_query后无患者回复)")
                
                if len(raw_patient_responses) != ans_count:
                    print(f"[WARN _build_trajectory] 患者回复数量不匹配! 期望={ans_count}, 实际={len(raw_patient_responses)}")
                
                if doctor_outputs:
                    action_lens = [len(a) if a else 0 for a in doctor_outputs]
                    print(f"[DEBUG _build_trajectory] all actions lengths: {action_lens}")
                    print(f"[DEBUG _build_trajectory] valid_turns: {valid_turns}")
                    # 检查最后一条 action 是否是诊断轮
                    last_turn = valid_turns[-1] if valid_turns else -1
                    is_diagnosis_in_trajectory = (last_turn >= max_turns - 1)
                    print(f"[DEBUG _build_trajectory] last_turn={last_turn}, is_diagnosis_in_trajectory={is_diagnosis_in_trajectory}")
                    # 打印第一条和最后一条 action 的开头，用于确认内容
                    if doctor_outputs[0]:
                        print(f"[DEBUG _build_trajectory] first action[:100]: {doctor_outputs[0][:100]}...")
                    if len(doctor_outputs) > 1 and doctor_outputs[-1]:
                        print(f"[DEBUG _build_trajectory] last action[:100]: {doctor_outputs[-1][:100]}...")
            
            # 【关键修复】如果 _actions 为空（所有轮次都是错误），需要特殊处理
            # 问题：当使用 use_remove_padding=True 时，mask=0 的 token 会被完全移除，
            # 导致有效序列长度为 0，从而在 attention 计算时 reshape 失败：
            # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1, 128]
            # 
            # 解决方案：使用 EOS token 作为占位符，且 mask=1，确保至少有一个有效 token
            # 这样可以避免 tensor reshape 错误，同时单 token 对 PPO 更新影响很小
            if not doctor_outputs:
                print(f"[WARNING] Env {env_idx} has empty _actions (all turns were errors) - using EOS placeholder")
                eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
                min_placeholder_len = 10  # 最小占位长度，确保有足够 buffer
                # 使用 EOS token 填充，mask=1，确保 remove_padding 后仍有有效 token
                all_responses.append(torch.full((min_placeholder_len,), eos_token_id, dtype=torch.long))
                all_action_masks.append(torch.ones(min_placeholder_len, dtype=torch.long))  # mask=1，确保有效
                all_action_end_positions.append([])  # 空轨迹没有动作边界
                continue
            
            # 获取系统 prompts（现在只有两阶段：问诊 → 诊断）
            # （max_turns 已在前面获取）
            
            # PPO 轨迹专用的简化版系统 prompt（更短，节省 token）
            sys_inquiry_raw = getattr(env, 'PPO_SYSTEM_PROMPT_INQUIRY', None) or \
                              getattr(env, 'DOCTOR_SYSTEM_PROMPT_UNIFIED', '')
            sys_diagnosis_raw = getattr(env, 'PPO_SYSTEM_PROMPT_DIAGNOSIS', None) or \
                                getattr(env, 'FINAL_DIAGNOSIS_SYSTEM_PROMPT', '')
            
            # 截断系统 prompt 到 MAX_SYSTEM_PROMPT_TOKENS
            sys_inquiry = self._truncate_text_to_tokens(sys_inquiry_raw, self.MAX_SYSTEM_PROMPT_TOKENS)
            sys_diagnosis = self._truncate_text_to_tokens(sys_diagnosis_raw, self.MAX_SYSTEM_PROMPT_TOKENS)
            
            # 构建轨迹：使用 _actions（医生输出）+ conversation_history（病人回复）
            trajectory_parts = []  # [(text, is_doctor), ...]
            
            # 在轨迹最前面插入简化版问诊 prompt
            inquiry_prompt_text = f'<|im_start|>system\n{sys_inquiry}<|im_end|>\n<|im_start|>assistant\n'
            trajectory_parts.append((inquiry_prompt_text, False))
            
            # 【对齐】优先使用 env._trajectory_patient_responses（与 _actions 严格一一对应）
            # 只有当对齐列表不存在时才回退到从 conversation_history 提取
            patient_responses = getattr(env, '_trajectory_patient_responses', None)
            if not patient_responses:
                # fallback: 从 conversation_history 抽取（不保证对齐，但兼容旧代码）
                patient_responses = []
                history = getattr(env, 'conversation_history', []) or []
                for msg in history:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        patient_responses.append(msg.get('content', ''))
            
            # 【A方案】获取 _consume_patient_flags（与 _actions 严格一一对应）
            # True = 该 action 后有患者回复需要消费，False = 跳过
            consume_patient_flags = getattr(env, '_consume_patient_flags', None)
            if env_idx == 0:
                print(f"[DEBUG _build_trajectory] _consume_patient_flags 长度={len(consume_patient_flags) if consume_patient_flags else 'None'}")
            
            # 【错误/污染轮截断配置】
            # 错误标记前缀：以这些开头的输出被判定为错误轮
            ERROR_MARKERS = ('【格式错误】', '【问号数量错误】', '【代码污染】', '【代码污染-重试耗尽】')
            # 错误轮只保留前 N 个字符（大约对应 128 token，足够学习"别这么输出"）
            ERROR_OUTPUT_MAX_CHARS = 300
            
            # 【二步流程】RAG query 的简写 system prompt（只插一次）
            # 【修改】固定从第 4 轮问诊开始做 RAG（turn=3，0-based）
            RAG_QUERY_PROMPT_INSERTED = False
            SYS_RAG_QUERY = "【检索语句生成】请输出 <rag_query>检索状态文本</rag_query>（80-200字），不要输出 <answer>。"
            # 固定从 turn=3（第 4 轮问诊）开始需要 RAG
            # 例如 max_turns=5 时，turn 3 需要 RAG，turn 4 是诊断轮
            # 例如 max_turns=6 时，turn 3,4 需要 RAG，turn 5 是诊断轮
            RAG_START_TURN = 3  # 固定从 turn=3（第 4 轮问诊）开始
            
            # 【修复】基于清洗后的 output_to_use 计数 answer，避免患者回复索引错位
            # 原来用 doctor_outputs（原始串）计数，如果某轮输出无 <answer> 会少计，导致 patient_idx 错位
            cleaned_answer_count = 0  # 累计到当前为止有多少个 answer 输出（基于清洗后的 output_to_use）
            
            for action_idx, full_output in enumerate(doctor_outputs):
                # 判断当前轮次，决定是否插入诊断 prompt
                turn_num = valid_turns[action_idx] if action_idx < len(valid_turns) else action_idx
                is_final_turn = (turn_num >= max_turns - 1)
                
                # 在最后一轮之前插入诊断系统 prompt
                if is_final_turn and action_idx == len(doctor_outputs) - 1:
                    sys_text = f'<|im_start|>system\n{sys_diagnosis}<|im_end|>\n<|im_start|>assistant\n'
                    trajectory_parts.append((sys_text, False))
                    # 【DEBUG】打印诊断 prompt 插入信息
                    if env_idx == 0:
                        print(f"[DEBUG][PPO轨迹] 诊断 prompt 已插入: turn_num={turn_num}, max_turns={max_turns}, "
                              f"action_idx={action_idx}/{len(doctor_outputs)-1}")
                
                # 【错误轮截断】检测是否为错误/污染输出，如果是，只保留前 N 字符
                # 目的：减少长垃圾尾巴带来的 KL/梯度噪声，同时保留一点让模型学习"别这么输出"
                is_error_output = (full_output or '').startswith(ERROR_MARKERS)
                if is_error_output:
                    # 错误轮：截断到 ERROR_OUTPUT_MAX_CHARS 字符
                    output_to_use = (full_output or '')[:ERROR_OUTPUT_MAX_CHARS].strip()
                    if env_idx == 0 and action_idx == 0:
                        print(f"[DEBUG _build_trajectory] Truncated error output: {output_to_use[:80]}...")
                elif is_final_turn and action_idx == len(doctor_outputs) - 1:
                    # 诊断轮：保留完整输出（包括 <think> 和 <answer>）
                    output_to_use = (full_output or '').strip()
                else:
                    # 【二步流程】问诊轮：保留 <rag_query> 或 <answer>（不保留 <think>）
                    # 
                    # 二步流程中，env._actions 中的每个元素可能是：
                    # - RAG query 输出：包含 <rag_query>...</rag_query>
                    # - 问诊 answer 输出：包含 <answer>...</answer>
                    # 
                    # 这两种输出分别进入轨迹，各自独立计算 reward
                    temp_output = re.sub(r'<think>.*?</think>', '', full_output or '', flags=re.DOTALL).strip()
                    rag_match = re.search(r'<rag_query>(.*?)</rag_query>', temp_output, flags=re.DOTALL)
                    answer_match = re.search(r'<answer>(.*?)</answer>', temp_output, flags=re.DOTALL)

                    # 注：降级反馈现在放在 <rag_query> 内部（[RAG_FAILED:原因]），不再使用独立 tag
                    if rag_match and not answer_match:
                        # 纯 RAG query 输出（二步流程第一步）
                        # 【关键】只在最后两轮问诊时插入 RAG prompt（只插一次）
                        # turn_num >= RAG_START_TURN 才是需要 RAG 的轮次
                        if not RAG_QUERY_PROMPT_INSERTED and turn_num >= RAG_START_TURN:
                            rag_sys_text = f'<|im_start|>system\n{SYS_RAG_QUERY}<|im_end|>\n<|im_start|>assistant\n'
                            trajectory_parts.append((rag_sys_text, False))
                            RAG_QUERY_PROMPT_INSERTED = True
                        rag_text = rag_match.group(1).strip()
                        output_to_use = f'<rag_query>{rag_text}</rag_query>'
                    elif answer_match and not rag_match:
                        # 纯 answer 输出（二步流程第二步，或第一轮无需RAG）
                        output_to_use = f'<answer>{answer_match.group(1).strip()}</answer>'
                    elif rag_match and answer_match:
                        # 兼容旧版单步流程：同时包含 rag_query 和 answer
                        rag_text = rag_match.group(1).strip()
                        ans_text = answer_match.group(1).strip()
                        output_to_use = f'<rag_query>{rag_text}</rag_query>\n<answer>{ans_text}</answer>'
                    else:
                        # 回退：如果有 </answer>，截断到第一个 </answer>
                        if '</answer>' in temp_output:
                            temp_output = temp_output.split('</answer>', 1)[0] + '</answer>'
                        output_to_use = temp_output
                
                # 确保以 <|im_end|> 结尾
                if output_to_use and not output_to_use.endswith('<|im_end|>'):
                    output_to_use += '<|im_end|>'
                
                # 【关键】Token 级硬截断：限制每轮 doctor 输出进入 PPO 的 token 数
                # 目的：防止长垃圾文本（胡言乱语/重复输出）导致 grad_norm 爆炸
                if output_to_use:
                    if is_error_output:
                        # 错误轮：严格限制 token 数，足够学习"别这么输出"
                        output_to_use = self._truncate_text_to_tokens(output_to_use, self.MAX_ERROR_TOKENS)
                    elif is_final_turn and action_idx == len(doctor_outputs) - 1:
                        # 诊断轮：稍微宽松，保留思考过程和诊断答案
                        output_to_use = self._truncate_text_to_tokens(output_to_use, self.MAX_DIAGNOSIS_TOKENS)
                    else:
                        # 【二步流程】问诊轮：按类型分别截断 rag_query 和 answer
                        is_rag_only = ('<rag_query>' in output_to_use and '</rag_query>' in output_to_use
                                       and '<answer>' not in output_to_use)
                        is_answer_only = ('<answer>' in output_to_use and '</answer>' in output_to_use
                                          and '<rag_query>' not in output_to_use)
                        
                        if is_rag_only:
                            # RAG query 步骤：最多 256 tokens
                            output_to_use = self._truncate_text_to_tokens(output_to_use, self.MAX_RAG_QUERY_TOKENS)
                        elif is_answer_only:
                            # 问诊 answer 步骤：最多 256 tokens
                            output_to_use = self._truncate_text_to_tokens(output_to_use, self.MAX_ANSWER_TOKENS)
                        else:
                            # 兜底：未知形态（或旧版合并输出），使用较大上限
                            output_to_use = self._truncate_text_to_tokens(output_to_use, self.MAX_INQUIRY_TOKENS_PER_TURN)
                    
                    # 截断后确保仍以 <|im_end|> 结尾
                    if not output_to_use.endswith('<|im_end|>'):
                        output_to_use += '<|im_end|>'
                    
                    trajectory_parts.append((output_to_use, True))  # True = 医生输出
                
                # 非最后一轮，使用病人的真实回复（截断到 MAX_PATIENT_RESPONSE_TOKENS）
                # 【二步流程】RAG query 输出后不需要插入患者回复，只有 answer 输出后才需要
                is_rag_query_output = '<rag_query>' in output_to_use and '</rag_query>' in output_to_use and '<answer>' not in output_to_use
                
                # 【二步流程】如果是 RAG query 输出，后面需要插入新的 assistant 起始标记
                # 使 rag_query 和 answer 成为两条独立的 assistant action
                if is_rag_query_output and action_idx < len(doctor_outputs) - 1:
                    # RAG query 后插入 assistant 起始标记（非训练片段）
                    trajectory_parts.append(('\n<|im_start|>assistant\n', False))
                
                # 【修复】严格判定有效 answer：只有单一、完整、非嵌套的 <answer>...</answer> 才计数
                # 避免 <answer><answer>... 这种嵌套格式错误导致的患者回复错位
                def is_valid_single_answer(text):
                    """检查是否是有效的单一 answer 标签（非嵌套、完整闭合）"""
                    if not text:
                        return False
                    # 计算 <answer> 和 </answer> 的数量
                    open_count = text.count('<answer>')
                    close_count = text.count('</answer>')
                    # 必须是单一配对：恰好 1 个 <answer> 和 1 个 </answer>
                    if open_count != 1 or close_count != 1:
                        return False
                    # 检查是否正确闭合（<answer> 在 </answer> 之前）
                    open_pos = text.find('<answer>')
                    close_pos = text.find('</answer>')
                    return open_pos < close_pos
                
                is_valid_answer = is_valid_single_answer(output_to_use)
                
                # 【清洗嵌套标签】如果有嵌套 <answer>，清洗后再包装
                if '<answer>' in output_to_use and not is_valid_answer:
                    # 尝试提取最内层的 answer 内容，去掉多余的标签
                    # 先去掉所有 <answer> 和 </answer>，提取纯文本
                    inner_text = re.sub(r'</?answer>', '', output_to_use)
                    # 重新包装成单一 answer（保留 <|im_end|> 等控制标记）
                    if '<|im_end|>' in inner_text:
                        inner_text = inner_text.replace('<|im_end|>', '')
                    output_to_use = f'<answer>{inner_text.strip()}</answer><|im_end|>'
                    is_valid_answer = True  # 清洗后视为有效
                    if env_idx == 0:
                        print(f"[DEBUG][清洗嵌套answer] action_idx={action_idx}, 原输出有嵌套标签，已清洗")
                
                if is_valid_answer:
                    cleaned_answer_count += 1
                
                if action_idx < len(doctor_outputs) - 1 and not is_rag_query_output:
                    # 【A方案】使用 _consume_patient_flags 精确判断是否需要患者回复
                    should_consume_patient = True  # 默认需要
                    if consume_patient_flags is not None and action_idx < len(consume_patient_flags):
                        should_consume_patient = consume_patient_flags[action_idx]
                    
                    if not should_consume_patient:
                        # env 显式标记了该 action 不需要患者回复（格式错误/诊断轮/问号数量错误）
                        # 不插入患者回复，直接跳过
                        if env_idx == 0:
                            print(f"[DEBUG][A方案-跳过患者回复] action_idx={action_idx}, consume_patient_flag=False")
                    else:
                        # 获取对应的病人回复
                        # 【二步流程】需要根据实际的问诊轮次获取患者回复，而不是 action_idx
                        # 因为每个问诊轮有两个 action（rag_query + answer），但只有一个患者回复
                        # 【修复】使用基于清洗后输出的 cleaned_answer_count，而不是原始 doctor_outputs
                        patient_idx = cleaned_answer_count - 1  # 患者回复索引（-1 因为当前 answer 对应当前患者回复）
                        
                        if 0 <= patient_idx < len(patient_responses):
                            patient_content = patient_responses[patient_idx]
                            # 截断病人回复（避免过长）
                            truncated_patient = self._truncate_text_to_tokens(patient_content, self.MAX_PATIENT_RESPONSE_TOKENS)
                            # 【调试日志】成功匹配患者回复
                            if env_idx == 0:
                                print(f"[DEBUG][患者回复匹配成功] action_idx={action_idx}/{len(doctor_outputs)-1}, "
                                      f"cleaned_answer_count={cleaned_answer_count}, patient_idx={patient_idx}, "
                                      f"patient_len={len(patient_responses)}, "
                                      f"action[:60]={output_to_use[:60] if output_to_use else 'None'}..., "
                                      f"patient[:40]={patient_content[:40] if patient_content else 'None'}...")
                        else:
                            # 【改进】使用更明确的占位提示语，帮助模型理解上下文缺失原因
                            # 可能原因：1) 格式错误/污染导致患者未被调用 2) 诊断轮跳过患者 3) 系统重试
                            truncated_patient = "[系统提示] 本轮患者回复缺失（可能因格式错误或进入诊断流程），请根据已有信息继续。"
                            # 【调试日志】患者回复不足 - 打印详细信息帮助排查
                            print(f"[WARN][患者回复缺失] env={env_idx}, action_idx={action_idx}/{len(doctor_outputs)-1}, "
                                  f"cleaned_answer_count={cleaned_answer_count}, patient_idx={patient_idx}, "
                                  f"patient_len={len(patient_responses)}, is_rag_query={is_rag_query_output}, "
                                  f"turn_num={turn_num}, max_turns={max_turns}, consume_flag=True(fallback)")
                            # 打印所有 actions 的类型概览
                            if env_idx == 0:
                                action_types = []
                                for i, act in enumerate(doctor_outputs):
                                    has_rag = '<rag_query>' in (act or '') and '</rag_query>' in (act or '')
                                    has_ans = '<answer>' in (act or '')
                                    if has_rag and not has_ans:
                                        action_types.append(f"[{i}:RAG]")
                                    elif has_ans and not has_rag:
                                        action_types.append(f"[{i}:ANS]")
                                    elif has_rag and has_ans:
                                        action_types.append(f"[{i}:RAG+ANS]")
                                    else:
                                        action_types.append(f"[{i}:OTHER]")
                                print(f"[DEBUG][Actions类型概览] {' '.join(action_types)}")
                                print(f"[DEBUG][当前action内容] {output_to_use[:100] if output_to_use else 'None'}...")
                        patient_text = f'\n<|im_start|>user\n{truncated_patient}<|im_end|>\n<|im_start|>assistant\n'
                        trajectory_parts.append((patient_text, False))
            
            # 优化：批量 tokenize 而不是逐个调用
            # 先收集所有文本，再一次性 tokenize
            texts_to_tokenize = [text for text, _ in trajectory_parts if text]
            is_doctor_flags = [is_doctor for text, is_doctor in trajectory_parts if text]
            
            if texts_to_tokenize:
                # 批量 tokenize（不返回 tensor，因为长度不一致）
                tokenized = self.tokenizer(
                    texts_to_tokenize,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False
                )
                
                # 处理每个 tokenized 结果
                part_ids_list = []
                part_masks_list = []
                
                for idx, is_doctor in enumerate(is_doctor_flags):
                    if 'input_ids' in tokenized:
                        if isinstance(tokenized['input_ids'], list):
                            part_ids = torch.tensor(tokenized['input_ids'][idx], dtype=torch.long)
                        else:
                            part_ids = tokenized['input_ids'][idx]
                    else:
                        continue
                    
                    part_ids_list.append(part_ids)
                    
                    if not is_doctor:
                        # 非医生输出（系统 prompt、病人回复）：全部 mask=0
                        part_mask = torch.zeros_like(part_ids)
                    else:
                        # 【精细化 mask】医生输出：标记 <answer>...</answer> 和 <rag_query>...</rag_query> 内部的 token
                        # 
                        # 【二步流程】需要同时覆盖 <rag_query> 和 <answer> 内容
                        # 使用"文本级别+逐字符对齐"的方式来确定 mask
                        # 原因：tokenizer 会把 <answer>" 编码成 [<, answer, >"]，而不是 [<, answer, >, "]
                        #       导致 token 级别的匹配失败
                        # 方案：先在文本中找到标签的位置，然后逐个 token 解码并对比偏移
                        
                        text = texts_to_tokenize[idx]
                        part_mask = torch.zeros_like(part_ids)
                        
                        # 在文本中找到所有 <answer>...</answer> 和 <rag_query>...</rag_query> 的字符区间
                        content_char_ranges = []  # list of (start_char, end_char)，标记内容区间（不含标签本身）
                        
                        # 查找 <answer>...</answer>
                        search_start = 0
                        while True:
                            start_tag_pos = text.find('<answer>', search_start)
                            if start_tag_pos == -1:
                                break
                            content_start = start_tag_pos + len('<answer>')
                            end_tag_pos = text.find('</answer>', content_start)
                            if end_tag_pos == -1:
                                search_start = content_start
                                continue
                            content_char_ranges.append((content_start, end_tag_pos))
                            search_start = end_tag_pos + len('</answer>')
                        
                        # 查找 <rag_query>...</rag_query>
                        search_start = 0
                        while True:
                            start_tag_pos = text.find('<rag_query>', search_start)
                            if start_tag_pos == -1:
                                break
                            content_start = start_tag_pos + len('<rag_query>')
                            end_tag_pos = text.find('</rag_query>', content_start)
                            if end_tag_pos == -1:
                                search_start = content_start
                                continue
                            content_char_ranges.append((content_start, end_tag_pos))
                            search_start = end_tag_pos + len('</rag_query>')
                        
                        # 注：降级反馈现在放在 <rag_query> 内部（[RAG_FAILED:原因]），不再使用独立 tag
                        # 因此不需要额外查找 <rag_query_failed>，其内容会被 <rag_query> 的区间覆盖
                        
                        if content_char_ranges:
                            # 逐个 token 检查，看它解码后的文本是否落在 <answer> 或 <rag_query> 内容区间
                            # 使用 offset_mapping 来精确判断
                            # 重新 tokenize 单个文本以获取 offset_mapping
                            single_tokenized = self.tokenizer(
                                text,
                                add_special_tokens=False,
                                padding=False,
                                truncation=False,
                                return_offsets_mapping=True
                            )
                            offset_mapping = single_tokenized.get('offset_mapping', None)
                            
                            # 【安全阀】一致性检查：single tokenize 的长度/ids 必须与 batch tokenize 完全一致
                            # 不一致时保守处理，把该段 mask 置 0，避免错位污染训练
                            single_ids = single_tokenized.get("input_ids", None)
                            tokenize_mismatch = False
                            if single_ids is None or len(single_ids) != len(part_ids):
                                if env_idx == 0:  # 只打印第一个 env 的警告
                                    print(
                                        f"[WARNING] offset_mapping 与 batch tokenize 长度不一致："
                                        f"single={len(single_ids) if single_ids is not None else 'None'} vs batch={len(part_ids)}，该段 mask 置 0",
                                        flush=True
                                    )
                                tokenize_mismatch = True
                            else:
                                # 进一步检查 token ids 是否完全一致
                                if isinstance(part_ids, torch.Tensor):
                                    batch_ids = part_ids.tolist()
                                else:
                                    batch_ids = list(part_ids)
                                if list(single_ids) != batch_ids:
                                    if env_idx == 0:
                                        print("[WARNING] single tokenize 的 input_ids 与 batch tokenize 不一致，该段 mask 置 0", flush=True)
                                    tokenize_mismatch = True
                            
                            if tokenize_mismatch:
                                # 保守处理：保持 part_mask 全 0，不参与训练
                                part_masks_list.append(part_mask)
                                continue
                            
                            if offset_mapping:
                                for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                                    if token_idx >= len(part_mask):
                                        break
                                    # 检查这个 token 的字符区间是否完全落在某个 <answer> 或 <rag_query> 内容区间内
                                    # 并且不包含标签本身
                                    for content_start, content_end in content_char_ranges:
                                        # token 必须完全在内容区间内
                                        if char_start >= content_start and char_end <= content_end:
                                            # 额外检查：排除 token 解码后包含 '<' 或 '>' 的情况
                                            # 这是为了排除标签被 merge 的情况
                                            token_text = text[char_start:char_end]
                                            if '<' not in token_text and '>' not in token_text:
                                                part_mask[token_idx] = 1
                                            break
                            else:
                                # 如果没有 offset_mapping（不应该发生），回退到老方法
                                print(f"[WARNING] tokenizer 不支持 offset_mapping，回退到旧方法", flush=True)
                                # 简单回退：mask=1 for all tokens in doctor output
                                part_mask = torch.ones_like(part_ids)
                        
                        # 如果没有找到任何 <answer> 标签（可能是诊断轮只有 <think> 内容），
                        # 保持全 0 mask，不参与训练
                        if part_mask.sum() == 0 and len(part_ids) > 0 and '<answer>' not in text:
                            # 这是预期的情况：纯 <think> 内容不参与 PPO loss
                            pass
                    
                    part_masks_list.append(part_mask)
            
            # 拼接所有部分，并记录每个医生动作的结束 token 位置
            # 【方案 A 关键】记录动作边界，用于 reward 分配对齐
            action_end_positions = []  # 每个医生动作结束的 token 位置（相对于 trajectory 开头）
            current_token_pos = 0  # 当前累积的 token 位置
            
            if part_ids_list:
                for part_idx, (part_ids, is_doctor_flag) in enumerate(zip(part_ids_list, is_doctor_flags)):
                    part_len = len(part_ids)
                    current_token_pos += part_len
                    
                    if is_doctor_flag:
                        # 记录这个医生动作的结束位置（即最后一个 token 的索引）
                        action_end_positions.append(current_token_pos - 1)
                
                trajectory_ids = torch.cat(part_ids_list, dim=0)
                trajectory_mask = torch.cat(part_masks_list, dim=0)
            else:
                trajectory_ids = torch.tensor([], dtype=torch.long)
                trajectory_mask = torch.tensor([], dtype=torch.long)
            
            # DEBUG: 打印第一个 env 的轨迹统计
            if env_idx == 0:
                print(f"[DEBUG _build_trajectory] trajectory_parts count: {len(trajectory_parts)}")
                print(f"[DEBUG _build_trajectory] final trajectory_ids length: {len(trajectory_ids)} tokens")
                mask_sum = trajectory_mask.sum().item() if len(trajectory_mask) > 0 else 0
                mask_ratio = mask_sum / len(trajectory_ids) * 100 if len(trajectory_ids) > 0 else 0
                print(f"[DEBUG _build_trajectory] trajectory_mask sum (<answer> 内容 tokens): {mask_sum} ({mask_ratio:.1f}%)")
                print(f"[DEBUG _build_trajectory] 说明：只有 <answer>...</answer> 内部的 token 参与 PPO loss 计算")
                print(f"[DEBUG _build_trajectory] action_end_positions: {action_end_positions}")
            
            all_responses.append(trajectory_ids)
            all_action_masks.append(trajectory_mask)
            all_action_end_positions.append(action_end_positions)
        
        # Pad 到相同长度
        if not all_responses or all(len(r) == 0 for r in all_responses):
            # 如果没有轨迹，返回空张量
            empty_responses = torch.empty(batch_size, 0, dtype=torch.long, 
                                         device=initial_input_ids.device)
            empty_masks = torch.empty(batch_size, 0, dtype=torch.long,
                                     device=initial_input_ids.device)
            return {
                'responses': empty_responses, 
                'action_mask': empty_masks,
                'action_end_positions': [[] for _ in range(batch_size)]  # 【方案 A】
            }
        
        # 找到最大长度
        max_seq_len = max(len(r) for r in all_responses)
        
        # 计算 prompts 的长度：使用 max_start_length（因为 _compose_final_output 中只取最后 max_start_length 个 tokens）
        # 而不是 initial_input_ids.shape[1]（可能是 pad 后的长度）
        prompt_len = self.config.max_start_length
        hard_limit_for_responses = self.HARD_LIMIT_TOTAL_SEQLEN - prompt_len
        
        # 截断到 min(max_response_length, hard_limit_for_responses)
        max_len = min(self.config.max_response_length, hard_limit_for_responses)
        if max_seq_len > max_len:
            print(f"[WARNING] Trajectory too long ({max_seq_len} tokens), truncating to {max_len} (prompt_len={prompt_len}, hard_limit={self.HARD_LIMIT_TOTAL_SEQLEN})")
            max_seq_len = max_len
        
        # Pad 所有序列
        pad_token_id = self.tokenizer.pad_token_id
        padded_responses = []
        padded_masks = []
        adjusted_action_end_positions = []  # 【方案 A】截断后调整的动作边界位置
        
        for env_idx, (resp_ids, mask, action_ends) in enumerate(zip(all_responses, all_action_masks, all_action_end_positions)):
            # 【从左边截断】保留最近的对话（右边），丢弃最早的对话（左边）
            # 这样可以确保诊断轮次不被丢失
            truncate_len = 0
            if len(resp_ids) > max_seq_len:
                truncate_len = len(resp_ids) - max_seq_len
                resp_ids = resp_ids[truncate_len:]  # 从左边开始截断
                mask = mask[truncate_len:]
            
            # 【方案 A】调整动作边界位置（减去截断长度，过滤掉被截断的）
            adjusted_ends = []
            for pos in action_ends:
                new_pos = pos - truncate_len
                if new_pos >= 0 and new_pos < len(resp_ids):  # 位置仍在有效范围内
                    adjusted_ends.append(new_pos)
            adjusted_action_end_positions.append(adjusted_ends)
            
            # Pad（右边补齐）
            pad_len = max_seq_len - len(resp_ids)
            if pad_len > 0:
                resp_ids = torch.cat([resp_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            
            padded_responses.append(resp_ids)
            padded_masks.append(mask)
        
        responses_tensor = torch.stack(padded_responses, dim=0).to(initial_input_ids.device)
        action_mask_tensor = torch.stack(padded_masks, dim=0).to(initial_input_ids.device)
        
        elapsed_total = time.time() - start_time
        if elapsed_total > 5:  # 只在超过 5 秒时打印警告
            print(f"[PERF] _build_trajectory_from_history took {elapsed_total:.1f}s for {batch_size} envs")
        
        return {
            'responses': responses_tensor, 
            'action_mask': action_mask_tensor,
            'action_end_positions': adjusted_action_end_positions  # 【方案 A】
        }


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        # 【防护】检查是否有全 pad 的序列，避免 vllm 的 _pre_process_inputs 崩溃
        input_ids = active_batch.batch.get('input_ids')
        if input_ids is not None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            # 检查每个序列是否全是 pad
            all_pad_mask = (input_ids == pad_token_id).all(dim=1)
            if all_pad_mask.any():
                num_all_pad = all_pad_mask.sum().item()
                bad_indices = torch.where(all_pad_mask)[0].tolist()
                print(f"[ALL_PAD_WARNING] Found {num_all_pad} sequences that are all pad tokens", flush=True)
                print(f"[ALL_PAD_WARNING] Bad sequence indices in active_batch: {bad_indices[:10]}{'...' if len(bad_indices) > 10 else ''}", flush=True)
                # 打印详细诊断信息
                for bad_idx in bad_indices[:3]:  # 只打印前 3 个
                    seq = input_ids[bad_idx]
                    seq_len = seq.shape[0]
                    unique_tokens = torch.unique(seq).tolist()
                    print(f"[ALL_PAD_DIAG] idx={bad_idx}: seq_len={seq_len}, unique_tokens={unique_tokens[:20]}", flush=True)
                    # 尝试 decode 看看内容
                    try:
                        decoded = self.tokenizer.decode(seq[:50], skip_special_tokens=False)
                        print(f"[ALL_PAD_DIAG] idx={bad_idx}: decoded_first_50_tokens='{decoded[:200]}'", flush=True)
                    except Exception as e:
                        print(f"[ALL_PAD_DIAG] idx={bad_idx}: decode failed: {e}", flush=True)
                # 找到一个非全 pad 的序列作为替换
                valid_indices = torch.where(~all_pad_mask)[0]
                if len(valid_indices) > 0:
                    replacement_idx = valid_indices[0].item()
                    print(f"[ALL_PAD_FIX] Replacing with sequence at idx={replacement_idx}", flush=True)
                    for k, v in active_batch.batch.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == input_ids.shape[0]:
                            v[all_pad_mask] = v[replacement_idx:replacement_idx+1].expand(num_all_pad, *v.shape[1:])
                else:
                    # 所有序列都是全 pad，这是严重问题，抛出异常让上层处理
                    print(f"[ALL_PAD_ERROR] All {input_ids.shape[0]} sequences are all pad tokens! This should not happen.", flush=True)
                    print(f"[ALL_PAD_ERROR] batch keys: {list(active_batch.batch.keys())}", flush=True)
                    print(f"[ALL_PAD_ERROR] input_ids shape: {input_ids.shape}", flush=True)
                    # 抛出异常，让 run_llm_loop 捕获并优雅终止
                    raise AllPadBatchError(
                        f"All {input_ids.shape[0]} sequences in active batch are all pad tokens. "
                        "This indicates a critical issue in env/prompt construction."
                    )

        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # 【修复】确保 meta_info 被传递到 padded_active_batch
        if hasattr(active_batch, 'meta_info') and active_batch.meta_info:
            padded_active_batch.meta_info = active_batch.meta_info.copy()
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def run_llm_loop(self, gen_batch, envs: List[Any],
                    initial_input_ids: torch.Tensor,
                    output_dir: str,
                    global_steps: int) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # Setup visualization and Initialize states
        trajectory = self._setup_visualization()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        # 【根修复】用 env.finished() 初始化 active_mask，而不是盲目用 ones
        # 这样可以避免一开始就有 finished 的 env 被当成 active
        active_mask = torch.tensor([not env.finished() for env in envs], dtype=torch.bool)
        initial_finished_count = (~active_mask).sum().item()
        if initial_finished_count > 0:
            print(f"[INIT_ACTIVE_MASK] {initial_finished_count}/{len(envs)} envs are already finished at start!", flush=True)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # 输出第一轮医生模型的输入 prompt 到日志
        try:
            from datetime import datetime
            log_file = "logs/readable/med_dialogue_debug.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"[{ts}] [第1轮] 医生模型初始输入 Prompt (共 {len(envs)} 个环境)\n")
                f.write("="*80 + "\n")
                
                # 解码前几个环境的初始 prompt
                for idx, env in enumerate(envs[:3]):  # 只打印前3个，避免日志过长
                    try:
                        input_ids = gen_batch.batch['input_ids'][idx]
                        # 去掉 padding
                        non_pad_mask = input_ids != self.tokenizer.pad_token_id
                        valid_ids = input_ids[non_pad_mask]
                        prompt_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                        
                        dlg_id = getattr(env, 'dialogue_id', f'env-{idx}')
                        f.write(f"\n[医生模型][{dlg_id}] 第1轮 输入 Prompt (tokens: {len(valid_ids)}):\n")
                        f.write("-"*80 + "\n")
                        f.write(prompt_text + "\n")
                        f.write("-"*80 + "\n")
                        
                        # 同时保存到 env 的 _last_doctor_prompt 属性
                        env._last_doctor_prompt = prompt_text
                    except Exception as e:
                        f.write(f"[ERROR] 解码第 {idx} 个环境的 prompt 失败: {e}\n")
                
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"[WARNING] 输出第一轮 prompt 失败: {e}")

        # Main generation loop
        # 【修改】循环步数 = max_turns + 3，给诊断轮留 3 次补考机会
        # 第 1-9 轮：问诊轮（current_turn 0-8），格式错误也推进 current_turn
        # 第 10-13 轮：诊断轮（current_turn >= 9），给 3 次补考机会
        max_loop_steps = self.config.max_turns + 3
        for step in range(max_loop_steps):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
            
            # DEBUG: 打印实际传给模型的 input_ids 内容
            try:
                debug_log_path = os.path.join(LOG_ROOT, "med_dialogue_debug.log")
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[DEBUG][run_llm_loop] Step {step} - 实际传给模型的 input_ids\n")
                    f.write(f"{'='*80}\n")
                    # 只打印第一个样本
                    input_ids_0 = rollings.batch['input_ids'][0]
                    # 去掉 padding
                    non_pad_mask = input_ids_0 != self.tokenizer.pad_token_id
                    valid_ids = input_ids_0[non_pad_mask]
                    decoded_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                    f.write(f"Token 数量: {len(valid_ids)}\n")
                    f.write(f"解码内容:\n{decoded_text}\n")
                    f.write(f"{'='*80}\n")
            except Exception as e:
                print(f"[WARNING] Debug logging failed: {e}")
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)

            # 【根修复】在构建 active_batch 之前，用 env.finished() 二次确认
            # 把已 finished 的 env 从 active_mask 中强制剔除，避免全 pad 序列被喂给 vLLM
            try:
                env_finished_mask = torch.tensor([env.finished() for env in envs], dtype=torch.bool)
                # 检测 active_mask 与 env.finished() 不一致的情况
                inconsistent = active_mask & env_finished_mask  # active=True 但 finished=True
                if inconsistent.any():
                    inconsistent_count = inconsistent.sum().item()
                    inconsistent_indices = torch.where(inconsistent)[0].tolist()[:5]
                    print(f"[ACTIVE_MASK_FIX] step={step}: {inconsistent_count} envs are marked active but already finished!", flush=True)
                    print(f"[ACTIVE_MASK_FIX] Fixing inconsistent env indices: {inconsistent_indices}...", flush=True)
                    # 强制修复：把 finished env 从 active_mask 中剔除
                    active_mask = active_mask & ~env_finished_mask
                    print(f"[ACTIVE_MASK_FIX] After fix: active_count={active_mask.sum().item()}", flush=True)
            except Exception as e:
                print(f"[ACTIVE_MASK_FIX] check failed: {e}", flush=True)

            # 如果修复后没有 active env 了，直接跳出循环
            if not active_mask.any():
                print(f"[ACTIVE_MASK_FIX] No active envs left after fix, breaking loop at step={step}", flush=True)
                break

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            # 【修复】继承 rollings 的 meta_info（包含 do_sample, validate 等采样参数）
            # 确保验证/训练的采样行为不会因为 meta_info 丢失而退回默认
            if hasattr(rollings, 'meta_info') and rollings.meta_info:
                rollings_active.meta_info = rollings.meta_info.copy()
            else:
                rollings_active.meta_info = {}

            # 【新增】诊断轮增加 max_tokens：检测是否有 env 进入诊断轮
            # 诊断轮定义：current_turn >= max_turns - 1（即最后一轮及之后的补考轮）
            max_turns = self.config.max_turns
            active_indices = torch.where(active_mask)[0].tolist()
            is_diagnosis_step = False
            for idx in active_indices:
                env = envs[idx]
                if hasattr(env, 'current_turn') and env.current_turn >= max_turns - 1:
                    is_diagnosis_step = True
                    break
            
            # 如果是诊断轮，设置更大的 max_tokens（默认 1024 → 诊断轮 2048）
            if is_diagnosis_step:
                rollings_active.meta_info["max_tokens"] = 2048
                if step == max_turns - 1:  # 只在第一次进入诊断轮时打印
                    print(f"[DIAGNOSIS_STEP] step={step}: 进入诊断轮，设置 max_tokens=2048", flush=True)

            # 【诊断】检查 active_batch 中是否有全 pad 的序列，并追踪是哪些 env
            try:
                active_input_ids = rollings_active.batch.get('input_ids')
                if active_input_ids is not None:
                    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    all_pad_check = (active_input_ids == pad_id).all(dim=1)
                    if all_pad_check.any():
                        # 找出对应的原始 env 索引
                        active_indices = torch.where(active_mask)[0].tolist()
                        bad_active_indices = torch.where(all_pad_check)[0].tolist()
                        bad_env_indices = [active_indices[i] for i in bad_active_indices if i < len(active_indices)]
                        print(f"[ALL_PAD_TRACE] step={step}, active_size={len(active_indices)}, bad_count={len(bad_active_indices)}", flush=True)
                        for bad_env_idx in bad_env_indices[:3]:
                            env = envs[bad_env_idx]
                            dlg_id = getattr(env, 'dialogue_id', 'unknown')
                            turn = getattr(env, 'current_turn', -1)
                            finished = env.finished() if hasattr(env, 'finished') else 'unknown'
                            print(f"[ALL_PAD_TRACE] env_idx={bad_env_idx}, dlg_id={dlg_id}, turn={turn}, finished={finished}", flush=True)
            except Exception as e:
                print(f"[ALL_PAD_TRACE] diagnostic failed: {e}", flush=True)

            # 尝试生成，如果遇到全 pad batch 则优雅终止
            try:
                            gen_output = self._generate_with_gpu_padding(rollings_active)
            except AllPadBatchError as e:
                print(f"[ALL_PAD_BATCH_ABORT] Caught AllPadBatchError at step={step}: {e}", flush=True)
                print(f"[ALL_PAD_BATCH_ABORT] Forcing all remaining active envs to finish and breaking loop.", flush=True)
                # 强制把所有 active env 标记为 done，并打印每个 env 的详细信息
                affected_envs = []
                for i, env in enumerate(envs):
                    if active_mask[i]:
                        dlg_id = getattr(env, 'dialogue_id', f'env-{i}')
                        turn = getattr(env, 'current_turn', -1)
                        diag_made = getattr(env, 'diagnosis_made', False)
                        last_prompt_len = len(getattr(env, '_last_doctor_prompt', '') or '')
                        
                        env._force_terminated = True
                        env._force_terminate_reason = f"AllPadBatchError at step={step}: input became all pad tokens"
                        affected_envs.append({
                            'idx': i, 'dlg_id': dlg_id, 'turn': turn,
                            'diagnosis_made': diag_made, 'last_prompt_len': last_prompt_len
                        })
                        print(
                            f"[FORCE_TERMINATED_ALLPAD] idx={i}, dlg={dlg_id}, turn={turn}, "
                            f"diagnosis_made={diag_made}, last_prompt_len={last_prompt_len}",
                            flush=True
                        )
                print(f"[ALL_PAD_BATCH_ABORT] Total {len(affected_envs)} envs force terminated.", flush=True)
                # 更新 active_mask 并跳出循环
                active_mask = torch.zeros(len(envs), dtype=torch.bool)
                active_num_list.append(0)
                break  # 跳出 for step in range(self.config.max_turns) 循环

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'],envs=envs)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Update visualization
            self._update_trajectory(trajectory, envs, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones = self.env_class.execute_predictions(
                envs, responses_str, responses_ids, self.tokenizer
            )
            
            # ========== [对齐检查 1] len(dones) == len(envs) ==========
            if len(dones) != len(envs):
                print(f"[DONES_LEN_MISMATCH] step={step}: len(dones)={len(dones)} != len(envs)={len(envs)}", flush=True)
                print(f"[DONES_LEN_MISMATCH] This is a critical alignment bug! Training may be corrupted.", flush=True)

            # ========== [对齐检查 2] dones[i] vs envs[i].finished() ==========
            try:
                env_finished_list = [env.finished() for env in envs]
                mismatches = []
                for i, (done_val, env_finished) in enumerate(zip(dones, env_finished_list)):
                    # dones 来自 execute_predictions，env_finished 来自 env.finished()
                    # 如果不一致，说明有对齐问题
                    done_bool = bool(done_val)
                    if done_bool != env_finished:
                        dlg_id = getattr(envs[i], 'dialogue_id', f'env-{i}')
                        turn = getattr(envs[i], 'current_turn', -1)
                        mismatches.append({
                            'idx': i,
                            'dlg_id': dlg_id,
                            'turn': turn,
                            'dones_val': done_bool,
                            'env_finished': env_finished
                        })
                if mismatches:
                    print(f"[DONES_FINISHED_MISMATCH] step={step}: found {len(mismatches)} mismatches!", flush=True)
                    for m in mismatches[:10]:  # 只打印前10个
                        print(f"  idx={m['idx']}, dlg_id={m['dlg_id']}, turn={m['turn']}, "
                              f"dones={m['dones_val']}, env.finished()={m['env_finished']}", flush=True)
            except Exception as e:
                print(f"[DONES_FINISHED_MISMATCH] check failed: {e}", flush=True)

            # 【根修复】以 env.finished() 为准更新 active_mask，而不是直接用 dones
            # 这样即使 dones 有对齐问题，也能保证 active_mask 是正确的
            active_mask = torch.tensor([not env.finished() for env in envs], dtype=torch.bool)
            active_num_list.append(active_mask.sum().item())
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            responses_ids = responses_ids.long()
            next_obs_ids = next_obs_ids.long()
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # Save trajectory and return final output
        self._save_trajectory(trajectory, output_dir, global_steps)
        
        # 使用新的轨迹构造方式：从 env.conversation_history 重建
        # 【关键改动】responses 部分已包含简化版问诊 prompt，所以 prompts 部分用空的或极简的
        rebuilt_result = self._build_trajectory_from_history(envs, initial_input_ids)
        rebuilt_right_side = {
            'responses': rebuilt_result['responses'],
            'action_mask': rebuilt_result['action_mask']
        }
        
        # 【方案 A】把 action_end_positions 传递给后续 reward 分配
        # 这个信息会通过 meta_info 传给 RewardManager
        action_end_positions = rebuilt_result.get('action_end_positions', None)
        if action_end_positions is not None:
            meta_info['action_end_positions'] = action_end_positions
            # 打印调试信息（只打印前 2 个样本）
            for i in range(min(2, len(action_end_positions))):
                print(f"[DEBUG run_llm_loop] env {i}: action_end_positions = {action_end_positions[i]}")
        
        # 【关键改动】创建简化版的 left_side（只包含 BOS token 或空）
        # 因为 responses 部分已经包含了简化版问诊 prompt
        batch_size = initial_input_ids.shape[0]
        device = initial_input_ids.device
        
        # 使用 BOS token（如果有）或 pad token 作为最小 prompt
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        if bos_token_id is None:
            bos_token_id = self.tokenizer.pad_token_id
        
        # 创建只有 1 个 token 的 minimal prompts
        minimal_prompts = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        simplified_left_side = {
            'input_ids': minimal_prompts
        }
        
        return self._compose_final_output(simplified_left_side, rebuilt_right_side, meta_info)

    def _setup_visualization(self) -> List[Dict]:
        """Setup visualization tracking if enabled."""
        if not self.config.logging.log_images:
            return None
        return [defaultdict(list) for _ in range(self.config.logging.log_n_image_per_batch)]

    def _update_trajectory(self, trajectory: List[Dict], 
                         envs: List[Any], responses: List[str], active_mask: torch.Tensor):
        """Update visualization trajectory if enabled."""
        if not trajectory:
            return
        n_visualize = self.config.logging.log_n_image_per_batch
        for idx, (env, response, active) in enumerate(zip(envs[:n_visualize],
                                                          responses[:n_visualize],
                                                          active_mask[:n_visualize])):
            if not active:
                continue
            
            state_repr = env.render('rgb_array')
            trajectory[idx]['state'].append(state_repr)
            
            parsed = parse_llm_output(response, strategy="raw")
            trajectory[idx]['answer'].append(response)
            trajectory[idx]['parsed_response'].append(parsed)
            
            prompt_text = getattr(env, "_last_doctor_prompt", "") or ""
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            thought_text = think_match.group(1).strip() if think_match else ""
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            answer_text = answer_match.group(1).strip() if answer_match else ""
            patient_status = getattr(env, "_last_patient_status", None)
            if patient_status:
                if patient_status.get("type") == "response":
                    patient_feedback = patient_status.get("content", "")
                else:
                    patient_feedback = f"(系统提示) {patient_status.get('reason', '')}"
            else:
                patient_feedback = env.render('rgb_array')
            
            steps_list = trajectory[idx].setdefault('steps', [])
            step_entry = {
                "turn": len(steps_list) + 1,
                "prompt": prompt_text,
                "observation": state_repr,
                "response": response,
                "thought": thought_text,
                "answer": answer_text,
                "patient_status": patient_feedback
            }
            steps_list.append(step_entry)

    def _save_trajectory(self, trajectory: List[Dict], 
                        output_dir: str, global_steps: int):
        """Save trajectory visualization if enabled."""
        if not trajectory:
            return
            
        save_step_size = self.config.logging.log_image_step_size
        if not global_steps % save_step_size or self.is_validation:
            # 将 output_dir 转换为绝对路径，避免 wandb.save() 报错
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            filenames = save_trajectory_to_output(trajectory, save_dir=output_dir)
            if 'wandb' in self.logger.logger:
                for filename in filenames:
                    # 确保 filename 也是绝对路径
                    abs_filename = os.path.abspath(filename)
                    self.logger.logger['wandb'].save(abs_filename)


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output.
        
        构建最终输出，包含：
        - input_ids: [prompts] + [responses]
        - attention_mask: 标记有效 token
        - response_mask: 标记哪些 token 需要计算 loss（只有医生输出）
        """
        final_output = {}
        
        # 获取实际的 prompt 长度（去掉 padding 后的有效长度）
        prompt_ids = left_side['input_ids']
        prompt_len = prompt_ids.shape[1]
        
        # 截断 responses，确保 prompt + responses <= HARD_LIMIT_TOTAL_SEQLEN
        # 【从左边截断】保留最近的对话（右边），丢弃最早的对话（左边）
        max_response_len = self.HARD_LIMIT_TOTAL_SEQLEN - prompt_len
        responses = right_side['responses']
        if responses.shape[1] > max_response_len:
            truncate_len = responses.shape[1] - max_response_len
            print(f"[WARNING] _compose_final_output: left-truncating responses from {responses.shape[1]} to {max_response_len} (removing first {truncate_len} tokens)")
            responses = responses[:, truncate_len:]  # 从左边开始截断
            # 同时截断 action_mask
            if 'action_mask' in right_side:
                right_side['action_mask'] = right_side['action_mask'][:, truncate_len:]
        
        final_output['responses'] = responses
        final_output['prompts'] = prompt_ids
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            prompt_ids,
            responses
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        # 构建 response_mask：只包含 responses 部分，标记哪些 token 需要计算 loss（只有医生输出部分）
        # response_mask 的长度必须等于 responses 的长度（而不是 prompts + responses）
        if 'action_mask' in right_side:
            final_output['response_mask'] = right_side['action_mask']
        else:
            # 回退：如果没有 action_mask，使用 responses 的 attention_mask
            final_output['response_mask'] = self.tensor_fn.create_attention_mask(final_output['responses'])
        
        # ===== 调试信息：检查最终输出的 shape =====
        max_seqlen = final_output['attention_mask'].sum(dim=-1).max().item()
        print(f"[DEBUG _compose_final_output] 最终输出:")
        print(f"  input_ids.shape: {final_output['input_ids'].shape}")
        print(f"  prompts.shape: {final_output['prompts'].shape}")
        print(f"  responses.shape: {final_output['responses'].shape}")
        print(f"  attention_mask.sum().max(): {max_seqlen}")
        if max_seqlen > self.HARD_LIMIT_TOTAL_SEQLEN:
            print(f"  [ERROR] 有效 token 数 ({max_seqlen}) 超过硬限制 ({self.HARD_LIMIT_TOTAL_SEQLEN})!")
        # ===== 调试信息结束 =====
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

        # ===== 调试信息：检查最终输出的 shape =====
        max_seqlen = final_output['attention_mask'].sum(dim=-1).max().item()
        print(f"[DEBUG _compose_final_output] 最终输出:")
        print(f"  input_ids.shape: {final_output['input_ids'].shape}")
        print(f"  prompts.shape: {final_output['prompts'].shape}")
        print(f"  responses.shape: {final_output['responses'].shape}")
        print(f"  attention_mask.sum().max(): {max_seqlen}")
        if max_seqlen > self.HARD_LIMIT_TOTAL_SEQLEN:
            print(f"  [ERROR] 有效 token 数 ({max_seqlen}) 超过硬限制 ({self.HARD_LIMIT_TOTAL_SEQLEN})!")
        # ===== 调试信息结束 =====
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
        
        # ===== 调试信息：检查最终输出的 shape =====
        max_seqlen = final_output['attention_mask'].sum(dim=-1).max().item()
        print(f"[DEBUG _compose_final_output] 最终输出:")
        print(f"  input_ids.shape: {final_output['input_ids'].shape}")
        print(f"  prompts.shape: {final_output['prompts'].shape}")
        print(f"  responses.shape: {final_output['responses'].shape}")
        print(f"  attention_mask.sum().max(): {max_seqlen}")
        if max_seqlen > self.HARD_LIMIT_TOTAL_SEQLEN:
            print(f"  [ERROR] 有效 token 数 ({max_seqlen}) 超过硬限制 ({self.HARD_LIMIT_TOTAL_SEQLEN})!")
        # ===== 调试信息结束 =====
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
