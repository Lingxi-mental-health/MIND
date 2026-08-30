import yaml
import os
import argparse
import json
from typing import Dict, Any
import time

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any], noassert_keys=["env", ""]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    assert isinstance(base_dict, dict) and isinstance(update_dict, dict)
    # base dict is the base yaml, update dict is the training args. make sure no training args is beyond what's in the base yaml
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            assert key in base_dict or key in noassert_keys, f"Key {key} not found in base config"
            base_dict[key] = value
    return base_dict

def load_config(env_name: str) -> Dict[str, Any]:
    """Load configuration from base and environment-specific configs."""
    # Load base config
    with open("config/base.yaml", 'r') as f:
        config = yaml.safe_load(f)
    #print("config1",config)
    # Load environment config
    env_config_path = f"config/env/{env_name}.yaml"
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            config = deep_update(config, env_config)
    else:
        raise ValueError(f"Environment config not found: {env_config_path}")
    
    return config

def get_train_command(config: Dict[str, Any]) -> str:
    """Generate the training command with all arguments."""
    # Check if we're doing RL or SFT training
    assert config['rl_or_sft'] in ["rl", "sft"]
    if config['rl_or_sft'] == "rl":
        return get_rl_train_command(config)
    else:
        return get_sft_train_command(config)

def get_rl_train_command(config: Dict[str, Any]) -> str:
    """Generate the RL training command with all arguments."""
    # Calculate MAX_PROMPT_LENGTH
    max_prompt_length = (config['training']['max_start_length'] +
                        config['training']['max_response_length'] * (config['training']['max_turns'] - 1) +
                        config['training']['max_obs_length'] * config['training']['max_turns'])
   
    # Define the command template with proper indentation
    env_kwargs = config['env']['env_kwargs']
    env_kwargs_str = " \\\n    ".join([
        f"+env.{key}={value}" if value is not None else f"+env.{key}=null" for key, value in env_kwargs.items()
    ])
    
    # Add env_llm configuration
    env_llm_config = config['env'].get('env_llm', {})
    env_llm_str = ""
    if env_llm_config:
        # FSDP config - only keep fsdp_size and param_offload
        fsdp_config = env_llm_config.get('fsdp_config', {})
        if fsdp_config:
            fsdp_params = {k: v for k, v in fsdp_config.items() if k in ['fsdp_size', 'param_offload']}
            if fsdp_params:
                env_llm_str += " \\\n    ".join([
                    f"env.env_llm.fsdp_config.{key}={value}" for key, value in fsdp_params.items()
                ])
                env_llm_str += " \\\n    "
        
        # vLLM config
        vllm_config = env_llm_config.get('vllm_config', {})
        if vllm_config:
            env_llm_str += " \\\n    ".join([
                f"env.env_llm.vllm_config.{key}={value}" for key, value in vllm_config.items()
            ])
            env_llm_str += " \\\n    "
        
        # Model config
        model_config = env_llm_config.get('model', {})
        if model_config:
            model_params = {k: v for k, v in model_config.items() if k in ['path', 'trust_remote_code', 'use_liger']}
            if model_params:
                env_llm_str += " \\\n    ".join([
                    f"env.env_llm.model.{key}={value}" for key, value in model_params.items()
                ])
            env_llm_str += " \\\n    "
        
        # Generation config
        generation_config = env_llm_config.get('generation', {})
        if generation_config:
            env_llm_str += " \\\n    ".join([
                f"env.env_llm.generation.{key}={value}" for key, value in generation_config.items()
            ])
            env_llm_str += " \\\n    "
        
        # Other env_llm configs
        other_configs = {k: v for k, v in env_llm_config.items() 
                        if k not in ['fsdp_config', 'vllm_config', 'model', 'generation']}
        if other_configs:
            env_llm_str += " \\\n    ".join([
                f"env.env_llm.{key}={value}" for key, value in other_configs.items()
            ])
            env_llm_str += " \\\n    "
    
    cmd = [
        f"VLLM_ATTENTION_BACKEND={config['system']['vllm_attention_backend']}",
        f"CUDA_VISIBLE_DEVICES={config['system']['cuda_visible_devices']}",
        "python -m ragen.trainer.main_ppo",
        f"hydra.run.dir={config['system']['hydra_output_subdir']}",
        f"data.train_files={config['env']['train_path']}",
        f"data.val_files={config['env']['val_path']}",
        f"data.train_data_num={config['training']['train_data_num'] or 'null'}",
        f"data.val_data_num={config['training']['val_data_num'] or 'null'}",
        f"data.train_batch_size={config['training']['train_batch_size']}",
        f"data.val_batch_size={config['training']['val_batch_size']}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={config['training']['max_response_length']}",
        f"data.max_start_length={config['training']['max_start_length']}",
        f"data.max_obs_length={config['training']['max_obs_length']}",
        "data.shuffle=True",
        f"algorithm.adv_estimator={config['optimization']['adv_estimator']}",
        f"actor_rollout_ref.model.path={config['model']['base_model']}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(config['model']['gradient_checkpointing']).lower()}",
        f"actor_rollout_ref.actor.optim.lr={config['optimization']['actor_lr']}",
        f"actor_rollout_ref.actor.use_kl_loss={config['optimization']['use_kl_loss']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config['training']['ppo_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={config['training']['micro_batch_size']}", # NOTE: This is deprecated, use ppo_micro_batch_size_per_gpu instead
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config['training']['rollout_tp_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config['optimization']['gpu_memory_utilization']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={config['training']['micro_batch_size']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size={config['training']['micro_batch_size']}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size={config['training']['micro_batch_size']}",
        f"critic.ppo_micro_batch_size={config['training']['micro_batch_size']}",
        # f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config['training']['micro_batch_size']}",
        # f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config['training']['micro_batch_size']}",
        # f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config['training']['micro_batch_size']}",
        # f"critic.ppo_micro_batch_size_per_gpu={config['training']['micro_batch_size']}",
        f"critic.optim.lr={config['optimization']['critic_lr']}",
        f"critic.model.path={config['model']['base_model']}",
        f"algorithm.kl_penalty={config['optimization']['kl_type']}",
        f"algorithm.kl_ctrl.kl_coef={config['optimization']['kl_coef']}",
        f"+algorithm.no_ref_policy={config['optimization']['no_ref_policy']}",
        f"+actor_rollout_ref.actor.use_ref_policy={not config['optimization']['no_ref_policy']}",
        f"actor_rollout_ref.actor.kl_loss_coef={config['optimization']['kl_coef']}", # for use_kl_loss=True. ARPO/BRPO/GRPO needs the original model with "low_var_kl"
        f"actor_rollout_ref.actor.kl_loss_type={config['optimization']['kl_type']}",
        f"+algorithm.no_think_rl={config['training']['no_think_rl']}",
        f"+algorithm.reward_norm_type={config['optimization']['reward_norm_type'] or 'null'}",
        f"+actor_rollout_ref.actor.optim.betas=[{config['optimization']['betas'][0]},{config['optimization']['betas'][1]}]",
        f"+critic.optim.betas=[{config['optimization']['betas'][0]},{config['optimization']['betas'][1]}]",
        f"actor_rollout_ref.rollout.n_agent={config['training']['n_rollout']}",
        f"actor_rollout_ref.rollout.temperature={config['training']['temperature']}",
        f"actor_rollout_ref.actor.state_masking={config['training']['state_masking']}",
        f"trainer.logger={config['logging']['mode']}",
        f"+trainer.val_only={str(config['trainer']['val_only']).lower()}",
        f"+trainer.val_before_train={str(config['trainer']['val_before_train']).lower()}",
        f"trainer.default_hdfs_dir={config['trainer']['default_hdfs_dir'] or 'null'}",
        f"trainer.n_gpus_per_node={config['system']['n_gpus']}",
        f"trainer.nnodes={config['trainer']['nnodes']}",
        f"trainer.save_freq={config['trainer']['save_freq']}",
        f"trainer.test_freq={config['trainer']['test_freq']}",
        f"trainer.project_name={config['trainer']['project_name']}",
        f"trainer.experiment_name={config['model']['experiment_name']}",
        f"trainer.total_epochs={config['training']['total_epochs']}",
        f"trainer.total_training_steps={config['training']['total_training_steps'] or 'null'}",
        f"+trainer.ref_update_steps={config['training']['ref_update_steps'] or 'null'}",
        f"env.name={config['env']['name']}",
        f"env.use_env_llm={config['env'].get('use_env_llm', False)}",
        env_kwargs_str,
        env_llm_str,  # Add env_llm configuration
        f"max_turns={config['training']['max_turns']}",
        f"logging.log_images={str(config['logging']['log_images']).lower()}",
        f"logging.log_image_dir={config['logging']['log_image_dir']}",
        f"logging.log_image_step_size={config['logging']['log_image_step_size']}",
        f"logging.log_n_image_per_batch={config['logging']['log_n_image_per_batch']}",
        "2>&1"
    ]
   
    return " \\\n    ".join(cmd)

def get_sft_train_command(config: Dict[str, Any],config_dir="./outputs/exp_configs/sft") -> str:
    """
    Generate the SFT training command by saving the config to a temporary YAML file
    and using it for the SFT pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary for SFT training
        
    Returns:
        str: The complete command string for running SFT training
    """
    # Create sft_configs directory if it doesn't exist
    
    os.makedirs(config_dir, exist_ok=True)
    # Generate unique filename using process ID and timestamp
    pid = os.getpid()
    timestamp = int(time.time())
    config_hash = f"{pid}_{timestamp}"
    config_path = os.path.join(config_dir, f"sft_config_{config_hash}.yaml")
    
    # Save config to YAML file
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    
    # Generate command using the temporary config file
    cmd = [
        "python -m sft.sft_pipeline",
        f"--config {config_path}",
        f"--env_type {config['env']['name']}"
    ]
    
    return " \\\n    ".join(cmd)

def parse_override_args(args_list):
    """Parse override arguments in the format key=value."""
    overrides = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
            
            # Handle nested keys
            current = overrides
            key_parts = key.split('.')
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[key_parts[-1]] = value
    return overrides

def main():
    parser = argparse.ArgumentParser(description='Generate training command with config')
    parser.add_argument('env_name', nargs='?', default='sokoban', help='Environment name')
    parser.add_argument('overrides', nargs='*', help='Config overrides in the format key=value')
    
    args = parser.parse_args()
    
    # Load base config
    config = load_config(args.env_name)
    
    # Apply command line overrides
    if args.overrides:
        overrides = parse_override_args(args.overrides)
        config = deep_update(config, overrides)
    
    import shlex
    import subprocess
    command = get_train_command(config)
    print("Running command:")
    print(command)
    # shell=False + shlex.split：命令由 YAML 配置拼装，交给 shell 解释会让配置值里的
    # ;  |  $(...) 等字符变成可执行语法。shlex.split 保留续行产生的空白 token，滤掉。
    argv = [tok for tok in shlex.split(command) if tok.strip()]
    subprocess.run(argv)

if __name__ == "__main__":
    main()