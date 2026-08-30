from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional, Tuple
import itertools
import subprocess
import os
from datetime import datetime
import time
import argparse
from enum import Enum, auto
import re
import ast
import shlex
import json
from pathlib import Path

import multiprocessing as mp
from multiprocessing import Process, Queue, Lock

class GPUManager:
    def __init__(self):
        self.lock = Lock()
        self.device_count = self._get_gpu_count()
        self.gpu_status = mp.Array('i', [0] * self.device_count)  # 0 = free, 1 = used
        print(f"Found {self.device_count} GPUs")
        
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs using nvidia-smi"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'],
                                 capture_output=True, text=True)
            return len(result.stdout.strip().split('\n'))
        except:
            print("Warning: Could not get GPU count, falling back to 1")
            return 1
    
    def get_gpu_memory_info(self, gpu_id: int) -> Tuple[int, int]:
        """Get memory usage for specific GPU using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            used, total = map(int, result.stdout.strip().split(','))
            return used, total
        except Exception as e:
            print(f"Error getting GPU {gpu_id} memory info: {e}")
            return 0, 0
    
    def acquire_gpu(self) -> Optional[int]:
        """Try to acquire an available GPU"""
        with self.lock:
            for gpu_id in range(self.device_count):
                if self.gpu_status[gpu_id] == 0:
                    used, total = self.get_gpu_memory_info(gpu_id)
                    if total > 0 and used/total < 0.5:  # Less than 50% memory used
                        self.gpu_status[gpu_id] = 1
                        return gpu_id
        return None

    def release_gpu(self, gpu_id: int):
        """Release a GPU back to the pool"""
        with self.lock:
            self.gpu_status[gpu_id] = 0

def _to_argv(command: str) -> List[str]:
    """把多行 shell 命令串转成 argv 列表，供 shell=False 使用。

    命令由本地配置拼装，但仍不该交给 shell 解释：配置值里的 ;  |  $(...)
    会变成可执行语法。shlex.split 会为续行留下纯空白 token，一并滤掉。
    """
    return [tok for tok in shlex.split(command) if tok.strip()]


def run_experiment(command: str, gpu_id: int, result_queue: Queue, log_dir: Path):
    """Run a single experiment on specified GPU"""
    try:
        # Ensure unique log file per process
        log_file = log_dir / f"gpu_{gpu_id}_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Modify command to use specific GPU
        # gpu_command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        gpu_command = f"system.cuda_visible_devices={gpu_id}"
        final_command_parts = [command, gpu_command]
        gpu_command = " \\\n".join(final_command_parts)
        print(f"Running on GPU {gpu_id}: {gpu_command}")
        
        # Run the experiment
        with open(log_file, 'w') as f:
            result = subprocess.run(_to_argv(gpu_command), check=True, stdout=f, stderr=subprocess.STDOUT)
        
        # Parse results
        metrics = None
        if result.returncode == 0:
            with open(log_file, 'r') as f:
                log_content = f.read()
            groups = re.findall(r"global_metrics\s+({[^}]+})", log_content)
            if groups:
                metrics = ast.literal_eval(groups[-1])
        
        result_queue.put((gpu_id, metrics))
        
    except Exception as e:
        print(f"Error in experiment on GPU {gpu_id}: {e}")
        result_queue.put((gpu_id, None))

# ============= Hyperparameter Configuration Part =============
class ParamType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()

@dataclass
class HyperParam:
    name: str
    param_type: ParamType
    default_value: Any
    search_space: Optional[List[Any]] = None
    group: int = 0
    
    def get_values(self, mode: str) -> List[Any]:
        """Get parameter values based on mode (search/fixed/default)"""
        if mode == "search" and self.search_space:
            return self.search_space
        return [self.default_value]

@dataclass
class HyperParamGroup:
    group_id: int
    params: List[HyperParam]
    description: str = ""

class HyperParamConfig:
    def __init__(self):
        self.groups: Dict[int, HyperParamGroup] = {}
        self.fixed_params: Dict[str, Any] = {
            "model.base_model": ["Qwen/Qwen2.5-3B-Instruct"],
            "optimization.adv_estimator": ["grpo"],
            "training.train_data_num": [500],
            "training.val_data_num": [50],
            "training.micro_batch_size": [2],
            "training.total_epochs": [1],
        }
        self._initialize_groups()

    def _initialize_groups(self):
        """Initialize all hyperparameter groups"""
        self.groups = {
            # 1: HyperParamGroup(1, [
            #     HyperParam("training.ppo_batch_size", ParamType.NUMERIC, 64, 
            #               search_space=[16, 32, 64, 128, 256], group=1)
            # ], "PPO Batch Size"), # 128
            
            1: HyperParamGroup(1, [
                HyperParam("training.train_batch_size", ParamType.NUMERIC, 8,
                          search_space=[4, 8, 16], group=1), # if bandits 128, if sokoban/fr 8
                HyperParam("training.n_rollout", ParamType.NUMERIC, 16,
                          search_space=[4, 8, 16], group=1) # if bandits 1, if sokoban/fr 16
            ], "Training Batch Size and Rollout"),

            2: HyperParamGroup(2, [
                HyperParam("optimization.actor_lr", ParamType.NUMERIC, 1e-6,
                          search_space=[5e-7, 1e-6, 5e-6, 1e-5], group=2) # 1e-6
            ], "Actor Learning Rate"),
            
            3: HyperParamGroup(3, [
                HyperParam("optimization.kl_coef", ParamType.NUMERIC, 0.04,
                          search_space=[0.001, 0.005, 0.01, 0.04, 0.1, 0.5], group=3) # 0.04
            ], "KL Coefficient"),
            
            4: HyperParamGroup(4, [
                HyperParam("training.max_turns", ParamType.NUMERIC, 5,
                          search_space=[2, 5, 8], group=4), # if fr/sokoban 5, if bandits 1; fr/sokobaninference 10, 
                HyperParam("training.temperature", ParamType.NUMERIC, 1.0,
                          search_space=[0.1, 0.5, 1], group=4) # eval 0.7
            ], "Max Turns and Temperature"),
        }


# ============= Hyperparameter Search Part =============
class HyperParamSearch:
    def __init__(self, config: HyperParamConfig):
        self.config = config
        self.param_grid: Dict[str, List[Any]] = {}
        self.searching_params: List[str] = []
        self.log_name = ""
        self.search_group_id = -1
        self.searched_param_log_path = Path('./log/searched_hyper_params')
        self.log_dir = Path('./log/terminal/hyper_param_search_logs')
        try:
            self.gpu_manager = GPUManager()
        except Exception as e:
            print(f"Error initializing GPU manager: {e}")
            print("Falling back to single GPU mode")
            self.gpu_manager = None
        # Create directories
        self.searched_param_log_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_search_group(self, search_group: int, fixed_values: Dict[str, Any]) -> None:
        """Set up parameter grid based on search group and fixed values"""
        # Add fixed parameters
        self.param_grid.update(self.config.fixed_params)
        self.search_group_id = search_group
        
        if 'system.n_gpus' in fixed_values:
            self.param_grid['system.n_gpus'] = [fixed_values['system.n_gpus']]
        if 'training.micro_batch_size' in fixed_values:
            self.param_grid['training.micro_batch_size'] = [fixed_values['training.micro_batch_size']]
        # Process each group based on its relation to the search group
        for group_id, group in self.config.groups.items():
            if group_id < search_group:
                # Fixed values for previous groups
                for param in group.params:
                    if param.name in fixed_values:
                        self.param_grid[param.name] = [fixed_values[param.name]]
                    else:
                        raise ValueError(f"Missing fixed value for {param.name} in group {group_id}")
            
            elif group_id == search_group:
                # Search space for current group
                for param in group.params:
                    if param.search_space:
                        self.param_grid[param.name] = param.search_space
                        self.searching_params.append(param.name)
            
            else:
                # Default values for future groups
                for param in group.params:
                    self.param_grid[param.name] = [param.default_value]

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combinations"""
        if all(key in params for key in ["training.ppo_batch_size", "training.train_batch_size", "training.n_rollout"]):
            if params["training.ppo_batch_size"] > params["training.train_batch_size"] * params["training.n_rollout"]:
                print(f"Invalid combination: ppo_batch_size ({params['training.ppo_batch_size']}) > "
                      f"train_batch_size ({params['training.train_batch_size']}) * "
                      f"n_rollout ({params['training.n_rollout']})")
                return False
        return True

    def generate_command(self, params: Dict[str, Any], base_name: str, env_name: str = "sokoban") -> str:
        """Generate command string for running experiment"""
        param_parts = []
        for param_name in self.searching_params:
            if param_name in params:
                value = params[param_name]
                param_name_short = param_name.split('.')[-1]
                formatted_value = self._format_value(value)
                param_parts.append(f"{param_name_short}_{formatted_value}")

        timestamp = datetime.now().strftime("%Hh%Mm%Ssec")
        self.log_name = f"[hs]_{'_'.join(param_parts)}"
        experiment_name = f"{base_name}_{'_'.join(param_parts)}_{timestamp}"

        cmd_parts = [
            f"bash train.sh {env_name}",
            f"model.experiment_name={experiment_name}"
        ]
        
        for key, value in params.items():
            cmd_parts.append(f"{key}={value}")

        return " \\\n".join(cmd_parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format parameter value for command string"""
        if isinstance(value, float):
            if value < 0.0001:
                return f"{value:.2e}".replace('-0', '-').replace('+0', '')
            return f"{value:.4f}".rstrip('0').rstrip('.')
        return str(value)
    
    def run_parallel_experiments(self, experiments: List[Tuple[Dict[str, Any], str]], 
                               dry_run: bool = True, is_parallel: bool = False) -> Optional[Dict[str, Any]]:
        if dry_run:
            return self._handle_dry_run(experiments)

        if not self.gpu_manager or not is_parallel: # if no GPUs or not parallel mode
            return self._run_sequential(experiments)

        result_queue = Queue()
        running_processes: List[Tuple[Process, int, Dict[str, Any]]] = []  # (process, gpu_id, params)
        completed_results: List[Tuple[Dict[str, Any], Dict]] = []
        remaining_experiments = experiments.copy()

        is_first_exp = True

        while remaining_experiments or running_processes:
            # Start new experiments if GPUs are available
            while remaining_experiments:
                gpu_id = self.gpu_manager.acquire_gpu()
                if gpu_id is None:
                    break
                    
                params, command = remaining_experiments.pop(0)
                process = Process(
                    target=run_experiment,
                    args=(command, gpu_id, result_queue, self.log_dir)
                )
                print("Starting experiment on GPU", gpu_id)
                if is_first_exp:
                    is_first_exp = False
                else:
                    print("Sleeping for 30 seconds to avoid ray conflicts")
                    time.sleep(30) # Wait for ray to be ready
                process.start()
                running_processes.append((process, gpu_id, params))
                print(f"Started experiment on GPU {gpu_id}")

            # Check for completed processes
            if not result_queue.empty():
                gpu_id, result = result_queue.get()
                for i, (process, proc_gpu_id, params) in enumerate(running_processes):
                    if proc_gpu_id == gpu_id:
                        process.join()
                        self.gpu_manager.release_gpu(gpu_id)
                        if result is not None:
                            completed_results.append((params, result))
                            print(f"Experiment completed with score: {result['global_score/mean']}")
                        running_processes.pop(i)
                        print(f"Completed experiment on GPU {gpu_id}")
                        break

            time.sleep(1)  # Prevent busy waiting

        # Find best result
        if completed_results:
            best_result = max(completed_results, key=lambda x: x[1]['global_score/mean'])
            return best_result[0]
        return None
    
    def _run_sequential(self, experiments: List[Tuple[Dict[str, Any], str]]) -> Optional[Dict[str, Any]]:
        """Fallback method for sequential execution"""
        best_params = None
        best_score = float('-inf')

        for params, command in experiments:
            print(f"Running experiment with params: {params}")
            try:
                log_file = self.log_dir / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
                with open(log_file, 'w') as f:
                    result = subprocess.run(_to_argv(command), check=True, stdout=f, stderr=subprocess.STDOUT)

                if result.returncode == 0:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    groups = re.findall(r"global_metrics\s+({[^}]+})", log_content)
                    if groups:
                        metrics = ast.literal_eval(groups[-1])
                        score = metrics['global_score/mean']
                        if score > best_score:
                            best_score = score
                            best_params = params
                            print(f"New best score: {score}")
            except Exception as e:
                print(f"Error running experiment: {e}")

        return best_params

    def run_grid_search(self, base_experiment_name: str, dry_run: bool = True, env_name: str = "sokoban", parallel: bool = False) -> None:
        """Run grid search with parallel GPU execution"""
        combinations = [dict(zip(self.param_grid.keys(), vals)) 
                       for vals in itertools.product(*self.param_grid.values())]
        
        print(f"Total combinations to try: {len(combinations)}")
        
        # Prepare experiments
        experiments = []
        for params in combinations:
            if not self.validate_params(params):
                print(f"Skipping invalid combination: {params}")
                continue
                
            params["training.ppo_batch_size"] = params["training.train_batch_size"] * params["training.n_rollout"]
            command = self.generate_command(params, base_experiment_name, env_name)
            experiments.append((params, command))

        # Run experiments in parallel
        best_params = self.run_parallel_experiments(experiments, dry_run, parallel)
        
        if best_params and not dry_run:
            print(f"Best combination found: {best_params}")
            
            # Save best parameters
            param_file = self.searched_param_log_path / f"searched_params_group_{self.search_group_id}.json"
            with open(param_file, 'w') as f:
                json.dump(best_params, f, indent=2)
    
    def _handle_dry_run(self, experiments: List[Tuple[Dict[str, Any], str]]) -> Optional[Dict[str, Any]]:
        """Handle dry run mode with simulated results"""
        for params, command in experiments:
            print(f"Would run command: {command}")
            fake_metrics = {
                'global_score/mean': float(hash(str(params)) % 100) / 10.0,
                'global_score/max': float(hash(str(params)) % 150) / 10.0,
                'global_score/min': float(hash(str(params)) % 50) / 10.0,
                'global_score/std': float(hash(str(params)) % 20) / 10.0
            }
            print('Dry run - simulated metrics:', fake_metrics)
        
        return experiments[0][0] if experiments else None


    def _setup_log_directory(self) -> str:
        """Set up logging directory"""
        log_dir = "./log/terminal/hyper_param_search_logs/"
        
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


# ============= Main Functionality Part =============
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description='Hyperparameter search with group-based control')
    
    # Required arguments
    parser.add_argument('--env_name', type=str, required=True,
                      help='Environment name (e.g., sokoban)')
    parser.add_argument('--base_experiment_name', type=str, required=True,
                      help='Base name for the experiment')
    parser.add_argument('--search_group', type=int, required=True, choices=range(1, 5),
                      help='Group number to search (1-4)')
    
    # Optional arguments
    parser.add_argument('--dry_run', action='store_true',
                      help='Print commands without executing')
    parser.add_argument('--parallel', action='store_true',
                      help='Run experiments in parallel (default: sequential)')
    
    # Group-specific parameters
    parser.add_argument('--ppo_batch_size', type=int,
                      help='Fixed value for PPO batch size (required for groups 2-5)')
    parser.add_argument('--train_batch_size', type=int,
                      help='Fixed value for training batch size (required for groups 3-5)')
    parser.add_argument('--n_rollout', type=int,
                      help='Fixed value for number of rollouts (required for groups 3-5)')
    parser.add_argument('--kl_coef', type=float,
                      help='Fixed value for KL coefficient (required for groups 4-5)')
    parser.add_argument('--max_turns', type=int,
                      help='Fixed value for max turns (required for group 5)')
    parser.add_argument('--temperature', type=float,
                      help='Fixed value for temperature (required for group 5)')
    parser.add_argument('--actor_lr', type=float,
                      help='Fixed value for actor learning rate (required for group 2)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--micro_batch_size', type=int, default=2,
                        help='Micro batch size for RAGEN training, must be greater than n_gpus')
    
    return parser

def read_searched_params(group_id: int, params: List[str]) -> Union[Dict[str, Any], None]:
    """Read searched parameters from JSON file"""
    if group_id == 0:
        return None
    file_name = os.path.join('./log/searched_hyper_params', f"searched_params_group_{group_id}.json")
    if not os.path.exists(file_name):
        assert False, f"searched params for group {group_id} not found, please run the search first"
    with open(file_name, 'r') as f:
        searched_params = json.load(f)
    return {param: searched_params[param] for param in params}

def get_param_arg_name(param_name: str) -> str:
    """Convert parameter name to argument name"""
    return param_name.split('.')[-1]

def validate_args(args: argparse.Namespace, config: HyperParamConfig) -> None:
    """
    Validate command line arguments based on search group requirements.
    Reads parameters from previous search groups if available.
    
    Args:
        args: Command line arguments namespace
    Raises:
        ValueError: If required arguments for the specified search group are missing
    """

    for group_id in range(1, args.search_group):
        group = config.groups.get(group_id)
        if not group:
            continue

        # Get all parameter names in this group
        param_names = [param.name for param in group.params]

        # Try to read parameters from previous searched results
        params = read_searched_params(group_id, param_names)
        print("params", params)

        for param in group.params:
            arg_name = get_param_arg_name(param.name)
            arg_value = getattr(args, arg_name, None)

            # If the argument is not provided, try to read from searched params
            if arg_value is None and params:
                searched_value = params.get(param.name)
                if searched_value is not None:
                    setattr(args, arg_name, searched_value)
                    print(f"Read {arg_name} from searched params:", searched_value)
                else:
                    print(f"Warning: {arg_name} not found in searched params for group {group_id}")
            
            # If still None, raise an error
            if getattr(args, arg_name, None) is None:
                raise ValueError(f"--{arg_name} required for group {args.search_group}")

    # # Group 2 requirements
    # if args.search_group > 1:
    #     params = read_searched_params(1, ["training.ppo_batch_size"])
    #     if params and args.ppo_batch_size is None:
    #         args.ppo_batch_size = params.get('training.ppo_batch_size')
    #         print("Read ppo_batch_size from searched params:", args.ppo_batch_size)
    #     if args.ppo_batch_size is None:
    #         raise ValueError("--ppo_batch_size required for groups 2-5")

    # # Group 3 requirements
    # if args.search_group > 2:
    #     params = read_searched_params(2, [
    #         "training.train_batch_size",
    #         "training.n_rollout"
    #     ])
    #     if params:
    #         if args.train_batch_size is None:
    #             args.train_batch_size = params.get('training.train_batch_size')
    #             print("Read train_batch_size from searched params:", args.train_batch_size)
    #         if args.n_rollout is None:
    #             args.n_rollout = params.get('training.n_rollout')
    #             print("Read n_rollout from searched params:", args.n_rollout)
    #     if args.train_batch_size is None or args.n_rollout is None:
    #         raise ValueError("--train_batch_size and --n_rollout required for groups 3-5")

    # # Group 4 requirements
    # if args.search_group > 3:
    #     params = read_searched_params(3, ["training.kl_coef"])
    #     if params and args.kl_coef is None:
    #         args.kl_coef = params.get('training.kl_coef')
    #         print("Read kl_coef from searched params:", args.kl_coef)
    #     if args.kl_coef is None:
    #         raise ValueError("--kl_coef required for groups 4-5")

    # # Group 5 requirements
    # if args.search_group > 4:
    #     params = read_searched_params(4, [
    #         "training.max_turns",
    #         "training.temperature"
    #     ])
    #     if params:
    #         if args.max_turns is None:
    #             args.max_turns = params.get('training.max_turns')
    #             print("Read max_turns from searched params:", args.max_turns)
    #         if args.temperature is None:
    #             args.temperature = params.get('training.temperature')
    #             print("Read temperature from searched params:", args.temperature)
    #     if args.max_turns is None or args.temperature is None:
    #         raise ValueError("--max_turns and --temperature required for group 5")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create config and search objects
    config = HyperParamConfig()

    validate_args(args, config)
    print("=" * 80)
    
    
    search = HyperParamSearch(config)
    
    # Collect fixed values from arguments
    fixed_values = {}
    if args.ppo_batch_size is not None:
        fixed_values["training.ppo_batch_size"] = args.ppo_batch_size
    if args.train_batch_size is not None:
        fixed_values["training.train_batch_size"] = args.train_batch_size
    if args.n_rollout is not None:
        fixed_values["training.n_rollout"] = args.n_rollout
    if args.kl_coef is not None:
        fixed_values["optimization.kl_coef"] = args.kl_coef
    if args.max_turns is not None:
        fixed_values["training.max_turns"] = args.max_turns
    if args.temperature is not None:
        fixed_values["training.temperature"] = args.temperature
    if args.actor_lr is not None:
        fixed_values["optimization.actor_lr"] = args.actor_lr
    if args.n_gpus is not None:
        fixed_values["system.n_gpus"] = args.n_gpus
    if args.micro_batch_size is not None:
        fixed_values["training.micro_batch_size"] = args.micro_batch_size
        assert args.micro_batch_size >= args.n_gpus, "micro_batch_size must be greater than n_gpus"
    elif args.micro_batch_size is None and args.n_gpus is not None:
        assert 1 >= args.n_gpus, "micro_batch_size must be greater than n_gpus"    
    # Set up search configuration
    search.setup_search_group(args.search_group, fixed_values)
    
    # Run the search
    search.run_grid_search(
        base_experiment_name=args.base_experiment_name,
        dry_run=args.dry_run,
        env_name=args.env_name,
        parallel=args.parallel
    )

if __name__ == "__main__":
    main()
