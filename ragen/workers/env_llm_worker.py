from verl.workers.fsdp_workers import *
from verl import DataProto
import torch
import logging
import warnings
import time

logger = logging.getLogger(__name__)

class EnvironmentLLMWorker(Worker):
    def __init__(self, config: DictConfig, role: str = 'env_llm'):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.role = role
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.fsdp_config.fsdp_size)

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        self.ulysses_device_mesh = None
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._is_offload_param = self.config.fsdp_config.get('param_offload', False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.model import update_model_config, get_generation_config, print_model_size
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload

        local_path = copy_to_local(self.config.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        self.generation_config = get_generation_config(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(self.config.model.get('override_config', {}))
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        torch_dtype = self.config.fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation='flash_attention_2',
                trust_remote_code=self.config.model.get('trust_remote_code', False)
            )
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)
            self.model.to(torch_dtype)

        if self.rank == 0:
            print_model_size(self.model)

        mixed_precision_config = self.config.fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(module=self.model, config=self.config.fsdp_config.get('wrap_policy', None))
        sharding_strategy = get_sharding_strategy(self.device_mesh)
        cpu_offload = CPUOffload(offload_params=True)

        self.model_fsdp = FSDP(
            self.model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False
        )

        self.model = self.model_fsdp._fsdp_wrapped_module
        self.use_vllm = self._init_vllm_engine(local_path, model_config)
        torch.distributed.barrier()

    def _init_vllm_engine(self, local_path, model_config):
        try:
            rollout_device_mesh = init_device_mesh(
                'cuda',
                mesh_shape=(torch.distributed.get_world_size() // self.config.get('vllm_config', {}).get('tensor_parallel_size', 1),
                            self.config.get('vllm_config', {}).get('tensor_parallel_size', 1)),
                mesh_dim_names=['dp', 'tp']
            )

            if self.config.use_fire_sampling:
                from verl.workers.rollout.vllm_rollout import FIREvLLMRollout as vLLMRollout
                from verl.workers.rollout.vllm_rollout import vllm_mode
            else:
                from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)

            if vllm_mode == 'customized':
                self.vllm_engine = vLLMRollout(
                    actor_module=self.model_fsdp,
                    config=self.config.generation,
                    tokenizer=self.tokenizer,
                    model_hf_config=model_config
                )
            elif vllm_mode == 'spmd':
                self.vllm_engine = vLLMRollout(
                    model_path=local_path,
                    config=self.config.generation,
                    tokenizer=self.tokenizer,
                    model_hf_config=model_config,
                    device_mesh=rollout_device_mesh
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            if torch.distributed.get_world_size() == 1:
                self.config.generation.load_format = 'dummy_hf'

            self.vllm_sharding_manager = FSDPVLLMShardingManager(
                module=self.model_fsdp,
                inference_engine=self.vllm_engine.inference_engine,
                model_config=model_config,
                full_params='hf' in self.config.generation.load_format,
                device_mesh=rollout_device_mesh
            )
            return True
        except ImportError:
            logger.warning("vLLM not available, falling back to FSDP implementation.")
            return False

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_responses(self, prompts: DataProto):
        prompts = prompts.to(torch.cuda.current_device())
        if not self.use_vllm and self._is_offload_param:
            load_fsdp_model_to_gpu(self.model_fsdp)

        with (self.vllm_sharding_manager if self.use_vllm else self.ulysses_sharding_manager):
            manager = self.vllm_sharding_manager if self.use_vllm else self.ulysses_sharding_manager
            prompts = manager.preprocess_data(prompts)
            output = self.vllm_engine.generate_sequences(prompts) if self.use_vllm else self._generate_with_fsdp(prompts)
            output = manager.postprocess_data(output)

            if not self.use_vllm and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.model_fsdp)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    def _generate_with_fsdp(self, prompts: DataProto):
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_length=self.config.generation.get('max_length', 512),
            temperature=self.config.generation.get('temperature', 0.7),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch.get('attention_mask', None)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return DataProto.from_dict(tensors={'generated_texts': generated_texts})
