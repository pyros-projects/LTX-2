import os
import re
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200
FIRST_STEP_TRACE_LIMIT = 25


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


def should_keep_quantized_model_on_device(
    world_size: str | None = None,
    cuda_available: bool | None = None,
) -> bool:
    """Decide whether single-process startup should keep quantized weights on the target device."""
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    if world_size is None:
        world_size = os.environ.get("WORLD_SIZE", "1")

    try:
        process_count = int(world_size)
    except (TypeError, ValueError):
        process_count = 1

    return cuda_available and process_count <= 1


def get_base_transformer_model(transformer: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying transformer model when wrapped by PEFT."""
    return transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer


def normalize_external_lora_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Normalize external LoRA checkpoints to PEFT-compatible key names."""
    return {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}


def extract_lora_target_modules(state_dict: dict[str, Tensor]) -> list[str]:
    """Extract full target module paths from LoRA A/B keys."""
    pattern = re.compile(r"(.+)\.lora_[AB]\.weight$")
    target_modules = {
        match.group(1)
        for key in state_dict
        if (match := pattern.match(key)) is not None
    }
    if not target_modules:
        raise ValueError("Could not extract target modules from LoRA state dict")
    return sorted(target_modules)


def build_lora_config_from_state_dict(state_dict: dict[str, Tensor]) -> LoraConfig:
    """Build a PEFT LoRA config from an external checkpoint with possibly dynamic ranks/alphas."""
    target_modules = extract_lora_target_modules(state_dict)
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for key, value in state_dict.items():
        if key.endswith(".lora_A.weight") and value.ndim == 2:
            module_name = key[: -len(".lora_A.weight")]
            rank_pattern[module_name] = int(value.shape[0])
        elif key.endswith(".alpha"):
            module_name = key[: -len(".alpha")]
            alpha_pattern[module_name] = int(float(value.item()))

    if not rank_pattern:
        raise ValueError("Could not infer LoRA rank pattern from state dict")

    for module_name, rank in rank_pattern.items():
        alpha_pattern.setdefault(module_name, rank)

    default_rank = max(rank_pattern.values())
    default_alpha = max(alpha_pattern.values()) if alpha_pattern else default_rank

    return LoraConfig(
        r=default_rank,
        lora_alpha=default_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )


def summarize_external_lora_state_dict(state_dict: dict[str, Tensor]) -> dict[str, object]:
    """Return a compact summary useful for debug logging of external LoRA checkpoints."""
    target_modules = extract_lora_target_modules(state_dict)
    rank_values = [
        int(value.shape[0])
        for key, value in state_dict.items()
        if key.endswith(".lora_A.weight") and value.ndim == 2
    ]
    alpha_values = [int(float(value.item())) for key, value in state_dict.items() if key.endswith(".alpha")]
    tensor_dtype_counts = Counter(str(value.dtype) for value in state_dict.values() if isinstance(value, torch.Tensor))
    tensor_device_counts = Counter(str(value.device) for value in state_dict.values() if isinstance(value, torch.Tensor))

    return {
        "target_module_count": len(target_modules),
        "rank_range": (min(rank_values), max(rank_values)) if rank_values else None,
        "alpha_range": (min(alpha_values), max(alpha_values)) if alpha_values else None,
        "tensor_dtype_counts": dict(tensor_dtype_counts),
        "tensor_device_counts": dict(tensor_device_counts),
    }


def should_enable_block_swap(
    distributed_type: DistributedType,
    training_mode: str,
    blocks_to_swap: int,
) -> bool:
    """Whether block swap should be enabled for the current run."""
    return distributed_type == DistributedType.NO and training_mode == "lora" and blocks_to_swap > 0


def enable_block_swap_for_training(
    transformer: torch.nn.Module,
    device: torch.device,
    blocks_to_swap: int,
    use_pinned_memory: bool,
) -> None:
    """Enable block swap on the base transformer and prepare initial residency."""
    base_transformer = get_base_transformer_model(transformer)
    base_transformer.enable_block_swap(
        blocks_to_swap=blocks_to_swap,
        device=device,
        supports_backward=True,
        use_pinned_memory=use_pinned_memory,
    )
    base_transformer.move_to_device_except_swap_blocks(device)
    base_transformer.prepare_block_swap_before_forward()


def restore_block_swap_residency(transformer: torch.nn.Module, device: torch.device) -> None:
    """Restore intended swap residency after wrapping or device placement."""
    base_transformer = get_base_transformer_model(transformer)
    if getattr(base_transformer, "blocks_to_swap", 0) <= 0:
        return

    base_transformer.move_to_device_except_swap_blocks(device)
    base_transformer.prepare_block_swap_before_forward()


def collect_expected_swap_cpu_module_prefixes(transformer: torch.nn.Module) -> set[str]:
    """Return module-name prefixes that are expected to stay on CPU due to block swap."""
    base_transformer = get_base_transformer_model(transformer)
    blocks_to_swap = int(getattr(base_transformer, "blocks_to_swap", 0) or 0)
    transformer_blocks = getattr(base_transformer, "transformer_blocks", None)

    if blocks_to_swap <= 0 or transformer_blocks is None:
        return set()

    total_blocks = len(transformer_blocks)
    start_idx = max(total_blocks - blocks_to_swap, 0)
    return {f"transformer_blocks.{idx}" for idx in range(start_idx, total_blocks)}


def filter_unexpected_cpu_modules(cpu_modules: list[str], expected_prefixes: set[str]) -> list[str]:
    """Filter out CPU modules that are expected because of configured block swapping."""
    if not expected_prefixes:
        return list(cpu_modules)

    return [
        module_name
        for module_name in cpu_modules
        if not any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in expected_prefixes)
    ]


def collect_module_device_summary(module: torch.nn.Module) -> dict[str, object]:
    """Collect parameter/buffer device distribution and modules with CPU state."""
    parameter_devices = Counter(str(param.device) for _, param in module.named_parameters())
    buffer_devices = Counter(str(buf.device) for _, buf in module.named_buffers())

    cpu_parameter_modules = sorted(
        {
            name.rsplit(".", 1)[0] if "." in name else ""
            for name, param in module.named_parameters()
            if param.device.type == "cpu"
        }
    )
    cpu_buffer_modules = sorted(
        {
            name.rsplit(".", 1)[0] if "." in name else ""
            for name, buf in module.named_buffers()
            if buf.device.type == "cpu"
        }
    )

    return {
        "parameter_devices": dict(parameter_devices),
        "buffer_devices": dict(buffer_devices),
        "cpu_parameter_modules": cpu_parameter_modules,
        "cpu_buffer_modules": cpu_buffer_modules,
    }


def compute_step_timing(elapsed_seconds: float, gradient_accumulation_steps: int) -> tuple[float, float]:
    """Return (microstep_seconds, optimization_step_seconds)."""
    return elapsed_seconds, elapsed_seconds * gradient_accumulation_steps


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._validation_sampling_lora_path: str | None = None
        self._cached_validation_embeddings = self._time_init_stage(
            "cache_validation_embeddings",
            self._load_text_encoder_and_cache_embeddings,
        )
        self._time_init_stage("load_models", self._load_models)
        self._time_init_stage("setup_accelerator", self._setup_accelerator)
        self._time_init_stage("collect_trainable_params", self._collect_trainable_params)
        self._time_init_stage("load_checkpoint", self._load_checkpoint)
        self._time_init_stage("prepare_models_for_training", self._prepare_models_for_training)
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._did_trace_first_step = False
        self._time_init_stage("init_wandb", self._init_wandb)

    def _time_init_stage(self, name: str, fn: Callable[[], object]) -> object:
        """Run an initialization stage with detailed timing logs."""
        logger.debug(f"INIT START: {name}")
        start_time = time.time()
        result = fn()
        elapsed = time.time() - start_time
        logger.debug(f"INIT DONE: {name} in {elapsed:.2f}s")
        return result

    def _log_transformer_device_summary(self, stage: str) -> None:
        """Log exact parameter/buffer device placement for the current transformer."""
        summary = collect_module_device_summary(self._transformer)
        expected_swap_prefixes = collect_expected_swap_cpu_module_prefixes(self._transformer)
        logger.debug(
            f"DEVICE SUMMARY [{stage}] params={summary['parameter_devices']} "
            f"buffers={summary['buffer_devices']}"
        )

        cpu_param_modules = summary["cpu_parameter_modules"]
        cpu_buffer_modules = summary["cpu_buffer_modules"]
        if cpu_param_modules:
            logger.debug(
                f"DEVICE SUMMARY [{stage}] CPU parameter modules "
                f"(showing up to {FIRST_STEP_TRACE_LIMIT}): {cpu_param_modules[:FIRST_STEP_TRACE_LIMIT]}"
            )
            unexpected_cpu_params = filter_unexpected_cpu_modules(cpu_param_modules, expected_swap_prefixes)
            if unexpected_cpu_params:
                logger.debug(
                    f"DEVICE SUMMARY [{stage}] unexpected CPU parameter modules "
                    f"(showing up to {FIRST_STEP_TRACE_LIMIT}): {unexpected_cpu_params[:FIRST_STEP_TRACE_LIMIT]}"
                )
        if cpu_buffer_modules:
            logger.debug(
                f"DEVICE SUMMARY [{stage}] CPU buffer modules "
                f"(showing up to {FIRST_STEP_TRACE_LIMIT}): {cpu_buffer_modules[:FIRST_STEP_TRACE_LIMIT]}"
            )
            unexpected_cpu_buffers = filter_unexpected_cpu_modules(cpu_buffer_modules, expected_swap_prefixes)
            if unexpected_cpu_buffers:
                logger.debug(
                    f"DEVICE SUMMARY [{stage}] unexpected CPU buffer modules "
                    f"(showing up to {FIRST_STEP_TRACE_LIMIT}): {unexpected_cpu_buffers[:FIRST_STEP_TRACE_LIMIT]}"
                )

    def _trace_first_training_step(self, batch: dict[str, dict[str, Tensor]]) -> None:  # noqa: PLR0915
        """Trace the first step to identify any modules that actually execute on CPU."""
        if self._did_trace_first_step:
            return

        self._did_trace_first_step = True
        transformer = get_base_transformer_model(self._transformer)
        expected_swap_prefixes = collect_expected_swap_cpu_module_prefixes(self._transformer)
        trace_records: dict[str, dict[str, object]] = {}
        hooks = []

        def extract_tensor_devices(obj: object) -> set[str]:
            devices: set[str] = set()
            if isinstance(obj, torch.Tensor):
                devices.add(str(obj.device))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    devices.update(extract_tensor_devices(item))
            elif isinstance(obj, dict):
                for item in obj.values():
                    devices.update(extract_tensor_devices(item))
            return devices

        def add_hooks() -> None:
            for name, module in transformer.named_modules():
                if name == "":
                    continue

                def pre_hook(mod, args, module_name=name) -> None:  # noqa: ANN001
                    record = trace_records.setdefault(
                        module_name,
                        {
                            "input_devices": set(),
                            "param_devices": set(),
                            "buffer_devices": set(),
                            "elapsed_seconds": 0.0,
                            "calls": 0,
                        },
                    )
                    record["input_devices"].update(extract_tensor_devices(args))
                    record["param_devices"].update(str(p.device) for p in mod.parameters(recurse=False))
                    record["buffer_devices"].update(str(b.device) for b in mod.buffers(recurse=False))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    record["_start_time"] = time.perf_counter()

                def post_hook(mod, args, output, module_name=name) -> None:  # noqa: ANN001, ARG001
                    record = trace_records[module_name]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = record.pop("_start_time", None)
                    if start_time is not None:
                        record["elapsed_seconds"] += time.perf_counter() - start_time
                    record["calls"] += 1
                    record["output_devices"] = sorted(extract_tensor_devices(output))

                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(post_hook))

        logger.debug("FIRST STEP TRACE: registering module hooks")
        add_hooks()
        try:
            self._training_step(batch)
        finally:
            for hook in hooks:
                hook.remove()

        cpu_records = []
        unexpected_cpu_records = []
        for module_name, record in trace_records.items():
            input_devices = sorted(record.get("input_devices", set()))
            param_devices = sorted(record.get("param_devices", set()))
            buffer_devices = sorted(record.get("buffer_devices", set()))
            output_devices = record.get("output_devices", [])
            if any(
                device.startswith("cpu")
                for device in (*input_devices, *param_devices, *buffer_devices, *output_devices)
            ):
                cpu_records.append((module_name, record))
                if not filter_unexpected_cpu_modules([module_name], expected_swap_prefixes):
                    continue
                unexpected_cpu_records.append((module_name, record))

        logger.debug(
            f"FIRST STEP TRACE: executed {len(trace_records)} modules, "
            f"cpu_involved={len(cpu_records)} unexpected_cpu_involved={len(unexpected_cpu_records)}"
        )
        for module_name, record in unexpected_cpu_records[:FIRST_STEP_TRACE_LIMIT]:
            logger.debug(
                "FIRST STEP TRACE CPU "
                f"{module_name}: inputs={sorted(record.get('input_devices', set()))} "
                f"params={sorted(record.get('param_devices', set()))} "
                f"buffers={sorted(record.get('buffer_devices', set()))} "
                f"outputs={record.get('output_devices', [])} "
                f"calls={record.get('calls', 0)} "
                f"time={record.get('elapsed_seconds', 0.0):.4f}s"
            )

        slowest = sorted(
            trace_records.items(),
            key=lambda item: float(item[1].get("elapsed_seconds", 0.0)),
            reverse=True,
        )[:FIRST_STEP_TRACE_LIMIT]
        for module_name, record in slowest:
            logger.debug(
                "FIRST STEP TRACE SLOW "
                f"{module_name}: inputs={sorted(record.get('input_devices', set()))} "
                f"params={sorted(record.get('param_devices', set()))} "
                f"outputs={record.get('output_devices', [])} "
                f"calls={record.get('calls', 0)} "
                f"time={record.get('elapsed_seconds', 0.0):.4f}s"
            )

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._time_init_stage("init_optimizer", self._init_optimizer)
        self._time_init_stage("init_dataloader", self._init_dataloader)
        data_iter = iter(self._dataloader)
        self._time_init_stage("init_timestep_sampler", self._init_timestep_sampler)

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = 0

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            # Initial validation before training starts
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            self._accelerator.wait_for_everyone()

            for step in range(cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    if step == 0:
                        self._trace_first_training_step(batch)

                    loss = self._training_step(batch)
                    self._accelerator.backward(loss)

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            # FSDP: All processes must participate in validation
                            sampled_videos_paths = self._sample_videos(progress)
                            if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
                        # DDP: Only main process runs validation
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    microstep_time, optimization_step_time = compute_step_timing(
                        elapsed_seconds=time.time() - step_start_time,
                        gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
                    )

                    progress.update_training(
                        loss=loss.item(),
                        lr=current_lr,
                        step_time=optimization_step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        self._log_metrics(
                            {
                                "train/loss": loss.item(),
                                "train/learning_rate": current_lr,
                                "train/step_time": optimization_step_time,
                                "train/microstep_time": microstep_time,
                                "train/global_step": self._global_step,
                            }
                        )

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                            f"Microstep: {microstep_time:.2f}s, "
                            f"OptStep: {optimization_step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = cfg.optimization.steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """Perform a single training step using the configured strategy."""
        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]

        if "video_prompt_embeds" in conditions:
            # New format: separate video/audio features from precompute()
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            # Legacy format: single prompt_embeds tensor — duplicate for both modalities
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )

        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        return loss

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder + embeddings processor, compute and cache validation embeddings."""

        # This method:
        #   1. Loads the pure Gemma text encoder on GPU
        #   2. Loads the embeddings processor (feature extractor + connectors)
        #   3. If validation prompts are configured, computes and caches their embeddings
        #   4. Unloads the Gemma model entirely, keeps the embeddings processor for training

        # Load text encoder (pure Gemma LLM) on GPU
        logger.debug("Loading text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=self._config.model.text_encoder_path,
            device="cuda",
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )
        logger.info(
            "Validation embedding cache: text encoder loaded "
            f"(8bit={self._config.acceleration.load_text_encoder_in_8bit})"
        )

        # Load embeddings processor (feature extractor + connectors)
        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device="cuda",
            dtype=torch.bfloat16,
        )
        logger.info("Validation embedding cache: embeddings processor loaded on cuda with bf16")

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            logger.info(f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts...")
            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    pos_hs, pos_mask = text_encoder.encode(prompt)
                    pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)

                    neg_hs, neg_mask = text_encoder.encode(self._config.validation.negative_prompt)
                    neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=pos_out.video_encoding.cpu(),
                            audio_context_positive=pos_out.audio_encoding.cpu(),
                            video_context_negative=neg_out.video_encoding.cpu(),
                            audio_context_negative=(
                                neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                            ),
                        )
                    )
                if cached_embeddings:
                    first_cached = cached_embeddings[0]
                    logger.info(
                        "Validation embedding cache summary: "
                        f"video_pos_shape={tuple(first_cached.video_context_positive.shape)} "
                        f"video_pos_dtype={first_cached.video_context_positive.dtype} "
                        f"audio_pos_shape={tuple(first_cached.audio_context_positive.shape)} "
                        f"audio_pos_dtype={first_cached.audio_context_positive.dtype}"
                    )

        # Unload Gemma model and feature extractor, keep only connectors for training
        del text_encoder
        self._embeddings_processor.feature_extractor = None

        logger.debug("Validation prompt embeddings cached. Gemma model unloaded")
        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Load all model components (except text encoder - already handled)
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._embeddings_processor was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
                keep_quantized_model_on_device=should_keep_quantized_model_on_device(),
            )
            self._log_transformer_device_summary("after_quantize")

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _ensure_validation_sampling_lora_adapter_loaded(self, transformer: torch.nn.Module) -> str | None:
        """Load the validation-only sampling LoRA as a non-trainable PEFT adapter when configured."""
        lora_path = getattr(self._config.validation, "sampling_lora_weight", None)
        if lora_path is None:
            return None

        if self._config.model.training_mode != "lora":
            raise ValueError("validation.sampling_lora_weight is currently supported only for LoRA training mode.")
        if not hasattr(transformer, "add_adapter") or not hasattr(transformer, "peft_config"):
            raise TypeError("Validation sampling LoRA requires a PEFT-wrapped transformer.")

        adapter_name = "__validation_sampling__"
        current_path = str(lora_path)
        loaded_path = getattr(self, "_validation_sampling_lora_path", None)
        logger.info(
            "Preparing validation sampling LoRA adapter load: "
            f"path={current_path} transformer_type={type(transformer).__name__} "
            f"active_adapter={getattr(transformer, 'active_adapter', None)} "
            f"loaded_path={loaded_path}"
        )

        if adapter_name in transformer.peft_config and loaded_path != current_path:
            transformer.delete_adapter(adapter_name)

        if adapter_name not in transformer.peft_config or loaded_path != current_path:
            load_start_time = time.perf_counter()
            logger.info(f"Validation sampling LoRA: loading safetensors from {current_path}")
            raw_state_dict = load_file(str(lora_path))
            logger.info(
                f"Validation sampling LoRA: load_file completed in "
                f"{time.perf_counter() - load_start_time:.2f}s with {len(raw_state_dict)} tensors"
            )

            normalize_start_time = time.perf_counter()
            state_dict = normalize_external_lora_state_dict(raw_state_dict)
            logger.info(
                "Validation sampling LoRA: state dict normalized in "
                f"{time.perf_counter() - normalize_start_time:.2f}s"
            )

            config_start_time = time.perf_counter()
            lora_config = build_lora_config_from_state_dict(state_dict)
            state_summary = summarize_external_lora_state_dict(state_dict)
            logger.info(
                "Validation sampling LoRA: PEFT config built in "
                f"{time.perf_counter() - config_start_time:.2f}s "
                f"(targets={state_summary['target_module_count']} rank_range={state_summary['rank_range']} "
                f"alpha_range={state_summary['alpha_range']})"
            )

            adapter_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".alpha")}
            logger.info(
                "Validation sampling LoRA: adapter state prepared "
                f"(tensor_count={len(adapter_state_dict)} dtypes={state_summary['tensor_dtype_counts']} "
                f"devices={state_summary['tensor_device_counts']})"
            )

            add_adapter_start_time = time.perf_counter()
            logger.info(f"Validation sampling LoRA: calling add_adapter({adapter_name})")
            transformer.add_adapter(adapter_name, lora_config)
            logger.info(
                f"Validation sampling LoRA: add_adapter completed in "
                f"{time.perf_counter() - add_adapter_start_time:.2f}s"
            )

            state_apply_start_time = time.perf_counter()
            logger.info(f"Validation sampling LoRA: calling set_peft_model_state_dict({adapter_name})")
            load_result = set_peft_model_state_dict(transformer, adapter_state_dict, adapter_name=adapter_name)
            logger.info(
                "Validation sampling LoRA: set_peft_model_state_dict completed in "
                f"{time.perf_counter() - state_apply_start_time:.2f}s"
            )
            missing_keys = list(getattr(load_result, "missing_keys", []) or [])
            unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Validation sampling LoRA: state load reported "
                    f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)} "
                    f"sample_missing={missing_keys[:5]} sample_unexpected={unexpected_keys[:5]}"
                )
            else:
                logger.info("Validation sampling LoRA: state load matched all injected adapter weights")

            freeze_start_time = time.perf_counter()
            transformer.set_requires_grad(adapter_name, False)
            logger.info(
                f"Validation sampling LoRA: set_requires_grad(False) completed in "
                f"{time.perf_counter() - freeze_start_time:.2f}s"
            )
            self._validation_sampling_lora_path = current_path
            logger.info(
                "Validation sampling LoRA loaded: "
                f"path={current_path} target_modules={state_summary['target_module_count']} "
                f"rank_range={state_summary['rank_range']} alpha_range={state_summary['alpha_range']} "
                f"tensor_dtypes={state_summary['tensor_dtype_counts']} "
                f"tensor_devices={state_summary['tensor_device_counts']}"
            )

        return adapter_name

    @contextmanager
    def _validation_sampling_lora_scope(self, transformer: torch.nn.Module):
        """Temporarily activate the training and sampling LoRAs together without materializing a merged adapter."""
        sampling_lora_path = getattr(self._config.validation, "sampling_lora_weight", None)
        if sampling_lora_path is None:
            yield
            return

        sampling_adapter = self._ensure_validation_sampling_lora_adapter_loaded(transformer)
        if sampling_adapter is None:
            yield
            return

        active_adapter = getattr(transformer, "active_adapter", "default")
        active_adapters = list(active_adapter) if isinstance(active_adapter, list) else [active_adapter]
        combined_adapters = [*active_adapters]
        if sampling_adapter not in combined_adapters:
            combined_adapters.append(sampling_adapter)
        multiplier = float(getattr(self._config.validation, "sampling_lora_multiplier", 1.0))

        logger.info(
            "Activating validation sampling LoRAs together: "
            f"train_adapters={active_adapters} sampling_adapter={sampling_adapter} multiplier={multiplier}"
        )

        scaled_modules: list[tuple[object, float]] = []
        peft_model = getattr(getattr(transformer, "base_model", None), "model", transformer)
        for module in peft_model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and sampling_adapter in scaling:
                original_scaling = float(scaling[sampling_adapter])
                scaling[sampling_adapter] = original_scaling * multiplier
                scaled_modules.append((module, original_scaling))

        logger.info(
            "Validation sampling LoRA: scaled sampling adapter "
            f"across {len(scaled_modules)} modules"
        )

        set_active_start_time = time.perf_counter()
        transformer.base_model.set_adapter(combined_adapters, inference_mode=True)
        logger.info(
            f"Validation sampling LoRA: set_adapter({combined_adapters}) completed in "
            f"{time.perf_counter() - set_active_start_time:.2f}s"
        )
        logger.debug(
            "Validation sampling adapter state: "
            f"top_level_active={getattr(transformer, 'active_adapter', None)} "
            f"base_active={combined_adapters} "
            f"available={sorted(getattr(transformer, 'peft_config', {}).keys())}"
        )

        try:
            yield
        finally:
            for module, original_scaling in scaled_modules:
                module.scaling[sampling_adapter] = original_scaling
            transformer.base_model.set_adapter(active_adapter, inference_mode=False)
            transformer.delete_adapter(sampling_adapter)
            self._validation_sampling_lora_path = None
            free_gpu_memory()
            logger.debug(
                "Validation sampling adapter restored: "
                f"top_level_active={getattr(transformer, 'active_adapter', None)} "
                f"base_active={active_adapter} "
                f"available={sorted(getattr(transformer, 'peft_config', {}).keys())}"
            )

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""
        logger.debug("Preparing models for training with Accelerate")

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = get_base_transformer_model(self._transformer)

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        if self._config.acceleration.blocks_to_swap > 0 and self._accelerator.distributed_type != DistributedType.NO:
            raise ValueError("Block swapping is currently supported only for single-GPU runs.")

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        if should_enable_block_swap(
            self._accelerator.distributed_type,
            self._config.model.training_mode,
            self._config.acceleration.blocks_to_swap,
        ):
            enable_block_swap_for_training(
                self._transformer,
                self._accelerator.device,
                self._config.acceleration.blocks_to_swap,
                self._config.acceleration.use_pinned_memory_for_block_swap,
            )
            self._log_transformer_device_summary("after_enable_block_swap")

        # For single-GPU quantized runs, quantize_model() can keep the quantized transformer on
        # the target CUDA device already. Avoid forcing a second full-model device transfer here.
        if self._accelerator.distributed_type == DistributedType.NO and (
            self._config.acceleration.quantization or self._config.acceleration.blocks_to_swap > 0
        ):
            self._log_transformer_device_summary("before_prepare")

        # noinspection PyTypeChecker
        logger.debug("Calling Accelerator.prepare() for transformer")
        self._transformer = self._accelerator.prepare(self._transformer)
        logger.debug("Accelerator.prepare() completed for transformer")
        self._log_transformer_device_summary("after_prepare")

        if should_enable_block_swap(
            self._accelerator.distributed_type,
            self._config.model.training_mode,
            self._config.acceleration.blocks_to_swap,
        ):
            restore_block_swap_residency(self._transformer, self._accelerator.device)
            self._log_transformer_device_summary("after_restore_block_swap")

        # Log GPU memory usage after model preparation
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        smi_gb = get_gpu_memory_gb(self._accelerator.device)
        logger.debug(
            "GPU memory usage after models preparation: "
            f"allocated={allocated_gb:.2f} GB reserved={reserved_gb:.2f} GB nvidia-smi={smi_gb:.2f} GB"
        )

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        logger.debug("DATALOADER: initialization started")
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            logger.debug(f"DATALOADER: building PrecomputedDataset from {self._config.data.preprocessed_data_root}")
            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        logger.debug(
            "DATALOADER: creating torch DataLoader "
            f"(batch_size={self._config.optimization.batch_size}, num_workers={num_workers})"
        )
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        logger.debug("DATALOADER: calling Accelerator.prepare()")
        self._dataloader = self._accelerator.prepare(dataloader)
        logger.debug("DATALOADER: Accelerator.prepare() completed")

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        logger.debug("OPTIMIZER: initialization started")
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        extra_args = dict(opt_cfg.optimizer_args)
        if opt_cfg.optimizer_type == "adamw":
            logger.debug("OPTIMIZER: constructing AdamW")
            optimizer = AdamW(self._trainable_params, lr=lr, **extra_args)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            logger.debug("OPTIMIZER: constructing AdamW8bit")
            optimizer = AdamW8bit(self._trainable_params, lr=lr, **extra_args)
        elif opt_cfg.optimizer_type == "adafactor":
            from transformers.optimization import Adafactor  # noqa: PLC0415

            logger.debug("OPTIMIZER: constructing Adafactor")
            # The trainer always supplies a manual learning rate, so default Adafactor
            # into the compatible non-relative-step mode unless the user overrides it.
            extra_args.setdefault("relative_step", False)
            extra_args.setdefault("scale_parameter", False)
            optimizer = Adafactor(self._trainable_params, lr=lr, **extra_args)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        logger.debug(f"OPTIMIZER: creating scheduler of type {opt_cfg.scheduler_type}")
        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        logger.debug("OPTIMIZER: calling Accelerator.prepare()")
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)
        logger.debug("OPTIMIZER: Accelerator.prepare() completed")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        sample_sigmas = self._config.validation.sample_sigmas
        inference_steps = (
            len(sample_sigmas) - 1 if sample_sigmas is not None else self._config.validation.inference_steps
        )
        logger.info(
            "Starting validation sampling: "
            f"prompts={len(self._config.validation.prompts)} steps={inference_steps} "
            f"custom_sigmas={'yes' if sample_sigmas is not None else 'no'} "
            f"sampling_lora={'yes' if self._config.validation.sampling_lora_weight is not None else 'no'}"
        )
        if sample_sigmas is not None:
            logger.info(f"Validation sampling sigmas: {sample_sigmas}")

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()
        self._log_transformer_device_summary("before_validation_sampling")

        # Start sampling progress tracking
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=inference_steps,
        )

        # Create validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims
        sampling_transformer = (
            self._accelerator.unwrap_model(self._transformer)
            if hasattr(self._accelerator, "unwrap_model")
            else self._transformer
        )

        with self._validation_sampling_lora_scope(sampling_transformer):
            for prompt_idx, prompt in enumerate(self._config.validation.prompts):
                # Update progress to show current video
                sampling_ctx.start_video(prompt_idx)
                logger.info(
                    f"Validation prompt {prompt_idx + 1}/{len(self._config.validation.prompts)} "
                    f"(cached_embeddings={'yes' if self._cached_validation_embeddings is not None else 'no'})"
                )

                # Load conditioning image if provided
                condition_image = None
                if use_images:
                    image_path = self._config.validation.images[prompt_idx]
                    image = open_image_as_srgb(image_path)
                    # Convert PIL image to tensor [C, H, W] in [0, 1]
                    condition_image = F.to_tensor(image)

                # Load reference video if provided (for IC-LoRA)
                reference_video = None
                if use_reference_videos:
                    ref_video_path = self._config.validation.reference_videos[prompt_idx]
                    # read_video returns [F, C, H, W] in [0, 1]
                    reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

                # Get cached embeddings for this prompt if available
                cached_embeddings = (
                    self._cached_validation_embeddings[prompt_idx]
                    if self._cached_validation_embeddings is not None
                    else None
                )

                # Create generation config
                gen_config = GenerationConfig(
                    prompt=prompt,
                    negative_prompt=self._config.validation.negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=self._config.validation.frame_rate,
                    num_inference_steps=inference_steps,
                    guidance_scale=self._config.validation.guidance_scale,
                    sample_sigmas=sample_sigmas,
                    seed=self._config.validation.seed,
                    condition_image=condition_image,
                    reference_video=reference_video,
                    reference_downscale_factor=self._config.validation.reference_downscale_factor,
                    generate_audio=generate_audio,
                    include_reference_in_output=self._config.validation.include_reference_in_output,
                    cached_embeddings=cached_embeddings,
                    stg_scale=self._config.validation.stg_scale,
                    stg_blocks=self._config.validation.stg_blocks,
                    stg_mode=self._config.validation.stg_mode,
                )

                # Generate sample
                prompt_start_time = time.perf_counter()
                video, audio = sampler.generate(
                    config=gen_config,
                    device=self._accelerator.device,
                )
                logger.info(
                    f"Validation prompt {prompt_idx + 1} sampling finished in "
                    f"{time.perf_counter() - prompt_start_time:.2f}s"
                )

                # Save output (image for single frame, video otherwise)
                if IS_MAIN_PROCESS:
                    ext = "png" if num_frames == 1 else "mp4"
                    output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
                    if num_frames == 1:
                        save_image(video, output_path)
                    else:
                        save_video(
                            video_tensor=video,
                            output_path=output_path,
                            fps=self._config.validation.frame_rate,
                            audio=audio,
                            audio_sample_rate=self._vocoder.output_sampling_rate if audio is not None else None,
                        )
                    video_paths.append(output_path)

        # Clean up progress tasks
        sampling_ctx.cleanup()
        self._log_transformer_device_summary("after_validation_sampling")

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

            # Build metadata for safetensors file
            metadata = self._build_checkpoint_metadata()

            # Save to disk with metadata
            save_file(state_dict, saved_weights_path, metadata=metadata)
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoints: {old_checkpoint}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")
        media_cls = wandb.Image if is_image else wandb.Video

        samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
