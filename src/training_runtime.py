"""
Shared training runtime utilities for SFT and alignment.
"""

import contextlib
import copy
import importlib.util
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments

from src.config import DistributedConfig, LoRAConfig, ModelConfig
from src.utils import load_model_and_processor


def ensure_trl_fsdp_compat():
    """
    TRL>=0.29 imports `torch.distributed.fsdp.FSDPModule`, which is not exposed
    in some torch builds. Provide a best-effort alias.
    """
    try:
        import torch.distributed.fsdp as fsdp_mod
    except Exception:
        return

    if hasattr(fsdp_mod, "FSDPModule"):
        return
    if hasattr(fsdp_mod, "FullyShardedDataParallel"):
        fsdp_mod.FSDPModule = fsdp_mod.FullyShardedDataParallel


def parse_report_to(report_to: Any) -> List[str]:
    if isinstance(report_to, list):
        return [str(item).strip() for item in report_to if str(item).strip()]
    value = str(report_to).strip().lower()
    if value in {"", "none", "null", "false"}:
        return []
    return [item.strip() for item in str(report_to).split(",") if item.strip()]


def filter_available_reporters(report_to: List[str], logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Drop reporting integrations that are not installed so training can still start.
    """
    requirements = {
        "tensorboard": "tensorboard",
        "wandb": "wandb",
    }
    filtered = []
    for reporter in report_to:
        module_name = requirements.get(reporter)
        if module_name and importlib.util.find_spec(module_name) is None:
            if logger is not None:
                logger.warning(
                    "Reporting integration '%s' is not installed; removing it from report_to.",
                    reporter,
                )
            continue
        filtered.append(reporter)
    return filtered


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(bf16_supported()) if callable(bf16_supported) else False


def configure_torch_runtime(distributed: DistributedConfig, logger: Optional[logging.Logger] = None):
    """
    Apply runtime settings that make distributed/container training more robust.
    """
    if distributed.strategy in {"ddp", "fsdp"}:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
            if logger is not None:
                logger.info("Using torch multiprocessing sharing strategy: file_system")
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not set torch sharing strategy: %s", exc)


def build_model_load_kwargs(model_config: ModelConfig) -> Dict[str, Any]:
    return {
        "torch_dtype": model_config.torch_dtype,
        "attn_implementation": model_config.attn_implementation,
        "trust_remote_code": model_config.trust_remote_code,
        "gradient_checkpointing": True,
        "bnb_4bit_compute_dtype": model_config.bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": model_config.bnb_4bit_use_double_quant,
        "bnb_4bit_quant_type": model_config.bnb_4bit_quant_type,
    }


def load_model_with_attention_fallback(
    model_name_or_path: str,
    model_config: ModelConfig,
    use_qlora: bool,
    logger: logging.Logger,
):
    """Try requested attention backend, then fallback to sdpa/eager."""
    if use_qlora and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "QLoRA was requested but bitsandbytes is not installed. "
            "Install it with: pip install -U 'bitsandbytes>=0.46.1'"
        )

    tried = []
    order = [
        model_config.attn_implementation,
        "sdpa",
        "eager",
    ]
    errors = []

    for attn_impl in order:
        if attn_impl in tried:
            continue
        tried.append(attn_impl)

        current_config = build_model_load_kwargs(model_config)
        current_config["attn_implementation"] = attn_impl
        try:
            model, processor = load_model_and_processor(
                model_name_or_path=model_name_or_path,
                model_config=current_config,
                use_qlora=use_qlora,
            )
            if attn_impl != model_config.attn_implementation:
                logger.warning("Fell back to attention backend '%s'", attn_impl)
            return model, processor
        except Exception as exc:
            errors.append(f"{attn_impl}: {exc}")
            logger.warning("Failed loading with attn '%s': %s", attn_impl, exc)

    raise RuntimeError(
        "Could not load model with any supported attention backend. "
        + " | ".join(errors)
    )


def apply_lora_if_enabled(model, lora_config: LoRAConfig, logger: logging.Logger):
    if not lora_config.use_lora:
        return model

    logger.info("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=getattr(TaskType, lora_config.task_type, TaskType.CAUSAL_LM),
        target_modules=lora_config.target_modules,
        modules_to_save=lora_config.modules_to_save,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_fsdp_kwargs(distributed: DistributedConfig) -> Dict[str, Any]:
    if distributed.strategy != "fsdp":
        return {
            "ddp_find_unused_parameters": distributed.ddp_find_unused_parameters,
        }

    fsdp_modes = distributed.fsdp or ["full_shard", "auto_wrap"]
    fsdp_config = {
        "min_num_params": distributed.fsdp_min_num_params,
        "backward_prefetch": distributed.fsdp_backward_prefetch,
        "forward_prefetch": distributed.fsdp_forward_prefetch,
        "cpu_ram_efficient_loading": distributed.fsdp_cpu_ram_efficient_loading,
        "offload_params": distributed.fsdp_offload_params,
        "sync_module_states": distributed.fsdp_sync_module_states,
        "use_orig_params": distributed.fsdp_use_orig_params,
        "activation_checkpointing": distributed.fsdp_activation_checkpointing,
    }
    if distributed.fsdp_transformer_layer_cls_to_wrap:
        fsdp_config["transformer_layer_cls_to_wrap"] = (
            distributed.fsdp_transformer_layer_cls_to_wrap
        )
    return {
        "fsdp": fsdp_modes,
        "fsdp_config": fsdp_config,
        "ddp_find_unused_parameters": distributed.ddp_find_unused_parameters,
    }


def _infer_fsdp_layer_classes(model) -> List[str]:
    """
    Infer likely transformer block classes for HF FSDP auto-wrap.
    Prefer decoder/text layers over generic vision blocks when available.
    """
    class_names = sorted({type(module).__name__ for module in model.modules()})
    preferred_patterns = (
        "DecoderLayer",
        "TransformerBlock",
        "EncoderLayer",
        "Layer",
        "Block",
    )

    preferred = []
    for name in class_names:
        if any(pattern in name for pattern in preferred_patterns):
            preferred.append(name)

    decoder_like = [
        name for name in preferred
        if "DecoderLayer" in name or name.startswith(("Qwen", "Llama", "Mistral", "Phi"))
    ]
    return decoder_like or preferred


def resolve_distributed_config_for_model(
    model,
    distributed: DistributedConfig,
    logger: Optional[logging.Logger] = None,
) -> DistributedConfig:
    """
    Ensure FSDP wrap classes match the actual model.
    """
    resolved = copy.deepcopy(distributed)
    if resolved.strategy != "fsdp":
        return resolved

    available = {type(module).__name__ for module in model.modules()}
    requested = list(resolved.fsdp_transformer_layer_cls_to_wrap)

    if requested and all(name in available for name in requested):
        return resolved

    inferred = _infer_fsdp_layer_classes(model)
    if not inferred:
        return resolved

    if requested and logger is not None:
        logger.warning(
            "FSDP wrap classes %s not found in model. Replacing with inferred classes %s.",
            requested,
            inferred,
        )
    elif logger is not None:
        logger.info("Using inferred FSDP wrap classes: %s", inferred)

    resolved.fsdp_transformer_layer_cls_to_wrap = inferred
    return resolved


def create_training_arguments(
    train_config: Any,
    distributed: DistributedConfig,
    output_dir: str,
    report_to: List[str],
    eval_dataset: Optional[Any],
    args_cls=TrainingArguments,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> TrainingArguments:
    sig = inspect.signature(args_cls.__init__).parameters
    eval_mode = "steps" if eval_dataset is not None else "no"
    use_bf16 = supports_bf16() and getattr(train_config, "bf16", True)
    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
        and getattr(train_config, "fp16", False)
    )
    dataloader_num_workers = getattr(train_config, "dataloader_num_workers", 0)
    if distributed.strategy in {"ddp", "fsdp"} and dataloader_num_workers > 0:
        # In containers, each rank spawning workers often exhausts /dev/shm quickly.
        dataloader_num_workers = 0

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": train_config.num_train_epochs,
        "per_device_train_batch_size": train_config.per_device_train_batch_size,
        "per_device_eval_batch_size": getattr(
            train_config, "per_device_eval_batch_size", train_config.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "learning_rate": train_config.learning_rate,
        "weight_decay": getattr(train_config, "weight_decay", 0.0),
        "warmup_ratio": getattr(train_config, "warmup_ratio", 0.0),
        "lr_scheduler_type": getattr(train_config, "lr_scheduler_type", "cosine"),
        "logging_steps": train_config.logging_steps,
        "save_strategy": "steps",
        "save_steps": train_config.save_steps,
        "save_total_limit": getattr(train_config, "save_total_limit", 3),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": getattr(train_config, "gradient_checkpointing", True),
        "dataloader_num_workers": dataloader_num_workers,
        "remove_unused_columns": getattr(train_config, "remove_unused_columns", False),
        "report_to": report_to,
        "seed": getattr(train_config, "seed", 42),
        "save_safetensors": True,
    }

    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = eval_mode
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = eval_mode

    if eval_dataset is not None:
        kwargs["eval_steps"] = getattr(train_config, "eval_steps", 100)

    filtered_kwargs = {}
    for key, value in kwargs.items():
        if key in sig:
            filtered_kwargs[key] = value

    for key, value in build_fsdp_kwargs(distributed).items():
        if value is not None and key in sig:
            filtered_kwargs[key] = value

    for key, value in (extra_kwargs or {}).items():
        if value is not None and key in sig:
            filtered_kwargs[key] = value

    return args_cls(**filtered_kwargs)


@contextlib.contextmanager
def force_trl_dpo_text_mode_for_model(model):
    """
    TRL DPOTrainer auto-detects vision models; this fallback disables that branch.
    """
    try:
        import trl.trainer.dpo_trainer as dpo_mod
    except Exception:
        yield
        return

    mapping = getattr(dpo_mod, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", None)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if mapping is None or model_type is None or model_type not in mapping:
        yield
        return

    original = dict(mapping)
    try:
        mapping.pop(model_type, None)
        yield
    finally:
        mapping.clear()
        mapping.update(original)


def build_trainer(
    trainer_cls,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    processor,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    data_collator: Optional[Any] = None,
):
    sig = inspect.signature(trainer_cls.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if eval_dataset is not None and "eval_dataset" in sig:
        kwargs["eval_dataset"] = eval_dataset
    elif "eval_dataset" in sig:
        kwargs["eval_dataset"] = None

    if "processing_class" in sig:
        kwargs["processing_class"] = processor
    elif "tokenizer" in sig:
        kwargs["tokenizer"] = getattr(processor, "tokenizer", processor)

    if data_collator is not None and "data_collator" in sig:
        kwargs["data_collator"] = data_collator

    for key, value in (extra_kwargs or {}).items():
        if value is not None and key in sig:
            kwargs[key] = value

    return trainer_cls(**kwargs)
