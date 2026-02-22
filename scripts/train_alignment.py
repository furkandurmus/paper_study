#!/usr/bin/env python3
"""
Alignment training for CTA-MIP VLM.

Supports multiple alignment methods:
- DPO (Direct Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)

Uses TRL library for implementation.
"""

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments

from src.collator import PreferenceDataCollator
from src.config import AlignmentConfig, Config
from src.dataset import PreferenceDataset
from src.utils import load_model_and_processor, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Alignment training for CTA-MIP VLM")
    parser.add_argument(
        "--alignment_type",
        type=str,
        required=True,
        choices=["dpo", "kto", "orpo", "grpo"],
        help="Alignment method to use",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to SFT model checkpoint"
    )
    parser.add_argument(
        "--preference_data", type=str, required=True, help="Path to preference data JSONL"
    )
    parser.add_argument(
        "--images_root", type=str, required=True, help="Root directory for images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument(
        "--beta", type=float, default=0.1, help="Beta parameter for DPO/KTO/ORPO"
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable LoRA adapters",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument(
        "--max_images_per_case", type=int, default=4, help="Maximum images to load per case"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Comma-separated trackers (e.g. tensorboard,wandb) or 'none'",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="Attention backend for model loading",
    )
    return parser.parse_args()


def parse_report_to(report_to: str) -> List[str]:
    value = report_to.strip().lower()
    if value in {"", "none", "null", "false"}:
        return []
    return [item.strip() for item in report_to.split(",") if item.strip()]


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(bf16_supported()) if callable(bf16_supported) else False


def create_training_arguments(
    args: AlignmentConfig,
    output_dir: str,
    report_to: List[str],
    eval_dataset: Optional[Any],
) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_mode = "steps" if eval_dataset is not None else "no"
    bf16 = supports_bf16()
    fp16 = torch.cuda.is_available() and not bf16

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": report_to,
    }

    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = eval_mode
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = eval_mode

    if eval_dataset is not None:
        kwargs["eval_steps"] = 100

    return TrainingArguments(**kwargs)


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


def setup_dpo_trainer(
    model,
    ref_model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
):
    """Setup DPO trainer using trl.DPOTrainer."""
    try:
        from trl import DPOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(args, output_dir, report_to, eval_dataset)
    return build_trainer(
        trainer_cls=DPOTrainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        data_collator=data_collator,
        extra_kwargs={
            "ref_model": ref_model,
            "beta": args.beta,
            "label_smoothing": args.label_smoothing,
        },
    )


def setup_kto_trainer(
    model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
):
    """Setup KTO trainer using trl.KTOTrainer."""
    try:
        from trl import KTOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(args, output_dir, report_to, eval_dataset)
    return build_trainer(
        trainer_cls=KTOTrainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        data_collator=data_collator,
        extra_kwargs={
            "beta": args.beta,
            "desirable_weight": args.desirable_weight,
            "undesirable_weight": args.undesirable_weight,
        },
    )


def setup_orpo_trainer(
    model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
):
    """Setup ORPO trainer using trl.ORPOTrainer."""
    try:
        from trl import ORPOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(args, output_dir, report_to, eval_dataset)
    return build_trainer(
        trainer_cls=ORPOTrainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        data_collator=data_collator,
        extra_kwargs={
            "beta": args.beta,
            "orpo_alpha": args.orpo_alpha,
        },
    )


def setup_grpo_trainer(
    model,
    ref_model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    output_dir: str,
    report_to: List[str],
):
    """Setup GRPO trainer using trl.GRPOTrainer."""
    try:
        from trl import GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "TRL library with GRPO support not installed. Run: pip install trl"
        ) from exc

    training_args = create_training_arguments(args, output_dir, report_to, eval_dataset)

    def reward_func(prompts, completions, **kwargs):
        """Placeholder reward: replace with task-specific clinical scoring."""
        rewards = []
        for completion in completions:
            rewards.append(min(len(completion) / 100.0, 1.0))
        return rewards

    return build_trainer(
        trainer_cls=GRPOTrainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        extra_kwargs={
            "ref_model": ref_model,
            "reward_funcs": reward_func,
            "reward_func": reward_func,
        },
    )


def prepare_dataset_for_trl(dataset, format_type: str):
    """
    Convert dataset to TRL-compatible text-only format.

    TRL expects specific column names:
    - DPO/ORPO: prompt, chosen, rejected
    - KTO: prompt, completion, label (True for desirable, False for undesirable)
    - GRPO: prompt, completion
    """

    if hasattr(dataset, "data"):
        records = dataset.data
    else:
        records = [dataset[i] for i in range(len(dataset))]

    formatted_data = []
    for example in records:
        if format_type == "pairwise":
            formatted_data.append(
                {
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                }
            )
        elif format_type == "binary":
            formatted_data.append(
                {
                    "prompt": example["prompt"],
                    "completion": example["response"],
                    "label": example["preference"] == "good",
                }
            )
        elif format_type == "group":
            candidates = example.get("candidates", [])
            best_completion = ""
            if candidates:
                best = max(candidates, key=lambda x: x.get("score", 0))
                best_completion = best.get("response", "")
            formatted_data.append(
                {"prompt": example["prompt"], "completion": best_completion}
            )
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    return formatted_data


def load_with_attention_fallback(model_path: str, model_config: Dict[str, Any], logger):
    """Try requested attention backend, fallback if unavailable."""
    tried = []
    order = [
        model_config.get("attn_implementation", "flash_attention_2"),
        "sdpa",
        "eager",
    ]

    for attn_impl in order:
        if attn_impl in tried:
            continue
        tried.append(attn_impl)

        current_config = dict(model_config)
        current_config["attn_implementation"] = attn_impl
        try:
            model, processor = load_model_and_processor(
                model_name_or_path=model_path,
                model_config=current_config,
                use_qlora=False,
            )
            if attn_impl != model_config.get("attn_implementation"):
                logger.warning(
                    "Fell back to attention backend '%s'", attn_impl
                )
            return model, processor
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed loading with attn '%s': %s", attn_impl, exc)

    raise RuntimeError("Could not load model with any supported attention backend")


def main():
    args = parse_args()

    if not os.path.exists(args.preference_data):
        raise FileNotFoundError(f"Preference data not found: {args.preference_data}")
    if not os.path.exists(args.images_root):
        raise FileNotFoundError(f"Images root not found: {args.images_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_level="INFO", log_file=os.path.join(args.output_dir, "training.log"))
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Alignment Training: %s", args.alignment_type.upper())
    logger.info("=" * 60)

    set_seed(args.seed)
    report_to = parse_report_to(args.report_to)

    config = Config()
    if args.config and os.path.exists(args.config):
        config = Config.from_json(args.config)

    config.alignment.alignment_type = args.alignment_type
    config.alignment.output_dir = args.output_dir
    config.alignment.beta = args.beta
    config.alignment.num_train_epochs = args.num_epochs
    config.alignment.per_device_train_batch_size = args.batch_size
    config.alignment.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.alignment.learning_rate = args.learning_rate
    config.to_json(os.path.join(args.output_dir, "config.json"))

    logger.info("Loading model from %s", args.model_path)
    model_config = {
        "torch_dtype": "bfloat16",
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
    }

    model, processor = load_with_attention_fallback(args.model_path, model_config, logger)

    if args.use_lora:
        logger.info("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    ref_model = None
    if args.alignment_type in {"dpo", "grpo"}:
        logger.info("Loading reference model...")
        ref_model, _ = load_with_attention_fallback(args.model_path, model_config, logger)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    format_type = "pairwise"
    if args.alignment_type == "kto":
        format_type = "binary"
    elif args.alignment_type == "grpo":
        format_type = "group"

    logger.info("Loading preference data from %s", args.preference_data)
    raw_dataset = PreferenceDataset(
        jsonl_path=args.preference_data,
        images_root=args.images_root,
        processor=processor,
        max_images_per_case=args.max_images_per_case,
        format_type=format_type,
    )
    logger.info("Train dataset size: %d", len(raw_dataset))

    # Use vision-aware collator for pairwise methods to keep image conditioning.
    # KTO/GRPO stay text-only due TRL schema differences across versions.
    train_dataset = raw_dataset
    data_collator = None
    if args.alignment_type in {"dpo", "orpo"}:
        data_collator = PreferenceDataCollator(
            processor=processor,
            max_length=config.data.max_seq_length,
            format_type=format_type,
        )
        logger.info("Using vision-aware preference collator for %s", args.alignment_type)
    else:
        train_dataset = prepare_dataset_for_trl(raw_dataset, format_type)
        logger.warning(
            "%s is using text-only formatted dataset; image tensors are not consumed in this mode.",
            args.alignment_type.upper(),
        )

    logger.info("Setting up %s trainer...", args.alignment_type.upper())

    if args.alignment_type == "dpo":
        trainer = setup_dpo_trainer(
            model=model,
            ref_model=ref_model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=processor,
            args=config.alignment,
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=data_collator,
        )
    elif args.alignment_type == "kto":
        trainer = setup_kto_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=processor,
            args=config.alignment,
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=None,
        )
    elif args.alignment_type == "orpo":
        trainer = setup_orpo_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=processor,
            args=config.alignment,
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=data_collator,
        )
    elif args.alignment_type == "grpo":
        trainer = setup_grpo_trainer(
            model=model,
            ref_model=ref_model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=processor,
            args=config.alignment,
            output_dir=args.output_dir,
            report_to=report_to,
        )
    else:
        raise ValueError(f"Unknown alignment type: {args.alignment_type}")

    logger.info("Starting training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    logger.info("Saving final model to %s", final_dir)
    trainer.save_model(final_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(final_dir)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
