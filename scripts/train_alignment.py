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
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.collator import PreferenceDataCollator
from src.config import AlignmentConfig, Config
from src.dataset import PreferenceDataset
from src.training_runtime import (
    apply_lora_if_enabled,
    build_trainer,
    configure_torch_runtime,
    create_training_arguments,
    ensure_trl_fsdp_compat,
    filter_available_reporters,
    force_trl_dpo_text_mode_for_model,
    load_model_with_attention_fallback,
    parse_report_to,
    resolve_distributed_config_for_model,
)
from src.utils import set_seed, setup_logging


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
        "--beta", type=float, default=None, help="Beta parameter for DPO/KTO/ORPO"
    )
    parser.add_argument(
        "--reference_free",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reference-free DPO smoke testing (skips loading ref model)",
    )
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable LoRA adapters",
    )
    parser.add_argument(
        "--use_qlora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable 4-bit QLoRA loading",
    )
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument(
        "--max_images_per_case", type=int, default=None, help="Maximum images to load per case"
    )
    parser.add_argument(
        "--dpo_max_prompt_length",
        type=int,
        default=None,
        help="Max prompt length for TRL DPO tokenization (important for multimodal prompts)",
    )
    parser.add_argument(
        "--dpo_max_length",
        type=int,
        default=None,
        help="Max total length for TRL DPO concatenated prompt+completion",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
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


def setup_dpo_trainer(
    model,
    ref_model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    distributed,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
    dpo_max_prompt_length: Optional[int] = None,
    dpo_max_length: Optional[int] = None,
):
    """Setup DPO trainer using trl.DPOTrainer."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(
        train_config=args,
        distributed=distributed,
        output_dir=output_dir,
        report_to=report_to,
        eval_dataset=eval_dataset,
        args_cls=DPOConfig,
    )
    if dpo_max_prompt_length is not None and hasattr(training_args, "max_prompt_length"):
        training_args.max_prompt_length = dpo_max_prompt_length
    if dpo_max_length is not None and hasattr(training_args, "max_length"):
        training_args.max_length = dpo_max_length
    build_kwargs = dict(
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
            "reference_free": getattr(args, "reference_free", False),
        },
    )
    if data_collator is None and not hasattr(processor, "tokenizer"):
        with force_trl_dpo_text_mode_for_model(model):
            return build_trainer(**build_kwargs)
    return build_trainer(**build_kwargs)


def setup_kto_trainer(
    model,
    train_dataset,
    eval_dataset,
    processor,
    args: AlignmentConfig,
    distributed,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
):
    """Setup KTO trainer using trl.KTOTrainer."""
    try:
        from trl import KTOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(
        train_config=args,
        distributed=distributed,
        output_dir=output_dir,
        report_to=report_to,
        eval_dataset=eval_dataset,
    )
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
    distributed,
    output_dir: str,
    report_to: List[str],
    data_collator: Optional[Any] = None,
):
    """Setup ORPO trainer using trl.ORPOTrainer."""
    try:
        from trl import ORPOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(
        train_config=args,
        distributed=distributed,
        output_dir=output_dir,
        report_to=report_to,
        eval_dataset=eval_dataset,
    )
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
    distributed,
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

    training_args = create_training_arguments(
        train_config=args,
        distributed=distributed,
        output_dir=output_dir,
        report_to=report_to,
        eval_dataset=eval_dataset,
    )

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


def prepare_multimodal_pairwise_hf_dataset(raw_dataset, processor=None):
    """
    Build a Hugging Face Dataset for TRL multimodal DPO/ORPO.

    Output columns:
    - prompt
    - chosen
    - rejected
    - images (list of decodable image paths)
    """
    try:
        from datasets import Dataset as HFDataset
        from datasets import Features, Image, Sequence, Value
    except ImportError as exc:
        raise ImportError(
            "datasets library is required for multimodal DPO/ORPO with this TRL version. "
            "Run: pip install datasets"
        ) from exc

    rows = []
    images_root = Path(raw_dataset.images_root)
    for item in raw_dataset.data:
        image_paths = item.get("images", [])[: raw_dataset.max_images_per_case]
        prompt_text = item["prompt"]
        if processor is not None and hasattr(processor, "apply_chat_template"):
            try:
                content = [{"type": "image"} for _ in image_paths]
                content.append({"type": "text", "text": item["prompt"]})
                prompt_text = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = item["prompt"]
        rows.append(
            {
                "prompt": prompt_text,
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "images": [str((images_root / p).resolve()) for p in image_paths],
            }
        )

    features = Features(
        {
            "prompt": Value("string"),
            "chosen": Value("string"),
            "rejected": Value("string"),
            "images": Sequence(Image()),
        }
    )
    return HFDataset.from_list(rows, features=features)


def main():
    args = parse_args()
    ensure_trl_fsdp_compat()

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

    config = Config()
    if args.config and os.path.exists(args.config):
        config = Config.from_json(args.config)
    configure_torch_runtime(config.distributed, logger)
    set_seed(args.seed if args.seed is not None else config.alignment.seed)
    report_to = filter_available_reporters(
        parse_report_to(
            args.report_to
            if args.report_to is not None
            else config.alignment.report_to
        ),
        logger=logger,
    )

    config.alignment.alignment_type = args.alignment_type
    config.alignment.output_dir = args.output_dir
    if args.beta is not None:
        config.alignment.beta = args.beta
    config.alignment.reference_free = args.reference_free
    if args.num_epochs is not None:
        config.alignment.num_train_epochs = args.num_epochs
    if args.batch_size is not None:
        config.alignment.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.alignment.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config.alignment.learning_rate = args.learning_rate
    config.alignment.report_to = report_to
    if args.seed is not None:
        config.alignment.seed = args.seed
    config.model.model_name_or_path = args.model_path
    config.model.attn_implementation = args.attn_implementation
    if args.use_lora is not None:
        config.lora.use_lora = args.use_lora
    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.lora_alpha = args.lora_alpha
    if args.use_qlora is not None:
        config.model.use_qlora = args.use_qlora
    config.to_json(os.path.join(args.output_dir, "config.json"))

    logger.info("Loading model from %s", args.model_path)
    model, processor = load_model_with_attention_fallback(
        model_name_or_path=args.model_path,
        model_config=config.model,
        use_qlora=config.model.use_qlora,
        logger=logger,
    )
    model = apply_lora_if_enabled(model, config.lora, logger)
    distributed_config = resolve_distributed_config_for_model(
        model,
        config.distributed,
        logger=logger,
    )

    ref_model = None
    if args.alignment_type in {"dpo", "grpo"} and not (
        args.alignment_type == "dpo" and args.reference_free
    ):
        logger.info("Loading reference model...")
        ref_model, _ = load_model_with_attention_fallback(
            model_name_or_path=args.model_path,
            model_config=config.model,
            use_qlora=False,
            logger=logger,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    elif args.alignment_type == "dpo" and args.reference_free:
        logger.warning("Reference-free DPO enabled: skipping reference model load")

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
        max_images_per_case=(
            args.max_images_per_case
            if args.max_images_per_case is not None
            else config.data.max_images_per_case
        ),
        format_type=format_type,
    )
    logger.info("Train dataset size: %d", len(raw_dataset))

    # Use vision-aware collator for pairwise methods to keep image conditioning.
    # KTO/GRPO stay text-only due TRL schema differences across versions.
    train_dataset = raw_dataset
    data_collator = None
    trainer_processing = processor
    if args.alignment_type in {"dpo", "orpo"}:
        # Prefer a real multimodal HF dataset for newer TRL versions (uses `.map()` and processor(images=...)).
        # Fall back to legacy custom-collator path if multimodal dataset creation is unavailable.
        if format_type == "pairwise":
            try:
                train_dataset = prepare_multimodal_pairwise_hf_dataset(raw_dataset, processor=processor)
                data_collator = None
                trainer_processing = processor
                logger.info(
                    "Using TRL multimodal HF dataset for %s (images enabled)",
                    args.alignment_type.upper(),
                )
            except Exception as exc:
                logger.warning(
                    "Could not build multimodal HF dataset (%s). Falling back to legacy collator path.",
                    exc,
                )
                data_collator = PreferenceDataCollator(
                    processor=processor,
                    max_length=config.data.max_seq_length,
                    format_type=format_type,
                )
                logger.info("Using vision-aware preference collator for %s", args.alignment_type)
        else:
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
            processor=trainer_processing,
            args=config.alignment,
            distributed=distributed_config,
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=data_collator,
            dpo_max_prompt_length=(
                args.dpo_max_prompt_length
                if args.dpo_max_prompt_length is not None
                else config.data.max_prompt_length
            ),
            dpo_max_length=(
                args.dpo_max_length
                if args.dpo_max_length is not None
                else config.data.max_seq_length
            ),
        )
    elif args.alignment_type == "kto":
        trainer = setup_kto_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=trainer_processing,
            args=config.alignment,
            distributed=distributed_config,
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=None,
        )
    elif args.alignment_type == "orpo":
        trainer = setup_orpo_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=trainer_processing,
            args=config.alignment,
            distributed=distributed_config,
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
            processor=trainer_processing,
            args=config.alignment,
            distributed=distributed_config,
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
