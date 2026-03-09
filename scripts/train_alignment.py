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
import contextlib
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


class TextOnlyVisionProcessorShim:
    """
    Minimal shim for TRL DPOTrainer vision preprocessing path.
    Accepts `images=` but tokenizes only text using an underlying tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        text = kwargs.pop("text", None)
        kwargs.pop("images", None)
        if text is None and args:
            return self.tokenizer(*args, **kwargs)
        return self.tokenizer(text, **kwargs)


@contextlib.contextmanager
def force_trl_dpo_text_mode_for_model(model):
    """
    TRL DPOTrainer (newer versions) auto-detects vision-language models and requires
    multimodal datasets/processors. For smoke tests we may intentionally pass a
    text-only fallback dataset. This context temporarily disables the vision branch
    for the current model_type during trainer initialization.
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
    parser.add_argument(
        "--reference_free",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reference-free DPO smoke testing (skips loading ref model)",
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
        "--dpo_max_prompt_length",
        type=int,
        default=2048,
        help="Max prompt length for TRL DPO tokenization (important for multimodal prompts)",
    )
    parser.add_argument(
        "--dpo_max_length",
        type=int,
        default=3072,
        help="Max total length for TRL DPO concatenated prompt+completion",
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


def ensure_trl_fsdp_compat():
    """
    TRL>=0.29 imports `torch.distributed.fsdp.FSDPModule`, which is not exposed
    in some torch builds (e.g. 2.5.x). Provide a best-effort alias.
    """
    try:
        import torch.distributed.fsdp as fsdp_mod
    except Exception:
        return

    if hasattr(fsdp_mod, "FSDPModule"):
        return
    if hasattr(fsdp_mod, "FullyShardedDataParallel"):
        fsdp_mod.FSDPModule = fsdp_mod.FullyShardedDataParallel


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
    args_cls=TrainingArguments,
) -> TrainingArguments:
    sig = inspect.signature(args_cls.__init__).parameters
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

    return args_cls(**kwargs)


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
    dpo_max_prompt_length: Optional[int] = None,
    dpo_max_length: Optional[int] = None,
):
    """Setup DPO trainer using trl.DPOTrainer."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError("TRL library not installed. Run: pip install trl") from exc

    training_args = create_training_arguments(
        args, output_dir, report_to, eval_dataset, args_cls=DPOConfig
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

    set_seed(args.seed)
    report_to = parse_report_to(args.report_to)

    config = Config()
    if args.config and os.path.exists(args.config):
        config = Config.from_json(args.config)

    config.alignment.alignment_type = args.alignment_type
    config.alignment.output_dir = args.output_dir
    config.alignment.beta = args.beta
    config.alignment.reference_free = args.reference_free
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
    if args.alignment_type in {"dpo", "grpo"} and not (
        args.alignment_type == "dpo" and args.reference_free
    ):
        logger.info("Loading reference model...")
        ref_model, _ = load_with_attention_fallback(args.model_path, model_config, logger)
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
        max_images_per_case=args.max_images_per_case,
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
            output_dir=args.output_dir,
            report_to=report_to,
            data_collator=data_collator,
            dpo_max_prompt_length=args.dpo_max_prompt_length,
            dpo_max_length=args.dpo_max_length,
        )
    elif args.alignment_type == "kto":
        trainer = setup_kto_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            processor=trainer_processing,
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
            processor=trainer_processing,
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
            processor=trainer_processing,
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
