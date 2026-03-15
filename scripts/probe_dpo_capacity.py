#!/usr/bin/env python3
"""
Probe feasible DPO batch size / image count combinations on the current GPU.

Runs a single real training step for each combination and reports:
- success or CUDA OOM/failure
- peak allocated and reserved GPU memory

This is intended to answer: "How many images per case and what batch size fit?"
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_alignment import (  # noqa: E402
    prepare_multimodal_pairwise_hf_dataset,
    setup_dpo_trainer,
)
from src.config import Config  # noqa: E402
from src.dataset import PreferenceDataset  # noqa: E402
from src.training_runtime import (  # noqa: E402
    ensure_trl_fsdp_compat,
    load_model_with_attention_fallback,
    parse_report_to,
)
from src.utils import set_seed, setup_logging  # noqa: E402


def parse_int_list(value: str) -> List[int]:
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(int(raw))
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="Probe DPO capacity on the current GPU")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preference_data", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/dpo_capacity_probe")
    parser.add_argument("--batch_sizes", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--image_counts", type=parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--dpo_max_prompt_length", type=int, default=1600)
    parser.add_argument("--dpo_max_length", type=int, default=2200)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    return parser.parse_args()


def format_gib(value_bytes: int) -> str:
    return f"{value_bytes / (1024 ** 3):.2f} GiB"


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = parse_args()
    ensure_trl_fsdp_compat()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_level="INFO", log_file=os.path.join(args.output_dir, "probe.log"))
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Capacity probing only makes sense on a GPU.")

    set_seed(args.seed)
    config = Config()
    config.alignment.reference_free = True
    config.alignment.num_train_epochs = 1
    config.alignment.gradient_accumulation_steps = 1
    config.alignment.learning_rate = args.learning_rate
    config.alignment.beta = 0.1
    report_to = parse_report_to("none")

    config.model.model_name_or_path = args.model_path
    config.model.attn_implementation = args.attn_implementation

    logger.info("Loading model from %s", args.model_path)
    model, processor = load_model_with_attention_fallback(
        model_name_or_path=args.model_path,
        model_config=config.model,
        use_qlora=False,
        logger=logger,
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        logger.info("Applying LoRA adapters for probe")
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

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {device_name}")
    print(f"Total VRAM: {format_gib(total_vram)}")
    print("")
    print("results:")

    for image_count in args.image_counts:
        for batch_size in args.batch_sizes:
            trainer = None
            dataloader = None
            batch = None
            train_dataset = None
            raw_dataset = None
            cleanup_cuda()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            config.alignment.per_device_train_batch_size = batch_size

            try:
                raw_dataset = PreferenceDataset(
                    jsonl_path=args.preference_data,
                    images_root=args.images_root,
                    processor=processor,
                    max_images_per_case=image_count,
                    format_type="pairwise",
                )
                train_dataset = prepare_multimodal_pairwise_hf_dataset(raw_dataset, processor=processor)
                trainer = setup_dpo_trainer(
                    model=model,
                    ref_model=None,
                    train_dataset=train_dataset,
                    eval_dataset=None,
                    processor=processor,
                    args=config.alignment,
                    distributed=config.distributed,
                    output_dir=os.path.join(
                        args.output_dir, f"bs{batch_size}_img{image_count}"
                    ),
                    report_to=report_to,
                    data_collator=None,
                    dpo_max_prompt_length=args.dpo_max_prompt_length,
                    dpo_max_length=args.dpo_max_length,
                )
                dataloader = trainer.get_train_dataloader()
                batch = next(iter(dataloader))
                trainer.model.train()
                trainer.training_step(trainer.model, batch, num_items_in_batch=None)
                cleanup_cuda()
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                print(
                    f"PASS batch_size={batch_size} max_images_per_case={image_count} "
                    f"peak_alloc={format_gib(peak_alloc)} peak_reserved={format_gib(peak_reserved)}"
                )
            except torch.cuda.OutOfMemoryError:
                cleanup_cuda()
                print(
                    f"OOM  batch_size={batch_size} max_images_per_case={image_count}"
                )
            except RuntimeError as exc:
                cleanup_cuda()
                message = str(exc).strip().splitlines()[0]
                if "out of memory" in message.lower():
                    print(
                        f"OOM  batch_size={batch_size} max_images_per_case={image_count}"
                    )
                else:
                    print(
                        f"FAIL batch_size={batch_size} max_images_per_case={image_count} error={message}"
                    )
            finally:
                # Drop references before the next probe iteration.
                trainer = None
                dataloader = None
                batch = None
                train_dataset = None
                raw_dataset = None
                cleanup_cuda()


if __name__ == "__main__":
    main()
