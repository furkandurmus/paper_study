#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for CTA-MIP VLM.

Usage:
    python scripts/train_sft.py --config configs/sft_config.json --output_dir outputs/sft
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    Trainer,
)

from src.config import Config
from src.dataset import CTAMIPDataset
from src.collator import SFTDataCollator
from src.training_runtime import (
    apply_lora_if_enabled,
    configure_torch_runtime,
    create_training_arguments,
    ensure_trl_fsdp_compat,
    filter_available_reporters,
    load_model_with_attention_fallback,
    parse_report_to,
    resolve_distributed_config_for_model,
)
from src.utils import (
    setup_logging,
    set_seed,
    create_output_directory
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTA-MIP VLM with SFT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default=None,
        help="Override train JSONL path from config"
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default=None,
        help="Override validation JSONL path from config"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Override images root directory from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_trl_fsdp_compat()
    
    # Load config
    if os.path.exists(args.config):
        config = Config.from_json(args.config)
    else:
        logging.warning(f"Config file not found: {args.config}, using defaults")
        config = Config()
    
    # Override config with CLI args
    if args.model_name:
        config.model.model_name_or_path = args.model_name
    if args.train_jsonl:
        config.data.train_jsonl = args.train_jsonl
    if args.val_jsonl:
        config.data.val_jsonl = args.val_jsonl
    if args.images_root:
        config.data.images_root = args.images_root
    if args.seed is not None:
        config.training.seed = args.seed
    
    # Setup output directory
    output_dir = create_output_directory(args.output_dir, "sft")
    config.to_json(os.path.join(output_dir, "config.json"))
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file=os.path.join(output_dir, "training.log")
    )
    logger = logging.getLogger(__name__)
    configure_torch_runtime(config.distributed, logger)
    
    # Set seed
    set_seed(config.training.seed)
    logger.info(f"Random seed: {config.training.seed}")
    
    # Log config
    logger.info(f"Model: {config.model.model_name_or_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model and processor
    logger.info("Loading model and processor...")
    model, processor = load_model_with_attention_fallback(
        model_name_or_path=config.model.model_name_or_path,
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
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = CTAMIPDataset(
        jsonl_path=config.data.train_jsonl,
        images_root=config.data.images_root,
        processor=processor,
        max_seq_length=config.data.max_seq_length,
        max_images_per_case=config.data.max_images_per_case,
        split="train"
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    eval_dataset = None
    if config.data.val_jsonl and os.path.exists(config.data.val_jsonl):
        eval_dataset = CTAMIPDataset(
            jsonl_path=config.data.val_jsonl,
            images_root=config.data.images_root,
            processor=processor,
            max_seq_length=config.data.max_seq_length,
            max_images_per_case=config.data.max_images_per_case,
            split="val"
        )
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = SFTDataCollator(
        processor=processor,
        max_length=config.data.max_seq_length
    )
    
    # Setup training arguments
    report_to = filter_available_reporters(
        parse_report_to(config.training.report_to),
        logger=logger,
    )

    training_args = create_training_arguments(
        train_config=config.training,
        distributed=distributed_config,
        output_dir=output_dir,
        report_to=report_to,
        eval_dataset=eval_dataset,
        extra_kwargs={
            "load_best_model_at_end": (
                config.training.load_best_model_at_end and eval_dataset is not None
            ),
            "metric_for_best_model": config.training.metric_for_best_model,
            "greater_is_better": config.training.greater_is_better,
            "dataloader_prefetch_factor": (
                2 if config.training.dataloader_num_workers > 0 else None
            ),
        },
    )
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))
    
    # Save training state
    trainer.save_state()
    
    logger.info(f"Training complete! Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
