#!/usr/bin/env python3
"""
Generate candidate responses for preference data creation.

This script generates N candidate responses per training example
using the SFT model with different sampling parameters.

Usage:
    python scripts/generate_candidates.py \
        --model_path outputs/sft/final \
        --train_jsonl data/train.jsonl \
        --images_root data/images \
        --output_path outputs/candidates.jsonl \
        --num_candidates 4
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import InferenceDataset
from src.collator import InferenceCollator
from src.utils import setup_logging, load_model_and_processor, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate candidate responses for preference data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained SFT model"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Root directory for images"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for candidates JSONL"
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=4,
        help="Number of candidates to generate per example"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature_range",
        type=str,
        default="0.5,0.7,0.9,1.0",
        help="Comma-separated temperatures for diversity"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_reference",
        action="store_true",
        help="Include reference response as one candidate"
    )
    return parser.parse_args()


def generate_candidates_for_example(
    model,
    processor,
    example: Dict[str, Any],
    num_candidates: int,
    max_new_tokens: int,
    temperatures: List[float],
    top_p: float,
    device: str
) -> List[Dict[str, Any]]:
    """Generate multiple candidates for a single example."""
    model.eval()
    candidates = []
    
    images = example["images"]
    prompt = example["prompt"]
    
    # Build input
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    
    conversation = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Flatten images for processing
    flattened_images = images
    
    inputs = processor(
        text=[text],
        images=flattened_images if flattened_images else None,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    
    with torch.no_grad():
        for i, temp in enumerate(temperatures[:num_candidates]):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            # Decode
            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            candidates.append({
                "text": generated_text,
                "temperature": temp,
                "top_p": top_p,
                "index": i
            })
    
    return candidates


def main():
    args = parse_args()
    
    # Parse temperatures
    temperatures = [float(t) for t in args.temperature_range.split(",")]
    
    # Setup logging
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting candidate generation...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Generating {args.num_candidates} candidates per example")
    
    # Set seed
    set_seed(args.seed)
    
    # Load model and processor
    logger.info("Loading model and processor...")
    
    is_peft = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    model_config = {
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "gradient_checkpointing": False
    }
    
    if is_peft:
        from peft import PeftModel
        model, processor = load_model_and_processor(
            model_name_or_path=args.model_path,
            model_config=model_config,
            use_qlora=False
        )
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        model, processor = load_model_and_processor(
            model_name_or_path=args.model_path,
            model_config=model_config,
            use_qlora=False
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = InferenceDataset(
        jsonl_path=args.train_jsonl,
        images_root=args.images_root,
        processor=processor,
        max_images_per_case=4
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Load reference responses if needed
    reference_responses = {}
    if args.use_reference:
        logger.info("Loading reference responses...")
        with open(args.train_jsonl, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                reference_responses[item["case_id"]] = item["response"]
    
    # Generate candidates
    logger.info("Generating candidates...")
    all_results = []
    
    for i in tqdm(range(len(dataset)), desc="Processing examples"):
        example = dataset[i]
        case_id = example["case_id"]
        
        # Generate candidates
        candidates = generate_candidates_for_example(
            model=model,
            processor=processor,
            example=example,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            temperatures=temperatures,
            top_p=args.top_p,
            device=device
        )
        
        # Add reference if requested
        if args.use_reference and case_id in reference_responses:
            candidates.append({
                "text": reference_responses[case_id],
                "is_reference": True,
                "index": len(candidates)
            })
        
        result = {
            "case_id": case_id,
            "images": example["image_paths"],
            "prompt": example["prompt"],
            "labels": example.get("labels", {}),
            "candidates": candidates
        }
        
        all_results.append(result)
    
    # Save results
    logger.info(f"Saving candidates to {args.output_path}")
    with open(args.output_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Generated candidates for {len(all_results)} examples")
    logger.info(f"Total candidates: {sum(len(r['candidates']) for r in all_results)}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
