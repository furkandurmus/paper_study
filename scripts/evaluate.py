#!/usr/bin/env python3
"""
Evaluation script for CTA-MIP VLM.

Computes automatic metrics (BLEU, ROUGE) and clinical rubric scores.
Outputs per-case evaluation JSON with model predictions and scores.

Usage:
    python scripts/evaluate.py \
        --model_path outputs/sft/final \
        --test_jsonl data/test.jsonl \
        --images_root data/images \
        --output_dir outputs/evaluation
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

from src.config import Config, EvaluationConfig
from src.dataset import CTAMIPDataset, InferenceDataset
from src.collator import InferenceCollator
from src.utils import setup_logging, load_model_and_processor
from src.scoring import (
    compute_bleu,
    compute_rouge,
    compute_structural_metrics,
    ClinicalRubricScorer,
    aggregate_scores
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CTA-MIP VLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Root directory for images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation"
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM-as-judge for evaluation"
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default=None,
        help="Model to use as judge"
    )
    return parser.parse_args()


def generate_predictions(
    model,
    processor,
    dataloader: DataLoader,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    do_sample: bool,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Generate predictions for all examples in dataloader."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            case_ids = batch.pop("case_ids")
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Prepare image inputs if present
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            # Decode
            # Remove input prompt from output
            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            for case_id, generated in zip(case_ids, generated_texts):
                predictions.append({
                    "case_id": case_id,
                    "prediction": generated.strip()
                })
    
    return predictions


def evaluate_example(
    prediction: str,
    reference: str,
    clinical_scorer: ClinicalRubricScorer
) -> Dict[str, Any]:
    """Evaluate a single example."""
    results = {}
    
    # Automatic metrics
    results["bleu"] = compute_bleu(reference, prediction)
    results["rouge"] = compute_rouge(reference, prediction)
    
    # Structural metrics
    results["structural"] = compute_structural_metrics(prediction)
    
    # Clinical rubrics
    results["clinical"] = clinical_scorer.score(prediction, reference)
    
    return results


def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file=os.path.join(args.output_dir, "evaluation.log")
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evaluation...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test data: {args.test_jsonl}")
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        config = Config.from_json(args.config)
        logger.info(f"Loaded config from {args.config}")
    
    # Load model and processor
    logger.info("Loading model and processor...")
    
    # Determine if model is PEFT
    is_peft = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    # Load base model config
    model_config = {
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "gradient_checkpointing": False
    }
    
    if is_peft:
        # Load base model first, then PEFT weights
        from peft import PeftModel
        base_model_path = args.model_path  # PEFT config has base_model_name_or_path
        
        model, processor = load_model_and_processor(
            model_name_or_path=base_model_path,
            model_config=model_config,
            use_qlora=False  # Don't quantize for eval
        )
        
        logger.info(f"Loading PEFT weights from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        model, processor = load_model_and_processor(
            model_name_or_path=args.model_path,
            model_config=model_config,
            use_qlora=False
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = CTAMIPDataset(
        jsonl_path=args.test_jsonl,
        images_root=args.images_root,
        processor=processor,
        max_seq_length=2048,
        max_images_per_case=4,
        split="test"
    )
    
    # Create inference dataset (for generation)
    inference_dataset = InferenceDataset(
        jsonl_path=args.test_jsonl,
        images_root=args.images_root,
        processor=processor,
        max_images_per_case=4
    )
    
    # Create dataloader
    collator = InferenceCollator(processor=processor, max_length=2048)
    dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2
    )
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(
        model=model,
        processor=processor,
        dataloader=dataloader,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        device=device
    )
    
    # Create prediction lookup
    pred_dict = {p["case_id"]: p["prediction"] for p in predictions}
    
    # Evaluate
    logger.info("Computing metrics...")
    clinical_scorer = ClinicalRubricScorer(
        use_llm_judge=args.use_llm_judge,
        llm_judge_model=args.llm_judge_model
    )
    
    per_case_results = []
    
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        case_id = example["case_id"]
        reference = example["response"]
        prediction = pred_dict.get(case_id, "")
        
        # Evaluate
        eval_results = evaluate_example(prediction, reference, clinical_scorer)
        
        per_case_results.append({
            "case_id": case_id,
            "prediction": prediction,
            "reference": reference,
            "labels": example.get("labels", {}),
            "metrics": eval_results
        })
    
    # Aggregate scores
    logger.info("Aggregating scores...")
    all_metrics = [r["metrics"] for r in per_case_results]
    
    # Extract scalar metrics for aggregation
    scalar_metrics = []
    for m in all_metrics:
        flat = {}
        # BLEU
        flat["bleu"] = m["bleu"]["bleu"]
        # ROUGE
        flat["rouge_1"] = m["rouge"]["rouge_1"]
        flat["rouge_2"] = m["rouge"]["rouge_2"]
        flat["rouge_l"] = m["rouge"]["rouge_l"]
        # Structural
        flat["structure_score"] = m["structural"]["structure_score"]
        flat["has_impression"] = 1.0 if m["structural"]["has_impression_section"] else 0.0
        flat["has_vascular"] = 1.0 if m["structural"]["has_vascular_section"] else 0.0
        # Clinical
        flat["clinical_score"] = m["clinical"]["overall_clinical_score"]
        flat["hallucination_score"] = m["clinical"]["hallucination"]["score"]
        flat["consistency_score"] = m["clinical"]["consistency"]["score"]
        flat["uncertainty_score"] = m["clinical"]["uncertainty"]["score"]
        
        scalar_metrics.append(flat)
    
    aggregated = aggregate_scores(scalar_metrics)
    
    # Prepare final results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "test_jsonl": args.test_jsonl,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample
            }
        },
        "aggregated_metrics": aggregated,
        "per_case_results": per_case_results
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for metric_name, stats in aggregated.items():
        print(f"{metric_name:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
