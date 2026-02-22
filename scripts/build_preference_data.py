#!/usr/bin/env python3
"""
Build preference datasets from generated candidates.

Supports 3 output formats:
- Pairwise: {images, prompt, chosen, rejected}
- Binary: {images, prompt, response, preference: good/bad}
- Group: {images, prompt, candidates: [{response, score}]}

Usage:
    python scripts/build_preference_data.py \
        --candidates_path outputs/candidates.jsonl \
        --output_dir outputs/preference_data \
        --reference_jsonl data/train.jsonl \
        --scoring_method rule_based \
        --formats pairwise,binary,group
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring import rule_based_score, llm_judge_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build preference datasets from candidates"
    )
    parser.add_argument(
        "--candidates_path",
        type=str,
        required=True,
        help="Path to candidates JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preference datasets"
    )
    parser.add_argument(
        "--reference_jsonl",
        type=str,
        default=None,
        help="Path to reference JSONL (for ground truth responses)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="pairwise,binary,group",
        help="Comma-separated list of formats to generate"
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="rule_based",
        choices=["rule_based", "llm_judge", "random"],
        help="Method to score candidates"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification (good/bad)"
    )
    parser.add_argument(
        "--top_k_for_group",
        type=int,
        default=4,
        help="Number of top candidates to keep in group format"
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default=None,
        help="Model to use as LLM judge"
    )
    return parser.parse_args()


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """Load candidates from JSONL file."""
    candidates = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_references(path: str) -> Dict[str, str]:
    """Load reference responses."""
    references = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                references[item["case_id"]] = item["response"]
    return references


def score_candidates(
    candidates_data: List[Dict[str, Any]],
    references: Dict[str, str],
    scoring_method: str,
    llm_judge_model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Score all candidates using the specified method."""
    scored_data = []
    
    for item in tqdm(candidates_data, desc="Scoring candidates"):
        case_id = item["case_id"]
        reference = references.get(case_id, None)
        
        scored_candidates = []
        for candidate in item["candidates"]:
            text = candidate["text"]
            
            # Skip reference candidates for scoring (they get max score)
            if candidate.get("is_reference", False):
                score = 1.0
            else:
                if scoring_method == "rule_based":
                    score = rule_based_score(text, reference)
                elif scoring_method == "llm_judge":
                    score = llm_judge_score(
                        text,
                        item["prompt"],
                        judge_model=llm_judge_model
                    )
                else:  # random
                    import random
                    score = random.uniform(0.0, 1.0)
            
            scored_candidates.append({
                "text": text,
                "score": score,
                "metadata": {k: v for k, v in candidate.items() if k != "text"}
            })
        
        scored_data.append({
            "case_id": case_id,
            "images": item["images"],
            "prompt": item["prompt"],
            "labels": item.get("labels", {}),
            "candidates": scored_candidates,
            "reference": reference
        })
    
    return scored_data


def build_pairwise_dataset(
    scored_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build pairwise preference dataset.
    
    For each example, creates pairs of (chosen, rejected) where
    chosen has higher score than rejected.
    """
    pairwise_data = []
    
    for item in scored_data:
        candidates = item["candidates"]
        if len(candidates) < 2:
            continue
        
        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        
        # Create pairs: best vs rest
        best = sorted_candidates[0]
        for worse in sorted_candidates[1:]:
            # Only create pair if there's a meaningful difference
            if best["score"] > worse["score"] + 0.05:
                pairwise_data.append({
                    "case_id": item["case_id"],
                    "images": item["images"],
                    "prompt": item["prompt"],
                    "labels": item.get("labels", {}),
                    "chosen": best["text"],
                    "rejected": worse["text"],
                    "chosen_score": best["score"],
                    "rejected_score": worse["score"]
                })
    
    return pairwise_data


def build_binary_dataset(
    scored_data: List[Dict[str, Any]],
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Build binary preference dataset.
    
    Labels each response as "good" or "bad" based on score threshold.
    """
    binary_data = []
    
    for item in scored_data:
        for candidate in item["candidates"]:
            preference = "good" if candidate["score"] >= threshold else "bad"
            
            binary_data.append({
                "case_id": item["case_id"],
                "images": item["images"],
                "prompt": item["prompt"],
                "labels": item.get("labels", {}),
                "response": candidate["text"],
                "preference": preference,
                "score": candidate["score"]
            })
    
    return binary_data


def build_group_dataset(
    scored_data: List[Dict[str, Any]],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Build group preference dataset.
    
    Keeps top-K candidates with their scores for each example.
    """
    group_data = []
    
    for item in scored_data:
        candidates = item["candidates"]
        
        # Sort by score and take top-K
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top_candidates = sorted_candidates[:top_k]
        
        group_data.append({
            "case_id": item["case_id"],
            "images": item["images"],
            "prompt": item["prompt"],
            "labels": item.get("labels", {}),
            "candidates": [
                {"response": c["text"], "score": c["score"]}
                for c in top_candidates
            ],
            "reference": item.get("reference")
        })
    
    return group_data


def main():
    args = parse_args()
    
    # Parse formats
    formats = [f.strip() for f in args.formats.split(",")]
    valid_formats = {"pairwise", "binary", "group"}
    formats = [f for f in formats if f in valid_formats]
    
    if not formats:
        raise ValueError(f"No valid formats specified. Choose from: {valid_formats}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Building preference datasets...")
    logger.info(f"Input: {args.candidates_path}")
    logger.info(f"Formats: {formats}")
    logger.info(f"Scoring method: {args.scoring_method}")
    
    # Load candidates
    logger.info("Loading candidates...")
    candidates_data = load_candidates(args.candidates_path)
    logger.info(f"Loaded {len(candidates_data)} examples")
    
    # Load references
    references = {}
    if args.reference_jsonl:
        logger.info("Loading references...")
        references = load_references(args.reference_jsonl)
        logger.info(f"Loaded {len(references)} references")
    
    # Score candidates
    logger.info("Scoring candidates...")
    scored_data = score_candidates(
        candidates_data,
        references,
        args.scoring_method,
        args.llm_judge_model
    )
    
    # Build and save datasets
    if "pairwise" in formats:
        logger.info("Building pairwise dataset...")
        pairwise = build_pairwise_dataset(scored_data)
        output_path = os.path.join(args.output_dir, "pairwise.jsonl")
        with open(output_path, "w") as f:
            for item in pairwise:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(pairwise)} pairs to {output_path}")
    
    if "binary" in formats:
        logger.info("Building binary dataset...")
        binary = build_binary_dataset(scored_data, args.score_threshold)
        output_path = os.path.join(args.output_dir, "binary.jsonl")
        with open(output_path, "w") as f:
            for item in binary:
                f.write(json.dumps(item) + "\n")
        
        # Print distribution
        good_count = sum(1 for item in binary if item["preference"] == "good")
        bad_count = len(binary) - good_count
        logger.info(f"Saved {len(binary)} examples to {output_path}")
        logger.info(f"  Good: {good_count}, Bad: {bad_count}")
    
    if "group" in formats:
        logger.info("Building group dataset...")
        group = build_group_dataset(scored_data, args.top_k_for_group)
        output_path = os.path.join(args.output_dir, "group.jsonl")
        with open(output_path, "w") as f:
            for item in group:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(group)} examples to {output_path}")
    
    # Save scored data (for debugging/analysis)
    scored_path = os.path.join(args.output_dir, "scored_candidates.jsonl")
    with open(scored_path, "w") as f:
        for item in scored_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved scored candidates to {scored_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
