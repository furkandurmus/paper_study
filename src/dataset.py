"""
Dataset classes for CTA-MIP VLM training.
Supports multi-image inputs from JSONL files.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset


class CTAMIPDataset(Dataset):
    """
    Dataset for CTA MIP images with multi-image support.
    
    Expected JSONL format:
    {
        "case_id": "cta_000123",
        "images": ["ax_mip_20mm_1.png", "ax_mip_20mm_2.png", ...],
        "prompt": "CTA head: describe major vascular abnormality...",
        "response": "Vascular summary: ... Impression: ...",
        "labels": {"anomaly_present": true, "main_region": "Right MCA"}
    }
    """
    
    def __init__(
        self,
        jsonl_path: str,
        images_root: str,
        processor: Any,
        max_seq_length: int = 2048,
        max_images_per_case: int = 4,
        split: str = "train"
    ):
        self.jsonl_path = jsonl_path
        self.images_root = Path(images_root)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.max_images_per_case = max_images_per_case
        self.split = split
        
        # Load data
        self.data = self._load_jsonl()
        
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images for a case."""
        images = []
        for img_path in image_paths[:self.max_images_per_case]:
            full_path = self.images_root / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                images.append(img)
            else:
                # Create a blank image if file not found
                print(f"Warning: Image not found: {full_path}")
                images.append(Image.new("RGB", (448, 448), color="black"))
        return images
    
    def _create_conversation(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create conversation format for chat template."""
        # Build content with images and text
        content = []
        
        # Add images as placeholders
        num_images = min(len(item["images"]), self.max_images_per_case)
        for _ in range(num_images):
            content.append({"type": "image"})
        
        # Add text prompt
        content.append({"type": "text", "text": item["prompt"]})
        
        conversation = [
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["response"]}]
            }
        ]
        return conversation
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load images
        images = self._load_images(item["images"])
        
        # Create conversation
        conversation = self._create_conversation(item)
        
        return {
            "case_id": item["case_id"],
            "images": images,
            "conversation": conversation,
            "prompt": item["prompt"],
            "response": item["response"],
            "labels": item.get("labels", {}),
            "image_paths": item["images"][:self.max_images_per_case]
        }


class PreferenceDataset(Dataset):
    """
    Dataset for preference-based training (DPO/KTO/ORPO).
    
    Supports multiple formats:
    - Pairwise: {images, prompt, chosen, rejected}
    - Binary: {images, prompt, response, preference}
    - Group: {images, prompt, candidates: [{response, score}]}
    """
    
    def __init__(
        self,
        jsonl_path: str,
        images_root: str,
        processor: Any,
        max_images_per_case: int = 4,
        format_type: str = "pairwise"  # pairwise, binary, group
    ):
        self.jsonl_path = jsonl_path
        self.images_root = Path(images_root)
        self.processor = processor
        self.max_images_per_case = max_images_per_case
        self.format_type = format_type
        
        self.data = self._load_jsonl()
    
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load preference data from JSONL file."""
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images for a case."""
        images = []
        for img_path in image_paths[:self.max_images_per_case]:
            full_path = self.images_root / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Image not found: {full_path}")
                images.append(Image.new("RGB", (448, 448), color="black"))
        return images
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        images = self._load_images(item["images"])
        
        result = {
            "case_id": item.get("case_id", f"pref_{idx}"),
            "images": images,
            "prompt": item["prompt"],
            "image_paths": item["images"][:self.max_images_per_case]
        }
        
        if self.format_type == "pairwise":
            result["chosen"] = item["chosen"]
            result["rejected"] = item["rejected"]
        elif self.format_type == "binary":
            result["response"] = item["response"]
            result["preference"] = item["preference"]  # "good" or "bad"
        elif self.format_type == "group":
            result["candidates"] = item["candidates"]  # List of {response, score}
        
        return result


class InferenceDataset(Dataset):
    """Dataset for inference/generation (no ground truth responses)."""
    
    def __init__(
        self,
        jsonl_path: str,
        images_root: str,
        processor: Any,
        max_images_per_case: int = 4
    ):
        self.jsonl_path = jsonl_path
        self.images_root = Path(images_root)
        self.processor = processor
        self.max_images_per_case = max_images_per_case
        
        self.data = self._load_jsonl()
    
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Only need case_id, images, prompt
                    data.append({
                        "case_id": item["case_id"],
                        "images": item["images"],
                        "prompt": item["prompt"],
                        "labels": item.get("labels", {})
                    })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images for a case."""
        images = []
        for img_path in image_paths[:self.max_images_per_case]:
            full_path = self.images_root / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Image not found: {full_path}")
                images.append(Image.new("RGB", (448, 448), color="black"))
        return images
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        images = self._load_images(item["images"])
        
        return {
            "case_id": item["case_id"],
            "images": images,
            "prompt": item["prompt"],
            "labels": item.get("labels", {}),
            "image_paths": item["images"][:self.max_images_per_case]
        }
