"""
Data collators for CTA-MIP VLM training.
Handles multi-image inputs and chat template formatting.
"""

from typing import Any, Dict, List, Optional, Union
import torch
from transformers import ProcessorMixin


class SFTDataCollator:
    """
    Collator for supervised fine-tuning.
    Formats conversations with chat template and handles multi-image inputs.
    """
    
    def __init__(
        self,
        processor: ProcessorMixin,
        max_length: int = 2048,
        ignore_index: int = -100
    ):
        self.processor = processor
        self.max_length = max_length
        self.ignore_index = ignore_index
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples.
        
        Each example contains:
        - images: List[PIL.Image]
        - conversation: List[Dict] with chat format
        """
        # Collect all images and build texts
        all_images = []
        all_texts = []
        
        for example in batch:
            images = example["images"]
            conversation = example["conversation"]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            all_images.append(images)
            all_texts.append(text)
        
        # Process with processor
        # Flatten images for batch processing
        flattened_images = [img for imgs in all_images for img in imgs]
        
        # Tokenize with images
        batch_inputs = self.processor(
            text=all_texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels (mask prompt tokens with ignore_index)
        labels = batch_inputs["input_ids"].clone()
        
        # Find the assistant response start and mask everything before it
        # This is model-specific; for Qwen, we look for assistant tokens
        for i, text in enumerate(all_texts):
            # Find where assistant response starts
            # This is a simplified approach - may need adjustment per model
            assistant_start = self._find_assistant_start(text)
            if assistant_start > 0:
                # Tokenize just the prompt part to find length
                prompt_text = text[:assistant_start]
                prompt_tokens = self.processor.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                prompt_len = prompt_tokens["input_ids"].shape[1]
                # Mask prompt tokens
                labels[i, :min(prompt_len, labels.shape[1])] = self.ignore_index
        
        batch_inputs["labels"] = labels
        
        return batch_inputs
    
    def _find_assistant_start(self, text: str) -> int:
        """Find the position where assistant response starts."""
        # Common patterns for different models
        patterns = [
            "assistant\n",  # Qwen format
            "Assistant:",
            "<|im_start|>assistant",
            "[/INST]",  # Llama format
        ]
        
        for pattern in patterns:
            pos = text.find(pattern)
            if pos != -1:
                return pos + len(pattern)
        
        return 0


class PreferenceDataCollator:
    """
    Collator for preference-based training (DPO/KTO).
    Handles chosen and rejected responses.
    """
    
    def __init__(
        self,
        processor: ProcessorMixin,
        max_length: int = 2048,
        format_type: str = "pairwise"  # pairwise, binary, group
    ):
        self.processor = processor
        self.max_length = max_length
        self.format_type = format_type
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of preference examples."""
        
        if self.format_type == "pairwise":
            return self._process_pairwise(batch)
        elif self.format_type == "binary":
            return self._process_binary(batch)
        elif self.format_type == "group":
            return self._process_group(batch)
        else:
            raise ValueError(f"Unknown format_type: {self.format_type}")
    
    def _process_pairwise(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process pairwise preference data."""
        chosen_texts = []
        rejected_texts = []
        all_images = []
        
        for example in batch:
            images = example["images"]
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Build conversations
            chosen_conv = self._build_conversation(prompt, chosen, images)
            rejected_conv = self._build_conversation(prompt, rejected, images)
            
            chosen_text = self.processor.apply_chat_template(
                chosen_conv, tokenize=False, add_generation_prompt=False
            )
            rejected_text = self.processor.apply_chat_template(
                rejected_conv, tokenize=False, add_generation_prompt=False
            )
            
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
            all_images.append(images)
        
        # Flatten images
        flattened_images = [img for imgs in all_images for img in imgs]
        
        # Process chosen
        chosen_inputs = self.processor(
            text=chosen_texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Process rejected
        rejected_inputs = self.processor(
            text=rejected_texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
            "pixel_values": chosen_inputs.get("pixel_values"),
            "image_grid_thw": chosen_inputs.get("image_grid_thw")
        }
    
    def _process_binary(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process binary preference data (KTO)."""
        texts = []
        all_images = []
        preferences = []
        
        for example in batch:
            images = example["images"]
            prompt = example["prompt"]
            response = example["response"]
            preference = example["preference"]  # "good" or "bad"
            
            conversation = self._build_conversation(prompt, response, images)
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            
            texts.append(text)
            all_images.append(images)
            preferences.append(1.0 if preference == "good" else 0.0)
        
        flattened_images = [img for imgs in all_images for img in imgs]
        
        inputs = self.processor(
            text=texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        inputs["preferences"] = torch.tensor(preferences)
        return inputs
    
    def _process_group(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process group preference data (for GRPO or ranking)."""
        # Group format requires special handling - return raw for custom processing
        return {
            "batch": batch,
            "format": "group"
        }
    
    def _build_conversation(
        self,
        prompt: str,
        response: str,
        images: List[Any]
    ) -> List[Dict[str, Any]]:
        """Build conversation from prompt and response."""
        content = []
        
        # Add images
        for _ in images:
            content.append({"type": "image"})
        
        # Add prompt
        content.append({"type": "text", "text": prompt})
        
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]


class InferenceCollator:
    """Collator for inference/generation."""
    
    def __init__(
        self,
        processor: ProcessorMixin,
        max_length: int = 2048
    ):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch for inference."""
        all_images = []
        all_texts = []
        case_ids = []
        
        for example in batch:
            images = example["images"]
            prompt = example["prompt"]
            
            # Build user conversation only (no assistant response)
            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            
            conversation = [{"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True  # Important: add generation prompt
            )
            
            all_images.append(images)
            all_texts.append(text)
            case_ids.append(example["case_id"])
        
        # Flatten images
        flattened_images = [img for imgs in all_images for img in imgs]
        
        # Tokenize
        inputs = self.processor(
            text=all_texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "case_ids": case_ids
        }
