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
        # Collect images, full texts, and prompt-only texts (per example).
        all_images = []
        all_texts = []
        prompt_texts = []

        for example in batch:
            conversation = example["conversation"]
            all_texts.append(
                self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )
            # Prompt-only = drop the final (assistant) turn, add the generation prompt.
            prompt_texts.append(
                self.processor.apply_chat_template(
                    conversation[:-1], tokenize=False, add_generation_prompt=True
                )
            )
            all_images.append(example["images"])

        # Flatten images for batch processing.
        flattened_images = [img for imgs in all_images for img in imgs]

        batch_inputs = self.processor(
            text=all_texts,
            images=flattened_images if flattened_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs.get("attention_mask")
        labels = input_ids.clone()

        # (1) Never compute loss on padding tokens.
        if attention_mask is not None:
            labels[attention_mask == 0] = self.ignore_index

        # (2) Completion-only masking: hide the prompt (INCLUDING the expanded image
        #     tokens) so loss is computed only on the assistant response. The prompt
        #     length is measured by running the *processor* (with images) on the
        #     prompt-only turns, which reproduces image-placeholder expansion -- a
        #     text-only token count would be far too short for a VLM and would leak
        #     image tokens into the supervised span.
        tokenizer = getattr(self.processor, "tokenizer", None)
        padding_side = getattr(tokenizer, "padding_side", "right")
        for i, (prompt_text, imgs) in enumerate(zip(prompt_texts, all_images)):
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=imgs if imgs else None,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            if attention_mask is not None and padding_side == "left":
                start = int((attention_mask[i] == 0).sum().item())
            else:
                start = 0
            end = min(start + prompt_len, labels.shape[1])
            labels[i, start:end] = self.ignore_index

        batch_inputs["labels"] = labels
        return batch_inputs


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
