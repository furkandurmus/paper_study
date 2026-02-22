"""
Utility functions for model loading, training setup, and common operations.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_model_and_processor(
    model_name_or_path: str,
    model_config: Dict[str, Any],
    lora_config: Optional[Dict[str, Any]] = None,
    use_qlora: bool = False
) -> Tuple[Any, Any]:
    """
    Load VLM model and processor with optional QLoRA/LoRA.
    
    Args:
        model_name_or_path: HuggingFace model identifier or local path
        model_config: Model configuration dict
        lora_config: LoRA configuration dict (optional)
        use_qlora: Whether to use 4-bit quantization
    
    Returns:
        Tuple of (model, processor)
    """
    logging.info(f"Loading model: {model_name_or_path}")
    
    # Setup quantization config if using QLoRA
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4")
        )
        logging.info("Using 4-bit quantization (QLoRA)")
    
    # Load processor first
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        quantization_config=quantization_config,
        trust_remote_code=model_config.get("trust_remote_code", True),
        device_map="auto" if use_qlora else None
    )
    
    # Prepare model for k-bit training if using QLoRA
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_config.get("gradient_checkpointing", True)
        )
    
    # Apply LoRA if configured
    if lora_config and lora_config.get("use_lora", False):
        logging.info("Applying LoRA adapters")
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            modules_to_save=lora_config.get("modules_to_save", ["embed_tokens", "lm_head"])
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, processor


def load_checkpoint(
    model: Any,
    checkpoint_path: str,
    is_peft: bool = True
) -> Any:
    """Load model from checkpoint."""
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    
    if is_peft:
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model = AutoModelForVision2Seq.from_pretrained(checkpoint_path)
    
    return model


def save_checkpoint(
    model: Any,
    output_dir: str,
    is_peft: bool = True
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving checkpoint to: {output_dir}")
    
    if is_peft:
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)


def count_trainable_parameters(model: Any) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: Any) -> int:
    """Count the total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory():
    """Get GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {}


def format_chat_message(
    role: str,
    content: str,
    images: Optional[list] = None
) -> Dict[str, Any]:
    """
    Format a chat message with optional images.
    
    Args:
        role: "user" or "assistant"
        content: Text content
        images: List of image paths or PIL images (optional)
    
    Returns:
        Formatted message dict
    """
    message_content = []
    
    if images:
        for _ in images:
            message_content.append({"type": "image"})
    
    message_content.append({"type": "text", "text": content})
    
    return {
        "role": role,
        "content": message_content
    }


def parse_conversation_text(text: str) -> Tuple[str, str]:
    """
    Parse conversation text to extract prompt and response.
    Model-specific parsing.
    """
    # Try to find assistant response
    patterns = [
        "assistant\n",
        "Assistant:",
        "<|im_start|>assistant",
    ]
    
    for pattern in patterns:
        if pattern in text:
            parts = text.split(pattern, 1)
            if len(parts) == 2:
                prompt = parts[0].strip()
                response = parts[1].strip()
                return prompt, response
    
    # Fallback: return as-is
    return text, ""


def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """Create timestamped output directory."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
