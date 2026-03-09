"""
Utility functions for model loading, training setup, and common operations.
"""

import inspect
import json
import os
import random
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers versions without this class
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers versions without this class
    AutoModelForVision2Seq = None


def _load_vision_model_from_pretrained(model_name_or_path: str, **kwargs):
    """
    Load multimodal model across transformers API variants.
    Prefers AutoModelForImageTextToText, falls back to AutoModelForVision2Seq.
    """
    candidates = [AutoModelForImageTextToText, AutoModelForVision2Seq]
    errors = []

    for model_cls in candidates:
        if model_cls is None:
            continue
        try:
            return model_cls.from_pretrained(model_name_or_path, **kwargs)
        except Exception as exc:
            errors.append(f"{model_cls.__name__}: {exc}")

    if errors:
        raise RuntimeError("Failed to load vision model. " + " | ".join(errors))
    raise RuntimeError(
        "No supported multimodal AutoModel class found in transformers. "
        "Expected AutoModelForImageTextToText or AutoModelForVision2Seq."
    )


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


def _load_embedded_bnb_quant_config(model_name_or_path: str) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig from a local checkpoint's config.json when present."""
    model_dir = Path(model_name_or_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except Exception:
        return None

    quant_cfg = raw_config.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return None
    if quant_cfg.get("quant_method") != "bitsandbytes":
        return None

    sig = inspect.signature(BitsAndBytesConfig.__init__).parameters
    kwargs = {}
    for key, value in quant_cfg.items():
        if key.startswith("_"):
            continue
        if key not in sig:
            continue
        if key == "bnb_4bit_compute_dtype" and isinstance(value, str):
            value = getattr(torch, value, value)
        kwargs[key] = value

    try:
        return BitsAndBytesConfig(**kwargs)
    except Exception:
        return None


@contextmanager
def _sanitized_local_model_dir(model_name_or_path: str):
    """
    Work around a Transformers config serialization bug for some bnb checkpoints.
    Creates a temporary local copy with `quantization_config` removed from config.json.
    """
    model_dir = Path(model_name_or_path)
    config_path = model_dir / "config.json"
    if not (model_dir.is_dir() and config_path.exists()):
        yield model_name_or_path
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except Exception:
        yield model_name_or_path
        return

    if "quantization_config" not in raw_config:
        yield model_name_or_path
        return

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="hf_model_cfg_fix_")
    tmp_dir = Path(tmp_dir_obj.name)
    try:
        for child in model_dir.iterdir():
            dst = tmp_dir / child.name
            if child.name == "config.json":
                continue
            try:
                os.symlink(child, dst)
            except OSError:
                if child.is_dir():
                    shutil.copytree(child, dst, symlinks=True)
                else:
                    shutil.copy2(child, dst)

        sanitized = dict(raw_config)
        sanitized.pop("quantization_config", None)
        with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(sanitized, f)

        yield str(tmp_dir)
    finally:
        tmp_dir_obj.cleanup()


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
    
    # Setup quantization config if using QLoRA or loading a local bnb-quantized checkpoint.
    quantization_config = None
    embedded_bnb_quant = _load_embedded_bnb_quant_config(model_name_or_path)
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4")
        )
        logging.info("Using 4-bit quantization (QLoRA)")
    elif embedded_bnb_quant is not None:
        quantization_config = embedded_bnb_quant
        logging.info("Using embedded bitsandbytes quantization config from local checkpoint")
    
    # Load processor first
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.get("trust_remote_code", True)
    )

    # Some local bnb checkpoints trigger a transformers config repr bug during AutoConfig loading.
    # Retry via a sanitized temporary config.json and pass quantization explicitly.
    load_path = model_name_or_path
    with _sanitized_local_model_dir(model_name_or_path) as maybe_sanitized_path:
        load_path = maybe_sanitized_path
        common_kwargs = {
            "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
            "attn_implementation": model_config.get("attn_implementation", "flash_attention_2"),
            "quantization_config": quantization_config,
            "trust_remote_code": model_config.get("trust_remote_code", True),
            "device_map": "auto" if use_qlora else None,
        }
        model = _load_vision_model_from_pretrained(load_path, **common_kwargs)
    
    # Prepare model for k-bit training if using QLoRA
    if use_qlora or (quantization_config is not None):
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
        model = _load_vision_model_from_pretrained(checkpoint_path)
    
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
