"""
Configuration dataclasses for CTA-MIP VLM training and alignment.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    model_type: str = "auto"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    
    # Quantization config (for QLoRA)
    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class DistributedConfig:
    """Distributed/runtime strategy configuration."""
    strategy: str = "single"  # single, ddp, fsdp
    ddp_find_unused_parameters: Optional[bool] = None

    # FSDP options mirror Hugging Face TrainingArguments.
    fsdp: List[str] = field(default_factory=list)
    fsdp_min_num_params: int = 0
    fsdp_transformer_layer_cls_to_wrap: List[str] = field(default_factory=list)
    fsdp_backward_prefetch: str = "backward_pre"
    fsdp_forward_prefetch: bool = False
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_offload_params: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_activation_checkpointing: bool = False


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    use_lora: bool = True
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    modules_to_save: Optional[List[str]] = field(default_factory=lambda: ["embed_tokens", "lm_head"])


@dataclass
class DataConfig:
    """Data configuration."""
    train_jsonl: str = "data/train.jsonl"
    val_jsonl: Optional[str] = "data/val.jsonl"
    test_jsonl: Optional[str] = "data/test.jsonl"
    images_root: str = "data/images"
    
    # Image processing
    image_size: int = 448
    max_images_per_case: int = 4
    
    # Text processing
    max_seq_length: int = 2048
    max_prompt_length: int = 512
    max_response_length: int = 1024


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Logging & Checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Other
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])


@dataclass
class GenerationConfig:
    """Generation configuration for candidate generation."""
    num_candidates: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class AlignmentConfig:
    """Alignment training configuration (DPO/KTO/ORPO/GRPO)."""
    alignment_type: str = "dpo"  # dpo, kto, orpo, grpo
    output_dir: str = "outputs/alignment"
    
    # DPO-specific
    beta: float = 0.1
    label_smoothing: float = 0.0
    
    # KTO-specific
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    
    # ORPO-specific
    orpo_alpha: float = 1.0
    
    # GRPO-specific
    grpo_group_size: int = 4
    grpo_epsilon: float = 0.2
    
    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    output_dir: str = "outputs/evaluation"
    metrics: List[str] = field(default_factory=lambda: [
        "bleu", "rouge", "structural", "clinical"
    ])
    
    # Clinical rubric weights
    hallucination_weight: float = 1.0
    consistency_weight: float = 1.0
    uncertainty_weight: float = 0.5
    
    # LLM judge (optional)
    use_llm_judge: bool = False
    llm_judge_model: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            distributed=DistributedConfig(**data.get("distributed", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            alignment=AlignmentConfig(**data.get("alignment", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {}))
        )
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        data = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "distributed": self.distributed.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "alignment": self.alignment.__dict__,
            "evaluation": self.evaluation.__dict__
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
