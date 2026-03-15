# CTA-MIP VLM Training & Alignment

Vision-Language Model training and alignment for acute ischemic stroke detection using CTA MIP images.

## Overview

This repository provides a complete pipeline for fine-tuning and aligning Vision-Language Models (VLMs) on CTA MIP (Maximum Intensity Projection) images for stroke radiology report generation.

### Pipeline Stages

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  SFT Train  │────▶│ Generate Candidates│───▶│ Build Preference │
│   (train_sft)│     │ (generate_candidates)│  │    Data         │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                       │
                              ┌────────────────────────┘
                              ▼
                       ┌─────────────┐
                       │  Alignment  │
                       │(train_alignment)
                       └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │  Evaluate   │
                       │  (evaluate) │
                       └─────────────┘
```

1. **SFT (Supervised Fine-Tuning)**: Train base VLM on radiology reports
2. **Generate Candidates**: Create multiple response candidates per input
3. **Build Preference Data**: Score and format candidates for alignment
4. **Alignment**: Apply DPO/KTO/ORPO/GRPO for preference optimization
5. **Evaluate**: Compute metrics and clinical rubric scores

## Repository Structure

```
.
├── configs/
│   └── example_config.json      # Example configuration file
├── data/
│   ├── train.jsonl              # Training data (you provide)
│   ├── val.jsonl                # Validation data (you provide)
│   ├── test.jsonl               # Test data (you provide)
│   └── images/                  # Image directory (you provide)
├── scripts/
│   ├── train_sft.py             # SFT training script
│   ├── generate_candidates.py   # Candidate generation for alignment
│   ├── build_preference_data.py # Build preference datasets
│   ├── train_alignment.py       # Alignment training stub
│   └── evaluate.py              # Evaluation script
├── src/
│   ├── config.py                # Configuration dataclasses
│   ├── dataset.py               # Dataset classes
│   ├── collator.py              # Data collators
│   ├── utils.py                 # Utility functions
│   └── scoring.py               # Scoring functions
├── outputs/                     # Training outputs (created)
└── README.md                    # This file
```

## Installation

```bash
# Create virtual environment
conda create -n cta-vlm python=3.10
conda activate cta-vlm

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages
pip install transformers accelerate peft bitsandbytes

# Install additional dependencies
pip install Pillow tqdm numpy

# Optional: Install flash-attention for efficiency
pip install flash-attn --no-build-isolation

# Optional: Install TRL for alignment training
pip install trl

# Optional: Install evaluation metrics
pip install rouge-score sacrebleu nltk
```

## Data Format

Your JSONL files should have this schema:

```json
{
  "case_id": "cta_000123",
  "images": ["ax_mip_20mm_1.png", "ax_mip_20mm_2.png", "cor_mip_10mm.png", "sag_mip_10mm.png"],
  "prompt": "CTA head: describe major vascular abnormality and write a short Impression.",
  "response": "Vascular summary: Occlusion of the right MCA M1 segment. Impression: Acute right MCA territory stroke.",
  "labels": {"anomaly_present": true, "main_region": "Right MCA"}
}
```

## Usage

## Systematic Training Pipeline

Training is now organized around shared runtime/config logic so the same stack can be
used for:

- SFT and alignment
- single GPU, DDP, or FSDP
- full finetuning, LoRA, or QLoRA
- different multimodal models via `model.model_name_or_path`

Key config sections:

- `model`: model path, dtype, attention backend, QLoRA settings
- `lora`: LoRA on/off and target modules
- `distributed`: `single`, `ddp`, or `fsdp` plus FSDP wrapping options
- `training` / `alignment`: task-specific optimization settings

Example presets:

- [configs/sft_fsdp_lora_example.json](/workspace/stroke_study/paper_study/configs/sft_fsdp_lora_example.json)
- [configs/sft_qlora_single_gpu_example.json](/workspace/stroke_study/paper_study/configs/sft_qlora_single_gpu_example.json)
- [configs/dpo_fsdp_fullft_example.json](/workspace/stroke_study/paper_study/configs/dpo_fsdp_fullft_example.json)

Multi-GPU launch examples:

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/sft_fsdp_lora_example.json
```

```bash
torchrun --nproc_per_node=2 scripts/train_alignment.py \
  --config configs/dpo_fsdp_fullft_example.json \
  --alignment_type dpo \
  --model_path /path/to/model \
  --preference_data data/prefs.jsonl \
  --images_root data/images \
  --output_dir outputs/dpo_fsdp
```

### 1. Supervised Fine-Tuning (SFT)

Train the base model on your radiology reports:

```bash
python scripts/train_sft.py \
    --config configs/example_config.json \
    --output_dir outputs/sft \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --train_jsonl data/train.jsonl \
    --val_jsonl data/val.jsonl \
    --images_root data/images
```

**Key Arguments:**
- `--model_name`: HuggingFace model (Qwen2-VL, Qwen3-VL, SmolVLM, etc.)
- `--config`: Path to config JSON (optional, uses defaults if not provided)
- `--output_dir`: Where to save checkpoints and logs

### 2. Generate Candidates for Alignment

Generate multiple candidate responses per training example:

```bash
python scripts/generate_candidates.py \
    --model_path outputs/sft/final \
    --train_jsonl data/train.jsonl \
    --images_root data/images \
    --output_path outputs/candidates.jsonl \
    --num_candidates 4 \
    --temperature_range 0.5,0.7,0.9,1.0 \
    --use_reference
```

**Key Arguments:**
- `--num_candidates`: Number of responses to generate per example
- `--temperature_range`: Temperatures for diversity (comma-separated)
- `--use_reference`: Include ground truth as a candidate

### 3. Build Preference Dataset

Score candidates and create preference data in multiple formats:

```bash
python scripts/build_preference_data.py \
    --candidates_path outputs/candidates.jsonl \
    --reference_jsonl data/train.jsonl \
    --output_dir outputs/preference_data \
    --scoring_method rule_based \
    --formats pairwise,binary,group \
    --score_threshold 0.5
```

**Output Formats:**
- **Pairwise** (`pairwise.jsonl`): `{images, prompt, chosen, rejected}`
- **Binary** (`binary.jsonl`): `{images, prompt, response, preference}`
- **Group** (`group.jsonl`): `{images, prompt, candidates: [{response, score}]}`

**Scoring Methods:**
- `rule_based`: Uses structural + clinical rubric scores
- `llm_judge`: Uses LLM-as-judge (requires `--llm_judge_model`)
- `random`: Random scoring (for testing)

### 4. Alignment Training

Train with preference optimization (stub - requires TRL implementation):

**DPO (Direct Preference Optimization):**
```bash
python scripts/train_alignment.py \
    --alignment_type dpo \
    --model_path outputs/sft/final \
    --preference_data outputs/preference_data/pairwise.jsonl \
    --images_root data/images \
    --output_dir outputs/dpo \
    --beta 0.1 \
    --num_epochs 1
```

**KTO (Kahneman-Tversky Optimization):**
```bash
python scripts/train_alignment.py \
    --alignment_type kto \
    --model_path outputs/sft/final \
    --preference_data outputs/preference_data/binary.jsonl \
    --images_root data/images \
    --output_dir outputs/kto
```

**ORPO (Odds Ratio Preference Optimization):**
```bash
python scripts/train_alignment.py \
    --alignment_type orpo \
    --model_path outputs/sft/final \
    --preference_data outputs/preference_data/pairwise.jsonl \
    --images_root data/images \
    --output_dir outputs/orpo
```

**GRPO (Group Relative Policy Optimization):**
```bash
python scripts/train_alignment.py \
    --alignment_type grpo \
    --model_path outputs/sft/final \
    --preference_data outputs/preference_data/group.jsonl \
    --images_root data/images \
    --output_dir outputs/grpo
```

### 5. Evaluation

Evaluate model on test set:

```bash
python scripts/evaluate.py \
    --model_path outputs/sft/final \
    --test_jsonl data/test.jsonl \
    --images_root data/images \
    --output_dir outputs/evaluation \
    --max_new_tokens 512 \
    --temperature 0.7
```

**Evaluation Output:**
- `evaluation_results.json` with:
  - Per-case predictions and references
  - BLEU/ROUGE scores
  - Structural metrics (Impression section presence, etc.)
  - Clinical rubric scores (hallucination, consistency, uncertainty)
  - Aggregated statistics

## Configuration

All hyperparameters can be configured via JSON config files or CLI arguments. See [`configs/example_config.json`](configs/example_config.json) for a complete example.

### Key Configuration Sections

**Model Config:**
- `model_name_or_path`: Base VLM to fine-tune
- `use_qlora`: Enable 4-bit quantization
- `torch_dtype`: `bfloat16` or `float16`

**LoRA Config:**
- `r`: LoRA rank (64 recommended)
- `lora_alpha`: LoRA alpha (128 recommended)
- `target_modules`: Which layers to adapt

**Training Config:**
- `learning_rate`: 2e-4 for SFT, 5e-5 for alignment
- `per_device_train_batch_size`: Usually 1 for VLMs
- `gradient_accumulation_steps`: 8 for effective batch size of 8

## Clinical Rubrics

The evaluation includes clinical-style rubrics:

### Hallucination Detection
- Flags unsupported vessel/site claims
- Detects vessels mentioned in prediction but not reference

### Consistency Checks
- Left/right laterality consistency
- Detects contradictory statements

### Uncertainty Language
- Flags overconfident language ("definitely", "absolutely")
- Checks for appropriate uncertainty qualifiers

## Extending the Code

### Adding a New Scoring Function

Edit [`src/scoring.py`](src/scoring.py):

```python
def my_custom_score(prediction: str, reference: str) -> float:
    # Your scoring logic
    return score
```

Then use it in [`scripts/build_preference_data.py`](scripts/build_preference_data.py).

### Implementing Alignment Training

The [`scripts/train_alignment.py`](scripts/train_alignment.py) file contains stubs for DPO/KTO/ORPO/GRPO. To implement:

1. Install TRL: `pip install trl`
2. Import the appropriate trainer: `from trl import DPOTrainer`
3. Replace the stub with actual implementation (see comments in file)

### Adding Custom Metrics

Edit [`src/scoring.py`](src/scoring.py) and update [`scripts/evaluate.py`](scripts/evaluate.py) to include them.

## Hardware Requirements

**Minimum (QLoRA):**
- GPU: 1x A10G (24GB) or RTX 3090 (24GB)
- RAM: 32GB
- Storage: 100GB

**Recommended (Full Fine-tuning):**
- GPU: 1x A100 (40GB/80GB) or 2x A10G
- RAM: 64GB
- Storage: 200GB

## Citation

If you use this code, please cite:

```bibtex
@software{cta_mip_vlm,
  title={CTA-MIP VLM Training Framework},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.

## Troubleshooting

**Out of Memory:**
- Enable QLoRA: `use_qlora: true`
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model (SmolVLM-256M instead of Qwen2-VL-7B)

**Flash Attention Not Available:**
- Install: `pip install flash-attn --no-build-isolation`
- Or change `attn_implementation` to `eager`

**Image Loading Errors:**
- Verify image paths are relative to `images_root`
- Check image files exist and are valid PNG/JPG

## Contact

For questions or issues, please open a GitHub issue.
