# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research/thesis project (Furkan Durmus) to fine-tune and preference-align a Vision-Language Model
(VLM) that reads **CTA-MIP** head images and writes a short stroke report. The clinical target is
narrowed to **MCA (middle cerebral artery) occlusion**: presence, side (left/right), and — as a bonus
target — segment (M1 vs M2). The intended contribution is *not* the DPO/KTO/ORPO/GRPO horse-race but
an **image-grounded, clinically-verifiable reward** shown to be optimizer-agnostic across all four
algorithms. Read `PROJECT_LOG.md` first — it captures the study design and preprocessing decisions
that are not derivable from the code. `RESEARCH.md` (literature/blueprint) and `REVIEW.md` (critique
of the original code) provide further background.

The repo is two loosely-coupled subsystems:

1. **Training/alignment pipeline** — `src/`, `scripts/`, `configs/`. Pure PyTorch/HF/TRL.
2. **Preprocessing (MIP generation)** — `preprocessing/`, `generate_mips_from_nrrt.py`,
   `run_mips_from_preprocessed.py`. Turns 3D brain-only CTA NRRD volumes into 2D MIP PNGs. These have
   their own dependency (`SimpleITK`) that is **not** in `requirements.txt`.

## Commands

```bash
# Tests — CPU-only, no GPU/model/network. This is the main verifiable unit test.
python -m tests.test_reward          # or: pytest tests/test_reward.py

# Format
black src scripts tests

# Full pipeline (see README.md for all flags)
python scripts/train_sft.py --config configs/<cfg>.json --output_dir outputs/sft
python scripts/generate_candidates.py --model_path outputs/sft/final --train_jsonl ... --output_path outputs/candidates.jsonl
python scripts/build_preference_data.py --candidates_path outputs/candidates.jsonl --reference_jsonl ... --output_dir outputs/preference_data
python scripts/train_alignment.py --alignment_type dpo --model_path outputs/sft/final --preference_data ... --images_root ... --output_dir outputs/dpo
python scripts/evaluate.py --model_path outputs/sft/final --test_jsonl ... --images_root ... --output_dir outputs/evaluation

# Multi-GPU (DDP/FSDP): launch the same scripts under torchrun
torchrun --nproc_per_node=2 scripts/train_sft.py --config configs/sft_fsdp_lora_example.json

# Check what image-count / batch-size combos fit on the current GPU before a real run
python scripts/probe_dpo_capacity.py

# Preprocessing (requires SimpleITK, numpy, Pillow — installed separately)
python preprocessing/run_fixed_mips.py --slabs_per_axis 1 2 3 4   # sweeps the image-budget ablation

# Container (CUDA 12.8, Python 3.11)
docker build -t paper_study .
```

There is no lint config beyond `black`, and no CI. `test_reward.py` is the only test and only covers
the reward/scoring logic.

## Architecture

### Config system (`src/config.py`)
One nested dataclass `Config` with sections `model / lora / distributed / data / training / generation
/ alignment / evaluation`. Loaded via `Config.from_json(path)`; scripts then let **CLI args override**
individual fields, and write the resolved config back to `<output_dir>/config.json`. When adding a
tunable, add it to the dataclass *and* the relevant script's argparse.

### Shared training runtime (`src/training_runtime.py`) — the load-bearing module
Both `train_sft.py` and `train_alignment.py` build their model/args/trainer through this module. It is
deliberately defensive so the same code runs across transformers/TRL versions and hardware:

- `load_model_with_attention_fallback` tries `flash_attention_2 → sdpa → eager` in order.
- `create_training_arguments` / `build_trainer` **introspect the target class signature** and only
  pass kwargs it accepts. This is why bumping transformers/TRL rarely breaks the scripts — but it also
  means a mistyped kwarg is silently dropped rather than erroring. Verify new kwargs actually land.
- FSDP: `resolve_distributed_config_for_model` auto-infers the transformer layer class to wrap from the
  loaded model when the configured one isn't present.
- TRL compat shims: `ensure_trl_fsdp_compat` (aliases missing `FSDPModule`) and
  `force_trl_dpo_text_mode_for_model` (disables TRL's vision auto-detection branch when needed).
- QLoRA/LoRA applied via `apply_lora_if_enabled`; low-level loading (bnb quant config, a workaround for
  a transformers config-serialization bug on local bnb checkpoints) lives in `src/utils.py`.

### Reward / scoring — the research contribution (`src/scoring.py` + `src/reward.py`)
This is where correctness matters most; changes here change what the model is trained to do.

- `src/scoring.py` — **label-grounded fact score**. `parse_labels()` normalizes case labels to
  `{anomaly, vessel, side}`; `parse_report_claims()` extracts what a report asserts (occlusion
  present/absent, vessel, side, hedging, overconfidence) using **clause-level negation** so "No large
  vessel occlusion" reads as negative. `fact_based_report()` scores on four axes — detection (0.45),
  vessel (0.20), **side (0.25, weighted high because wrong-side errors misdirect treatment)**,
  calibration (0.10) — with a multiplicative contradiction penalty. Also holds the older
  `rule_based_score` (structure + keyword rubric + ROUGE) used as a fallback and the `ClinicalRubricScorer`.
- `src/reward.py` — `LLMJudge` (multimodal, OpenAI-compatible endpoint; **actually sends the CTA
  images**) and `CompositeReward` (blends fact + judge + structure + similarity). Weights are
  **renormalized over whichever components are available** for a case, so a missing label set or an
  unconfigured judge degrades gracefully instead of injecting noise.
- **Design invariant: no silent random rewards.** A component that can't run reports `available=False`
  and is dropped. The old `llm_judge_score` placeholder now *raises* on purpose. Preserve this — do not
  reintroduce a stub that returns a fake/random/length-based reward. (Note: the GRPO `reward_func` in
  `train_alignment.py` is still a length-based placeholder and needs wiring to `CompositeReward`.)
- Default scoring method in `build_preference_data.py` is `composite`; the reference report is scored
  like any other candidate (no automatic 1.0) so DPO doesn't collapse back toward SFT.

### Data flow
Training JSONL rows: `{case_id, images, prompt, response, labels}` (labels e.g.
`{"anomaly_present": true, "main_region": "Right MCA"}`). `src/dataset.py` provides `CTAMIPDataset`
(SFT), `PreferenceDataset` (formats: `pairwise` / `binary` / `group`), `InferenceDataset`.
`src/collator.py` builds chat-templated, multi-image batches. `train_alignment.py` converts preference
data to TRL's expected columns; for pairwise DPO/ORPO it prefers a real **multimodal** HF `datasets`
object (images enabled) and falls back to the legacy collator. **KTO and GRPO currently run text-only**
(image tensors not consumed) — a known limitation flagged in `PROJECT_LOG.md`.

## Project-specific gotchas

- **Execution environment.** Training runs in **WSL/Docker with CUDA**, not on the Windows host where
  this repo sits. Config `model_name_or_path` values are absolute WSL paths (e.g.
  `/workspace/stroke_study/...`) and must be adjusted per machine. `outputs/`, model weights, and
  `data/images/` are git-ignored.
- **Frozen preprocessing protocol — do not vary per patient (it's a confound).** The chosen protocol is
  K=4 slabs/axis + global MIPs, `--norm window`, `low_hu 30`, LPS orientation, and **exclude
  `axial_slab_00` at dataset-build time** (by filtering the index, *not* by deleting files). Rationale
  and the QC still outstanding are in `PROJECT_LOG.md` §5. `HANDOFF_generate_mips.md` is the runbook.
  `generate_mips_from_nrrt.py` / `run_mips_from_preprocessed.py` are the *older* sliding-window
  generator; `preprocessing/make_fixed_mips.py` / `run_fixed_mips.py` are the current fixed-protocol one.
- `dummy_data/` is a single synthetic case duplicated 5× — smoke-test only, not real training signal.
- The stray directory literally named `{data,src,scripts,configs,outputs}` is a leftover from a shell
  brace-expansion that didn't expand; ignore it.
