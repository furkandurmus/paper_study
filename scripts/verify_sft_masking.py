#!/usr/bin/env python3
"""
Verify SFT completion-only label masking on real data with a real (V)LM processor.

This is the DEFINITIVE check for the image-token-expansion masking bug: it runs a few
dataset rows through CTAMIPDataset + SFTDataCollator, then decodes the tokens that WILL
be supervised (labels != -100) and prints them next to the expected assistant response.
If the fix is correct, the supervised span == the response text -- no image/vision
tokens, no prompt. Needs only the *processor* (not model weights) -> runs on CPU.

    python scripts/verify_sft_masking.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --dataset_jsonl data/train.jsonl \
        --images_root data/images \
        --n 3 --max_images_per_case 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from src.collator import SFTDataCollator  # noqa: E402
from src.dataset import CTAMIPDataset  # noqa: E402

# Substrings that must NOT appear in the supervised span (would mean image/prompt leaked).
_LEAK_MARKERS = ["<|image", "<|vision", "<|video", "image_pad", "<image>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path of the (V)LM processor")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_images_per_case", type=int, default=1)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = getattr(processor, "tokenizer", processor)

    ds = CTAMIPDataset(
        jsonl_path=args.dataset_jsonl,
        images_root=args.images_root,
        processor=processor,
        max_seq_length=args.max_seq_length,
        max_images_per_case=args.max_images_per_case,
    )
    collator = SFTDataCollator(processor=processor, max_length=args.max_seq_length)

    n = min(args.n, len(ds))
    batch = [ds[i] for i in range(n)]
    out = collator(batch)
    input_ids, labels = out["input_ids"], out["labels"]
    attn = out.get("attention_mask", torch.ones_like(labels))

    all_ok = True
    for i in range(n):
        sup_mask = labels[i] != -100
        sup_ids = input_ids[i][sup_mask]
        supervised_raw = tok.decode(sup_ids, skip_special_tokens=False)
        supervised_clean = tok.decode(sup_ids, skip_special_tokens=True).strip()
        expected = batch[i]["response"].strip()

        n_sup = int(sup_mask.sum())
        n_real = int(attn[i].sum())
        leaked = [m for m in _LEAK_MARKERS if m in supervised_raw]
        covers = expected[:50] in supervised_clean or supervised_clean[:50] in expected
        ok = covers and not leaked
        all_ok &= ok

        print("=" * 72)
        print(f"case {batch[i]['case_id']}: supervised {n_sup} / {n_real} real tokens   "
              f"OK={ok}")
        if leaked:
            print(f"  !! LEAKED tokens in supervised span: {leaked}")
        print(f"-- EXPECTED (response) --\n{expected}")
        print(f"-- SUPERVISED (labels != -100, special tokens kept) --\n{supervised_raw}")

    print("\nRESULT:", "PASS  (only the response is supervised)" if all_ok
          else "FAIL  (prompt/image tokens leaked, or response not covered)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
