#!/usr/bin/env python3
"""
Batch-generate FIXED-PROTOCOL CTA-MIP images for every preprocessed patient,
and aggregate per-patient manifests into one dataset-level index.

Pipeline this drives (per patient):
    preprocessed_brains/<patient>/<brain_only>.nrrd
        -> make_fixed_mips.py  (deterministic equal-count slabs + manifest.json)
        -> mip_images/slabs<K>[ _global ]/<patient>/{axial,coronal,sagittal,...}.png
                                                     + manifest.json

Two things this adds on top of make_fixed_mips.py:
  1) Runs ALL patients with ONE frozen protocol (so nothing varies case-to-case).
  2) Sweeps the image-budget ablation: pass several K values and it produces one
     output tree per K, otherwise identical -> image count is the only variable.
  3) Aggregates each patient's manifest.json into dataset_manifest.json per budget,
     a single file the training side can read.

Examples
--------
Dry run (just show what would happen):
    python run_fixed_mips.py --dry_run

Single budget (K=3 -> 9 images/patient) for all patients:
    python run_fixed_mips.py --slabs_per_axis 3 --include_global

Full ablation sweep (3, 6, 9, 12 images) in one go:
    python run_fixed_mips.py --slabs_per_axis 1 2 3 4 --include_global

A few patients only:
    python run_fixed_mips.py --slabs_per_axis 3 --patients patient_0001 patient_0002
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_MIP_SCRIPT = HERE / "make_fixed_mips.py"
DEFAULT_INPUT_ROOT = HERE / "preprocessed_brains"
DEFAULT_OUTPUT_ROOT = HERE / "mip_images"


# ----- patient discovery (mirrors the original batch runner) -----

def patient_name_from_folder(folder: Path) -> str:
    name = folder.name
    return name[:-len("_brain")] if name.endswith("_brain") else name


def choose_brain_nrrd(patient_dir: Path) -> Optional[Path]:
    nrrds = sorted(patient_dir.glob("*.nrrd"))
    candidates = [p for p in nrrds
                  if ".seg." not in p.name and "segmentation" not in p.name.lower()]
    preferred = [
        f"{patient_dir.name}.nrrd",
        f"{patient_name_from_folder(patient_dir)}_brain.nrrd",
        "brain_only.nrrd",
    ]
    for pref in preferred:
        for c in candidates:
            if c.name == pref:
                return c
    brain_named = [p for p in candidates if p.stem.endswith("_brain") or p.stem == "brain_only"]
    if len(brain_named) == 1:
        return brain_named[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def config_dirname(k: int, include_global: bool) -> str:
    return f"slabs{k}" + ("_global" if include_global else "")


def build_command(args, nrrd_path: Path, out_dir: Path, k: int) -> List[str]:
    cmd = [
        sys.executable, str(args.mip_script),
        "--nrrd_path", str(nrrd_path),
        "--out_root", str(out_dir),
        "--slabs_per_axis", str(k),
        "--orientation", args.orientation,
        "--filter_mode", args.filter_mode,
        "--low_hu", str(args.low_hu),
        "--high_hu", str(args.high_hu),
        "--window_level", str(args.window_level),
        "--window_width", str(args.window_width),
        "--norm", args.norm,
        "--p_low", str(args.p_low),
        "--p_high", str(args.p_high),
    ]
    if args.trim_area_frac > 0:
        cmd += ["--trim_area_frac", str(args.trim_area_frac)]
    if args.drop_bottom_frac > 0:
        cmd += ["--drop_bottom_frac", str(args.drop_bottom_frac)]
    if args.drop_top_frac > 0:
        cmd += ["--drop_top_frac", str(args.drop_top_frac)]
    if args.include_global:
        cmd.append("--include_global")
    if args.radiological_flip:
        cmd.append("--radiological_flip")
    if args.crop_to_foreground:
        cmd += ["--crop_to_foreground", "--crop_threshold_hu", str(args.crop_threshold_hu),
                "--crop_pad_mm", str(args.crop_pad_mm)]
    return cmd


def aggregate_manifests(budget_root: Path) -> Path:
    """Combine every <patient>/manifest.json under a budget root into one index."""
    patients = []
    for manifest_path in sorted(budget_root.glob("*/manifest.json")):
        patient = manifest_path.parent.name
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
        patients.append({
            "patient": patient,
            "image_budget": m.get("protocol", {}).get("image_budget"),
            "orientation_original": m.get("orientation", {}).get("original"),
            # image paths are relative to the patient folder; prefix with patient name
            "images": [f"{patient}/{img['file']}" for img in m.get("images", [])],
            "coverage": m.get("images", []),
        })
    out = budget_root / "dataset_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"budget_root": str(budget_root), "n_patients": len(patients),
                   "patients": patients}, f, indent=2)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Batch fixed-protocol MIP generation + manifest aggregation")
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--mip_script", type=Path, default=DEFAULT_MIP_SCRIPT)
    p.add_argument("--patients", nargs="*", default=None,
                   help="Optional patient names (space- or comma-separated).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N patients (after sorting/filtering). Handy for a quick test.")
    p.add_argument("--slabs_per_axis", type=int, nargs="+", default=[3],
                   help="One or more K values. Each becomes its own output tree (ablation sweep).")
    p.add_argument("--include_global", action="store_true")
    p.add_argument("--radiological_flip", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Write run_code.txt only; do not generate.")
    p.add_argument("--skip_existing", action="store_true")
    # FROZEN protocol (keep identical for the whole dataset)
    p.add_argument("--orientation", default="LPS")
    p.add_argument("--filter_mode", choices=["none", "low", "clip", "low_clip", "bandpass"], default="low_clip")
    p.add_argument("--low_hu", type=float, default=30.0)
    p.add_argument("--high_hu", type=float, default=420.0)
    p.add_argument("--window_level", type=float, default=200.0)
    p.add_argument("--window_width", type=float, default=470.0)
    p.add_argument("--norm", choices=["window", "per_image", "per_volume"], default="window",
                   help="HU->8bit mapping. Try 'per_image' to fix shiny-bottom/faded-top across slabs.")
    p.add_argument("--p_low", type=float, default=1.0)
    p.add_argument("--p_high", type=float, default=99.5)
    p.add_argument("--trim_area_frac", type=float, default=0.0,
                   help="Drop thin partial end-slices (try ~0.35).")
    p.add_argument("--drop_bottom_frac", type=float, default=0.0,
                   help="Drop fraction of Z-extent from the inferior end (keep small, <=0.15).")
    p.add_argument("--drop_top_frac", type=float, default=0.0)
    p.add_argument("--crop_to_foreground", action="store_true", default=True)
    p.add_argument("--crop_threshold_hu", type=float, default=-500.0)
    p.add_argument("--crop_pad_mm", type=float, default=10.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.mip_script = args.mip_script.resolve()

    if not args.input_root.exists():
        print(f"Input root not found: {args.input_root}", file=sys.stderr); return 1
    if not args.mip_script.exists():
        print(f"MIP script not found: {args.mip_script}", file=sys.stderr); return 1

    patient_dirs = sorted(d for d in args.input_root.iterdir() if d.is_dir())
    if args.patients:
        wanted = {p for item in args.patients for p in item.split(",") if p}
        patient_dirs = [d for d in patient_dirs if patient_name_from_folder(d) in wanted]
    if args.limit is not None:
        patient_dirs = patient_dirs[:args.limit]
    if not patient_dirs:
        print(f"No patient folders under {args.input_root}", file=sys.stderr); return 1

    failures: List[str] = []
    for k in args.slabs_per_axis:
        budget_root = args.output_root / config_dirname(k, args.include_global)
        budget_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Budget K={k} (image_budget={3*k + (3 if args.include_global else 0)}) "
              f"-> {budget_root} ===")

        for pdir in patient_dirs:
            patient = patient_name_from_folder(pdir)
            nrrd = choose_brain_nrrd(pdir)
            out_dir = budget_root / patient
            out_dir.mkdir(parents=True, exist_ok=True)

            if nrrd is None:
                failures.append(f"K{k} {patient}: no brain-only NRRD found"); continue

            cmd = build_command(args, nrrd, out_dir, k)
            (out_dir / "run_code.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

            if args.skip_existing and (out_dir / "manifest.json").exists():
                print(f"[skip] {patient}"); continue
            if args.dry_run:
                print(f"[dry-run] {patient}: {nrrd.name}"); continue

            print(f"[run] {patient}: {nrrd.name}")
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                failures.append(f"K{k} {patient}: exit {r.returncode}")

        if not args.dry_run:
            idx = aggregate_manifests(budget_root)
            print(f"Aggregated dataset manifest -> {idx}")

    if failures:
        print("\nFailures:")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 1
    print("\nAll MIP jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
