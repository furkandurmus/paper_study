#!/usr/bin/env python3
"""
Fixed-protocol CTA-MIP generator for the stroke VLM study.

Why this exists
---------------
The original sliding-window generator produced ~50 overlapping slabs per patient
and a variable count per case. For a clean alignment study the image set fed to
the VLM must be:

  * FIXED & DETERMINISTIC  - identical layout/count for every patient, train and
    test, with no per-case human cherry-picking (cherry-picking leaks the label
    and is not deployable at inference).
  * ABLATABLE             - one knob (`--slabs_per_axis K`) controls the image
    budget: total slab images = 3*K  (K=1 -> 3, 2 -> 6, 3 -> 9, 4 -> 12),
    optionally + 3 global MIPs.
  * ORIENTATION-SAFE      - the volume is reoriented to a canonical anatomical
    orientation so patient left/right is consistent across all cases. This is
    critical: the reward grades laterality, and a single flipped volume silently
    corrupts the side signal.
  * SELF-DESCRIBING       - a manifest.json records, per image, the physical
    extent and a coarse coverage band (+ side for sagittal). This gives the
    coverage-aware calibration reward its labels automatically, with no manual
    coverage annotation.

Slabs are EQUAL-COUNT (not equal-thickness): each axis extent is split into K
contiguous groups of slices. Count/layout is therefore identical across patients;
physical thickness adapts to head size. (For tighter cross-patient anatomical
correspondence, register to a template first - see NOTES at the bottom.)

Install:
    pip install SimpleITK numpy pillow

Single case:
    python make_fixed_mips.py \
        --nrrd_path /path/CTA_brainOnly.nrrd \
        --out_root  /path/out/<patient> \
        --slabs_per_axis 3 \
        --include_global

Freeze ONE HU/window protocol for the whole dataset and never change it between
patients (a per-patient difference is a confound).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image


# ============================ LOAD + ORIENT ============================

def load_and_orient(nrrd_path: Path, orientation: str = "LPS"):
    """
    Load an NRRD volume and reorient to a canonical anatomical orientation so
    that voxel axes mean the same thing for every patient.

    With orientation "LPS" (SimpleITK/DICOM default), after GetArrayFromImage the
    numpy array is (Z, Y, X) where increasing:
        Z -> superior, Y -> posterior, X -> patient LEFT.
    """
    if not nrrd_path.exists():
        raise FileNotFoundError(f"NRRD not found: {nrrd_path}")

    img = sitk.ReadImage(str(nrrd_path))
    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection()
    )
    img = sitk.DICOMOrient(img, orientation)

    arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    sx, sy, sz = (float(v) for v in img.GetSpacing())          # (X, Y, Z) spacing
    # Convert to (Y, X, Z) = (H, W, N) to match the projection math below.
    vol = np.transpose(arr_zyx, (1, 2, 0))
    meta = {
        "orientation_canonical": orientation,
        "orientation_original": orig_orient,
        "spacing_mm": {"dx": sx, "dy": sy, "dz": sz},
        "shape_HWN": list(vol.shape),
    }
    return vol, sx, sy, sz, meta


# ============================ CROP + HU FILTER ============================

def crop_to_foreground(vol, dy, dx, dz, threshold_hu=-500.0, pad_mm=10.0):
    fg = vol > threshold_hu
    ys, xs, zs = np.where(fg)
    if ys.size == 0:
        return vol, [0, vol.shape[0] - 1, 0, vol.shape[1] - 1, 0, vol.shape[2] - 1]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    z0, z1 = int(zs.min()), int(zs.max())
    py, px, pz = int(round(pad_mm / dy)), int(round(pad_mm / dx)), int(round(pad_mm / dz))
    y0, y1 = max(0, y0 - py), min(vol.shape[0] - 1, y1 + py)
    x0, x1 = max(0, x0 - px), min(vol.shape[1] - 1, x1 + px)
    z0, z1 = max(0, z0 - pz), min(vol.shape[2] - 1, z1 + pz)
    return vol[y0:y1 + 1, x0:x1 + 1, z0:z1 + 1], [y0, y1, x0, x1, z0, z1]


def trim_z(vol, floor, area_frac=0.0, drop_bottom_frac=0.0, drop_top_frac=0.0):
    """
    Trim the inferior/superior ends of the axial (Z) range so the slab stack is
    defined over relevant brain only. vol is (H, W, N) with N = Z; under LPS,
    Z index 0 = inferior (bottom), Z max = superior (top).

      area_frac:        keep only the contiguous Z-range whose per-slice brain
                        area >= area_frac * (max slice area). Removes thin partial
                        end-slices (fixes "some patients full head, some partial").
      drop_bottom_frac: additionally drop this fraction of the kept Z-extent from
                        the INFERIOR end (your "cut the first slices" idea).
      drop_top_frac:    same, from the SUPERIOR end.

    NOTE: this trims by position/area, not anatomy, so it does NOT specifically
    remove the cerebellum. The cleaner fix for that is a supratentorial-only brain
    mask upstream. Keep drop_bottom_frac small (<=0.15) so you don't clip the
    circle of Willis, where M1 occlusions sit.
    """
    H, W, N = vol.shape
    z0, z1 = 0, N - 1
    if area_frac > 0:
        area = (vol > floor).sum(axis=(0, 1))
        if area.max() > 0:
            keep = np.where(area >= area_frac * float(area.max()))[0]
            if keep.size:
                z0, z1 = int(keep.min()), int(keep.max())
    span = z1 - z0 + 1
    z0 = z0 + int(round(drop_bottom_frac * span))
    z1 = z1 - int(round(drop_top_frac * span))
    z0 = max(0, min(z0, N - 1))
    z1 = max(z0, min(z1, N - 1))
    return vol[:, :, z0:z1 + 1], [z0, z1]


def apply_hu_filter(vol, mode, low_hu, high_hu, background_hu=-1000.0):
    out = vol.astype(np.float32, copy=True)
    if mode == "none":
        return out
    if mode in ("low", "low_clip"):
        out[out < low_hu] = background_hu
    if mode in ("clip", "low_clip"):
        out[out > high_hu] = high_hu
    if mode == "bandpass":
        out[(out < low_hu) | (out > high_hu)] = background_hu
    return out


# ============================ WINDOW + SAVE ============================

def _foreground_percentiles(arr, p_low, p_high, floor=-100.0):
    """Percentiles over non-background voxels (background is set to -1000 by the HU filter)."""
    fg = arr[arr > floor]
    if fg.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(fg, p_low))
    hi = float(np.percentile(fg, p_high))
    return (lo, hi if hi > lo else lo + 1.0)


def to_uint8(arr_hu, norm: Dict):
    """
    Map HU -> 8-bit using one of three strategies (norm['mode']):
      window      : fixed HU window (wl/ww). Same mapping for everything.
      per_volume  : percentile stretch from the whole volume (vlo/vhi precomputed).
                    Keeps brightness comparable across slabs of one patient.
      per_image   : percentile stretch computed from THIS image only.
                    Most uniform vessel visibility across slabs (bottom stops
                    blowing out, top stops washing out). Left/right asymmetry
                    within an image is preserved (uniform scaling).
    """
    mode = norm["mode"]
    if mode == "window":
        lo = norm["wl"] - norm["ww"] / 2.0
        hi = norm["wl"] + norm["ww"] / 2.0
    elif mode == "per_volume":
        lo, hi = norm["vlo"], norm["vhi"]
    elif mode == "per_image":
        lo, hi = _foreground_percentiles(arr_hu, norm["p_low"], norm["p_high"])
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    img = np.clip(arr_hu.astype(np.float32), lo, hi)
    return ((img - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)


def save_png(arr_hu, out_path: Path, spacing_hw: Tuple[float, float], norm: Dict,
             flip_lr: bool = False):
    """spacing_hw = (vertical_mm, horizontal_mm) for aspect correction."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = to_uint8(arr_hu, norm)
    if flip_lr:
        png = png[:, ::-1]
    img = Image.fromarray(png)
    h_mm, w_mm = spacing_hw
    if abs(h_mm - w_mm) > 1e-6:
        ow, oh = img.size
        base = min(w_mm, h_mm)
        img = img.resize((int(round(ow * w_mm / base)), int(round(oh * h_mm / base))),
                         resample=Image.BICUBIC)
    img.save(out_path)


# ============================ EQUAL-COUNT SLABS ============================

def equal_groups(n: int, k: int) -> List[Tuple[int, int]]:
    """Split range [0, n) into k contiguous (start, end) groups as evenly as possible."""
    k = max(1, min(k, n))
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def axial_band(frac_center: float) -> str:
    return "inferior" if frac_center < 1 / 3 else ("superior" if frac_center > 2 / 3 else "middle")


def coronal_band(frac_center: float) -> str:
    return "anterior" if frac_center < 1 / 3 else ("posterior" if frac_center > 2 / 3 else "middle")


def sagittal_side(frac_center: float) -> str:
    # X increases toward patient LEFT under LPS. Tune the midline band if needed.
    return "right" if frac_center < 0.4 else ("left" if frac_center > 0.6 else "midline")


def generate(vol, dx, dy, dz, out_root: Path, K: int, norm: Dict,
             include_global: bool, flip_lr: bool) -> List[Dict]:
    """Generate the fixed image set; return per-image coverage records."""
    H, W, N = vol.shape  # (Y, X, Z)
    records: List[Dict] = []

    def record(file, view, axis, idx, total, lo_idx, hi_idx, axis_len, axis_mm):
        c = ((lo_idx + hi_idx) / 2.0) / max(axis_len, 1)
        rec = {
            "file": str(file.relative_to(out_root)),
            "view": view, "axis": axis,
            "slab_index": idx, "slabs_total": total,
            "extent_mm": [round(lo_idx * axis_mm, 2), round(hi_idx * axis_mm, 2)],
            "extent_frac": [round(lo_idx / max(axis_len, 1), 4), round(hi_idx / max(axis_len, 1), 4)],
        }
        if axis == "Z":
            rec["coverage_band"] = axial_band(c); rec["side"] = None
        elif axis == "Y":
            rec["coverage_band"] = coronal_band(c); rec["side"] = None
        else:  # X
            rec["coverage_band"] = None; rec["side"] = sagittal_side(c)
        records.append(rec)

    # ----- AXIAL slabs: project along Z -> (Y, X) -----
    for i, (z0, z1) in enumerate(equal_groups(N, K)):
        mip = vol[:, :, z0:z1].max(axis=2)
        f = out_root / "axial" / f"axial_slab_{i:02d}_of_{K}.png"
        save_png(mip, f, (dy, dx), norm, flip_lr)
        record(f, "axial", "Z", i, K, z0, z1, N, dz)

    # ----- CORONAL slabs: project along Y -> (Z, X) -----
    for i, (y0, y1) in enumerate(equal_groups(H, K)):
        mip = vol[y0:y1, :, :].max(axis=0).T
        f = out_root / "coronal" / f"coronal_slab_{i:02d}_of_{K}.png"
        save_png(mip, f, (dz, dx), norm, flip_lr)
        record(f, "coronal", "Y", i, K, y0, y1, H, dy)

    # ----- SAGITTAL slabs: project along X -> (Z, Y) -----
    for i, (x0, x1) in enumerate(equal_groups(W, K)):
        mip = vol[:, x0:x1, :].max(axis=1).T
        f = out_root / "sagittal" / f"sagittal_slab_{i:02d}_of_{K}.png"
        # NOTE: sagittal view is intrinsically a side view; flip_lr does not apply.
        save_png(mip, f, (dz, dy), norm, flip_lr=False)
        record(f, "sagittal", "X", i, K, x0, x1, W, dx)

    # ----- Optional global MIPs (one per axis) -----
    if include_global:
        gax = vol.max(axis=2)
        save_png(gax, out_root / "global" / "global_axial.png", (dy, dx), norm, flip_lr)
        records.append({"file": "global/global_axial.png", "view": "global_axial",
                        "axis": "Z", "slab_index": 0, "slabs_total": 1,
                        "extent_frac": [0.0, 1.0], "coverage_band": "all", "side": None})
        gco = vol.max(axis=0).T
        save_png(gco, out_root / "global" / "global_coronal.png", (dz, dx), norm, flip_lr)
        records.append({"file": "global/global_coronal.png", "view": "global_coronal",
                        "axis": "Y", "slab_index": 0, "slabs_total": 1,
                        "extent_frac": [0.0, 1.0], "coverage_band": "all", "side": None})
        gsa = vol.max(axis=1).T
        save_png(gsa, out_root / "global" / "global_sagittal.png", (dz, dy), norm, flip_lr=False)
        records.append({"file": "global/global_sagittal.png", "view": "global_sagittal",
                        "axis": "X", "slab_index": 0, "slabs_total": 1,
                        "extent_frac": [0.0, 1.0], "coverage_band": None, "side": "both"})

    return records


# ============================ MAIN ============================

def parse_args():
    p = argparse.ArgumentParser(description="Fixed-protocol CTA-MIP generator + coverage manifest")
    p.add_argument("--nrrd_path", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--slabs_per_axis", type=int, default=3,
                   help="K equal slabs per axis. Image budget = 3*K (+3 if --include_global). Ablation knob.")
    p.add_argument("--include_global", action="store_true", help="Also emit 3 global MIPs.")
    p.add_argument("--orientation", default="LPS", help="Canonical orientation (keep fixed for the whole dataset).")
    p.add_argument("--radiological_flip", action="store_true",
                   help="Flip axial/coronal/global columns so patient-left displays on image-right (radiological convention). Recorded in manifest.")
    # FROZEN protocol defaults - pick once for the whole dataset.
    p.add_argument("--filter_mode", choices=["none", "low", "clip", "low_clip", "bandpass"], default="low_clip")
    p.add_argument("--low_hu", type=float, default=30.0)
    p.add_argument("--high_hu", type=float, default=420.0)
    p.add_argument("--background_hu", type=float, default=-1000.0)
    p.add_argument("--window_level", type=float, default=200.0)
    p.add_argument("--window_width", type=float, default=470.0)
    p.add_argument("--norm", choices=["window", "per_image", "per_volume"], default="window",
                   help="HU->8bit mapping. 'window'=fixed HU window (default, current behaviour). "
                        "'per_image'=percentile stretch per image (most uniform vessel visibility "
                        "across slabs; fixes shiny-bottom/faded-top). 'per_volume'=percentile stretch "
                        "from the whole volume (keeps brightness comparable across a patient's slabs).")
    p.add_argument("--p_low", type=float, default=1.0, help="Low percentile for per_image/per_volume norm.")
    p.add_argument("--p_high", type=float, default=99.5, help="High percentile for per_image/per_volume norm.")
    p.add_argument("--crop_to_foreground", action="store_true")
    p.add_argument("--crop_threshold_hu", type=float, default=-500.0)
    p.add_argument("--crop_pad_mm", type=float, default=10.0)
    # Inferior/superior axial trim (off by default). See trim_z() docstring.
    p.add_argument("--trim_area_frac", type=float, default=0.0,
                   help="Keep only Z-slices with brain area >= this fraction of the max (drops thin partial end-slices). Try ~0.35.")
    p.add_argument("--drop_bottom_frac", type=float, default=0.0,
                   help="Drop this fraction of the Z-extent from the inferior end. Keep small (<=0.15) to avoid clipping the circle of Willis.")
    p.add_argument("--drop_top_frac", type=float, default=0.0,
                   help="Drop this fraction of the Z-extent from the superior end.")
    return p.parse_args()


def main():
    a = parse_args()
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vol, dx, dy, dz, meta = load_and_orient(Path(a.nrrd_path), a.orientation)

    crop_bbox = None
    if a.crop_to_foreground:
        vol, crop_bbox = crop_to_foreground(vol, dy, dx, dz, a.crop_threshold_hu, a.crop_pad_mm)

    trim_z_range = None
    if a.trim_area_frac > 0 or a.drop_bottom_frac > 0 or a.drop_top_frac > 0:
        vol, trim_z_range = trim_z(vol, a.crop_threshold_hu, a.trim_area_frac,
                                   a.drop_bottom_frac, a.drop_top_frac)

    vol = apply_hu_filter(vol, a.filter_mode, a.low_hu, a.high_hu, a.background_hu)

    norm = {"mode": a.norm, "wl": a.window_level, "ww": a.window_width,
            "p_low": a.p_low, "p_high": a.p_high, "vlo": None, "vhi": None}
    if a.norm == "per_volume":
        norm["vlo"], norm["vhi"] = _foreground_percentiles(vol, a.p_low, a.p_high)

    records = generate(
        vol, dx, dy, dz, out_root,
        K=a.slabs_per_axis, norm=norm,
        include_global=a.include_global, flip_lr=a.radiological_flip,
    )

    manifest = {
        "nrrd_path": str(a.nrrd_path),
        "protocol": {
            "slabs_per_axis": a.slabs_per_axis,
            "include_global": a.include_global,
            "filter_mode": a.filter_mode, "low_hu": a.low_hu, "high_hu": a.high_hu,
            "window_level": a.window_level, "window_width": a.window_width,
            "norm": a.norm, "norm_percentiles": [a.p_low, a.p_high],
            "radiological_flip": a.radiological_flip,
            "image_budget": 3 * max(1, a.slabs_per_axis) + (3 if a.include_global else 0),
        },
        "orientation": {"canonical": meta["orientation_canonical"], "original": meta["orientation_original"]},
        "spacing_mm": meta["spacing_mm"],
        "shape_HWN_after_orient": meta["shape_HWN"],
        "crop_bbox_yxz": crop_bbox,
        "trim_z_range": trim_z_range,
        "trim": {"area_frac": a.trim_area_frac, "drop_bottom_frac": a.drop_bottom_frac,
                 "drop_top_frac": a.drop_top_frac},
        "images": records,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(records)} images + manifest.json to {out_root} "
          f"(orientation {meta['orientation_original']} -> {a.orientation}, budget {manifest['protocol']['image_budget']})")


if __name__ == "__main__":
    main()


# ============================ NOTES ============================
#  * VERIFY ORIENTATION on a few cases: open an axial slab and confirm patient
#    left/right matches your expectation given --radiological_flip. The side
#    reward depends on this. The manifest records the original orientation per
#    case so you can audit any outliers.
#  * COVERAGE bands here are coarse (thirds along each axis; sagittal -> side).
#    Good enough for a calibration proxy ("is the labeled vessel's territory
#    plausibly shown?"). For precise vascular-territory coverage, register each
#    volume to a template/atlas and intersect slab extents with territory masks.
#  * ISOTROPIC RESAMPLING is intentionally omitted (keeps native resolution). If
#    you want exactly comparable physical geometry across patients, resample to
#    isotropic spacing with sitk.ResampleImageFilter before generate().
#  * ABLATION: run with --slabs_per_axis in {1,2,3,4} (optionally --include_global)
#    into separate output roots; the image set is otherwise identical, so image
#    budget is the only variable.
