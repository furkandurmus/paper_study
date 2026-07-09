# Preprocessing — CTA → fixed-protocol MIP images

Turns 3D **brain-only CTA** NRRD volumes into a small, **fixed** set of 2D
maximum-intensity-projection (MIP) PNGs that the VLM reads. The protocol is deterministic and
identical for every patient — any per-patient variation is a confound. See the frozen settings below.

> **No patient data is included in this repo.** Raw DICOM (`raw_stroke_images/`), brain-only NRRD
> volumes (`preprocessed_brains/`), and generated MIPs (`mip_images*/`) are patient data (PHI, also in
> the DICOM headers) and are **git-ignored**. Provide your own inputs.

## Dependencies (install separately — NOT in the top-level `requirements.txt`)
```bash
pip install SimpleITK numpy Pillow
```

## The two scripts
- **`make_fixed_mips.py`** — one brain-only NRRD → deterministic **equal-count slabs** per axis
  (`--slabs_per_axis K` ⇒ 3·K slab images: axial / coronal / sagittal), optionally + 3 global MIPs, plus
  a per-patient **`manifest.json`** recording the protocol, orientation, spacing, crop box, and per-image
  coverage metadata (view, slab index, physical extent, coverage band, side).
- **`run_fixed_mips.py`** — batch wrapper: runs **every** patient with **one frozen protocol**, can
  **sweep the image-budget ablation** (pass several K values), and aggregates every per-patient
  manifest into a single **`dataset_manifest.json`** per budget.

## Frozen protocol (do not vary per patient)
| setting | value | why |
|---|---|---|
| `--slabs_per_axis` | **4** (+ globals) | image-budget knob; K=4's thin quarters keep the inferior slab below the circle of Willis |
| `--norm` | **window** | consistent brightness across patients; `per_image` amplifies background noise |
| `--low_hu` / `--high_hu` | **30 / 420** | keep faint distal (M2) vessels; cap bright calcium |
| `--window_level` / `--window_width` | 200 / 470 | display window used by `--norm window` |
| `--filter_mode` | `low_clip` | removes dim tissue, caps calcium; avoids `bandpass` fake-occlusion risk |
| `--orientation` | **LPS** | consistent patient left/right (the reward grades laterality) |
| `--crop_to_foreground` | on | trims empty background |
| dataset build | **exclude `axial_slab_00`** | inferior quarter (cerebellum / base), no MCA — drop at index-build time, not by deleting files |

## Usage
Single patient:
```bash
python preprocessing/make_fixed_mips.py \
    --nrrd_path /path/to/<patient>_brainOnly.nrrd \
    --out_root  mip_images/ \
    --slabs_per_axis 4 --include_global \
    --norm window --low_hu 30 --orientation LPS --crop_to_foreground
```
All patients + ablation sweep (K = 1,2,3,4):
```bash
python preprocessing/run_fixed_mips.py \
    --input_root  preprocessed_brains/ \
    --output_root mip_images/ \
    --slabs_per_axis 1 2 3 4 --include_global \
    --norm window --low_hu 30 --orientation LPS
```
`--dry_run` writes the exact command per patient without generating; `--skip_existing` resumes;
`--limit N` / `--patients <name...>` select a subset.

## Output layout
```
mip_images/slabs<K>[_global]/<patient>/
    axial/     axial_slab_00_of_K.png ...
    coronal/   coronal_slab_00_of_K.png ...
    sagittal/  sagittal_slab_00_of_K.png ...
    global/    global_{axial,coronal,sagittal}.png     (with --include_global)
    manifest.json
dataset_manifest.json                                  (aggregated by run_fixed_mips.py)
```

## Notes
- These are the **current, fixed-protocol** generators. The older sliding-window scripts
  (`generate_mips_from_nrrt.py`, `run_mips_from_preprocessed.py`) at the repo root are superseded.
- Upstream (in 3D Slicer): export a **brain-only** CTA (e.g. TotalSegmentator brain mask, eroded ~3 mm
  to drop residual skull-base bone) as `*_brainOnly.nrrd`.
