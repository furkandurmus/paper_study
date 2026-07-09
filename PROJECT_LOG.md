# Project Log — CTA-MIP VLM Alignment Study

_A running record of the work done in this repository. Preprocessing is documented in the most
detail (Section 5), since that is where most of the hands-on decisions were made._

---

## 1. Project in one paragraph

The goal is to fine-tune a vision-language model (VLM) to read CTA-MIP images of the head and write
a short stroke report, then **align** it with preference-based methods (DPO / KTO / ORPO / GRPO) and
**benchmark** those methods fairly. The clinical target is narrowed to **middle cerebral artery
(MCA) occlusion**: whether an occlusion is present, on which **side** (left/right), and — as a
secondary/bonus target — the **segment** (M1 vs M2). The study's intended contribution is not the
optimizer horse-race itself but an **image-grounded, clinically-verifiable reward** plus a
preference-construction recipe, evaluated across all four alignment algorithms to show it is
optimizer-agnostic.

Two supporting documents were produced alongside this log:
- `REVIEW.md` — critical review of the original codebase.
- `RESEARCH.md` (+ `scripts/build_research_docx.js`) — literature review and a blueprint for the study.
- `HANDOFF_generate_mips.md` — instructions for another assistant to run the MIP generation.

**Environment note:** during these sessions the assistant could edit and read files in the repo but
could not execute code (the sandbox shell cannot reach the WSL folder). All scripts are therefore run
by the user in their own WSL environment; the assistant validated images by opening them directly.

---

## 2. Codebase review (`REVIEW.md`)

The original repository was a solid training framework but had weaknesses as an *alignment study*.
Key findings:

- **The reward was "image-blind."** `rule_based_score` combined structural regex + keyword rubric +
  ROUGE similarity to the reference — none of which look at the image. Alignment would optimize
  report format and lexical overlap, not diagnostic correctness (a reward-hacking risk).
- **Stub objectives.** `llm_judge_score` returned a random number; the GRPO reward rewarded output
  length. Both would produce meaningless training if used.
- **SFT label masking** was likely misaligned for multimodal inputs (prompt-length token counting
  vs. image-token expansion) — flagged to verify by decoding supervised spans.
- **Text-only KTO/GRPO paths**, brittle keyword rubric, non-standard BLEU/ROUGE, non-deterministic
  evaluation decoding, and "reference-as-chosen" preference construction that collapses toward SFT.
- The shipped data was a single dummy case duplicated 5× (smoke test only).

---

## 3. Reward-system work (code)

Based on the decision to ground the reward in **structured labels + an image-capable judge** (and to
get **DPO** right first), the following was implemented:

- **`src/scoring.py`** — added a **label-grounded fact score**:
  - `parse_labels()` normalizes case labels into `{anomaly, vessel, side}`.
  - `parse_report_claims()` extracts what a report actually claims (occlusion present/absent, vessel,
    side, hedging, overconfidence) using clause-level negation handling.
  - `fact_based_report()` / `fact_based_score()` score a report against the labels on four axes:
    **detection (0.45), vessel (0.20), side (0.25), calibration (0.10)**, with a contradiction
    penalty. Laterality is weighted heavily because wrong-side errors misdirect treatment.
  - The random `llm_judge_score` placeholder now raises instead of injecting noise.
- **`src/reward.py`** — new file:
  - `LLMJudge` — a multimodal LLM-as-judge (OpenAI-compatible) that actually sends the CTA images and
    returns correctness/grounding/laterality/calibration scores. It **abstains cleanly** (no random
    rewards) when no API key/model is configured.
  - `CompositeReward` — blends fact + judge + structure + similarity, **renormalizing** over whichever
    components are available for a case.
- **`scripts/build_preference_data.py`** — new scoring methods `composite` (default) and `fact_based`;
  `--images_root`, `--use_llm_judge`, `--judge_model`, and weight flags; the reference report is now
  scored like any other candidate (no automatic 1.0) to stop DPO collapsing back to SFT.
- **`tests/test_reward.py`** — CPU-only tests asserting the scorer ranks correct > wrong-side >
  wrong-vessel > garbage > confident-miss, and that the composite still works when the judge abstains.

---

## 4. Study-design decisions

- **Benchmark design:** compare DPO/KTO/ORPO/GRPO under **one shared reward** (each algorithm gets the
  same signal converted to its native format — pairs / binary / online scalar). This avoids
  confounding the algorithm with the reward, and makes the reward the contribution.
- **Novel method (working name VC-CVPO):** verifiable clinical core reward + multi-view consistency
  (axial/coronal/sagittal agreement) + counterfactual/image-corruption negatives + coverage-aware
  calibration. The four-algorithm comparison is the *evidence* the reward is optimizer-agnostic.
- **Clinical scope:** MCA **M1/M2** only.
  - Groups: Left M1, Right M1, Left M2, Right M2, No-MCA-occlusion (normal), Stroke-symptoms-but-no-LVO.
  - **Both sides** included (to learn/measure laterality).
  - **M1 vs M2 is a secondary/bonus target** (hard to see on MIPs; lightly weighted).
  - **Negatives are truly occlusion-free** — patients with non-MCA occlusions (ICA, basilar, …) are
    excluded, so "normal" is unambiguous.
  - Symptoms are **not** put in the prompt, so the two negative groups look identical to the model and
    simply serve as true-negatives; `clinical_group` is kept as a metadata tag for analysis.
- **Dataset:** under 500 cases, with **both** structured labels and reference reports (no radiologist
  preference pairs). Small N pushes toward LoRA/QLoRA, offline methods as primary, counterfactual data
  augmentation, and cross-validation with confidence intervals.
- **Label schema (per patient):**
  ```
  occlusion_present : true / false
  side              : "left" / "right" / null
  segment           : "M1" / "M2" / null      (bonus)
  clinical_group    : "normal" / "symptoms_no_lvo"   (metadata only)
  ```

---

## 5. Preprocessing — MIP generation (main focus)

### 5.1 Starting point

Two original scripts generated MIPs from 3D-Slicer-exported **brain-only** CTA NRRD volumes:
- `generate_mips_from_nrrt.py` — sliding-window slabs (`slab_mm=20, step_mm=10`) + global MIPs.
- `run_mips_from_preprocessed.py` — batch wrapper.

Problems identified:
- **Too many images:** ~50 overlapping slabs per patient, and a **variable count** per patient.
- **No orientation handling** — a risk for a laterality-graded study.
- **Inconsistent settings** between the two scripts (`low_hu` 80 vs 30, etc.).
- Reliance on **manual slice selection** would leak the label and not be reproducible at test time.

### 5.2 New fixed-protocol generator — `preprocessing/make_fixed_mips.py`

A single, deterministic, label-blind protocol producing a small fixed image set per patient:

- **Equal-count slabs:** each axis (axial/coronal/sagittal) is split into **K equal slabs**
  (`--slabs_per_axis K`). Total slab images = 3·K; layout is identical for every patient, so **image
  count is a clean ablation knob** (K=1→3, 2→6, 3→9, 4→12), optionally + 3 global MIPs.
- **Canonical orientation:** every volume is reoriented to **LPS** so patient left/right is consistent
  across cases. `--radiological_flip` flips display columns if needed; the manifest records the
  original and canonical orientation for auditing.
- **Foreground crop** (`--crop_to_foreground`) to trim empty background.
- **HU filtering** before projection (`--filter_mode`, default `low_clip`): removes dim brain tissue
  and caps very bright calcium. **`bandpass` is intentionally avoided** — it can erase vessel pixels
  and create fake "occlusions."
- **Intensity → 8-bit mapping** (`--norm`): `window` (fixed HU window; **chosen** — see 5.4),
  `per_image` (percentile stretch per image), `per_volume` (percentile stretch per patient).
- **Optional Z-trim** of the inferior/superior ends (`--trim_area_frac`, `--drop_bottom_frac`,
  `--drop_top_frac`) — see 5.5 for why we ended up handling the inferior region differently.
- **`manifest.json`** per patient: protocol used, orientation, spacing, crop box, and for **each image**
  its view, slab index, physical extent, a coarse **coverage band** (inferior/middle/superior for
  axial; anterior/middle/posterior for coronal) and **side** (left/right/midline for sagittal). This
  metadata feeds the coverage-aware calibration reward automatically — no manual coverage labeling.

### 5.3 Batch runner — `preprocessing/run_fixed_mips.py`

- Iterates patient subfolders, picks the brain-only NRRD, runs `make_fixed_mips.py` with **one frozen
  protocol**.
- **Sweeps the ablation:** pass several K values (`--slabs_per_axis 1 2 3 4`) to produce one output
  tree per budget, otherwise identical.
- Aggregates every patient's `manifest.json` into a per-budget **`dataset_manifest.json`**.
- `--limit N` for quick tests; `--patients` to select cases.

### 5.4 Frozen protocol settings (current)

| Setting | Value | Reason |
|---|---|---|
| `filter_mode` | `low_clip` | Removes tissue, caps calcium; avoids bandpass fake-occlusion risk |
| `low_hu` | **30** | Keeps faint distal (M2) vessels; skull already removed |
| `high_hu` | 420 | Caps bright calcium |
| `window_level / width` | 200 / 470 | Display window (used by `--norm window`) |
| `--norm` | **window** | Cleaner + consistent across patients; `per_image` added background noise |
| `orientation` | LPS | Consistent left/right |
| `crop_to_foreground` | on | Trim background |
| image budget | ablation | Currently exploring K=4 (see 5.5) |

**Freeze these for the entire dataset** — any per-patient variation is a confound.

### 5.5 Problems encountered and how they were resolved

1. **Bright white bone arcs (worst on the inferior slab).**
   Cause: the brain mask included residual skull-base bone. **Fix (upstream, in 3D Slicer):** switch
   the mask from *grow* to **shrink/erode**. The user eroded the brain masks by **3 mm**, which removed
   the bone without losing important vessels. Confirmed clean afterward. (Rule of thumb: 1–3 mm is
   safe; heavier erosion risks trimming cortical/distal vessels.)

2. **Contrast uneven across slabs — bottom "shiny," top "faded."**
   Cause: a single fixed window applied to slabs with very different vessel density; bottom slabs
   saturate, top slabs wash out. Compared `--norm window` vs `--norm per_image` on real patients
   (a representative case). **Decision: use `window`.** `per_image` made faint upper-slab vessels brighter but
   amplified background noise and is less reproducible. `window` is cleaner and keeps brightness
   consistent across patients; the slightly faint vertex slab is acceptable (little MCA content there).

3. **Inferior slab catches cerebellum / posterior fossa / partial brain — and inconsistently between
   patients.**
   Cause: brain masks extend to different inferior levels, so the "bottom slab" lands on different
   anatomy per patient. Tried `--trim_area_frac` / `--drop_bottom_frac`; these are only partial fixes
   (area-trim doesn't remove the large cerebellum; a big bottom-crop risks clipping the circle of
   Willis).
   **Resolution — validated on data:** at **K=4**, the **axial slab_00 is reliably the inferior
   quarter** (cerebellum / posterior fossa / temporal base, no MCA), and the **circle of Willis sits in
   axial slab_01**. Verified across 6 patients (4 with occlusion, 2 healthy) — in all, axial slab_00 had no MCA/circle-of-Willis. So the plan
   is to **discard the axial slab_00** for both healthy and sick cases. Notes:
   - **Axial only.** Keep coronal slab_00 (anterior) and sagittal slab_00 (lateral) — those are
     relevant.
   - **Why K=4 not K=3:** thicker K=3 thirds sometimes reached the circle of Willis in the bottom slab
     (seen in one case earlier); K=4's thinner quarters keep the bottom slab safely below it.
   - **Do it at dataset-build time** (exclude `axial_slab_00` when building the training index), not by
     deleting files — reproducible and reversible.
   - **QC still needed:** sweep ~10–15 more patients' axial slab_00 to confirm none contain the circle
     of Willis (failure mode: a volume cropped so high it lacks cerebellum).

### 5.6 Current preprocessing status

- Image **quality is good** after the mask erosion + `window` normalization: clean background, crisp
  circle of Willis and MCA branches on the diagnostic slabs.
- The **pipeline is essentially ready**; the generation itself is deterministic and reproducible.
- **Chosen approach going forward:** K=4 slabs per axis + global MIPs, **discard axial slab_00**,
  `--norm window`, `low_hu 30`, LPS orientation.
- Two example output trees exist: `preprocessing/mip_images_k4_window_no_trim/` (sick) and
  `preprocessing/mip_images_healthy_k4_window_no_trim/` (healthy).

### 5.7 Open preprocessing items (before scaling to the full dataset)

1. **Verify left/right orientation** on a case with a known occluded side (the one check that protects
   the laterality signal). Not yet confirmed.
2. **QC axial slab_00** across more patients to confirm the drop is universally safe.
3. **Finalize the image budget.** K=4 + drop-slab_00 + globals = 14 images, which is heavy for a VLM on
   <500 cases; consider fewer coronal/sagittal slabs. Confirm it fits GPU memory with
   `scripts/probe_dpo_capacity.py`.
4. **Check native PNG resolution** is adequate for subtle M2 branches.
5. **Confirm per-group counts** after excluding non-MCA occlusions (enough negatives and enough of each
   side).

---

## 6. Files created / modified in this repo

- `REVIEW.md` — codebase review.
- `RESEARCH.md`, `scripts/build_research_docx.js` — literature review + Word export script.
- `HANDOFF_generate_mips.md` — run instructions for another assistant.
- `PROJECT_LOG.md` — this document.
- `src/scoring.py` — added fact-based label-grounded scoring; disabled random judge.
- `src/reward.py` — new: `LLMJudge`, `CompositeReward`.
- `scripts/build_preference_data.py` — composite/fact_based scoring, image-grounded options.
- `tests/test_reward.py` — new tests.
- `preprocessing/make_fixed_mips.py` — new fixed-protocol MIP generator + coverage manifest.
- `preprocessing/run_fixed_mips.py` — new batch runner + manifest aggregation.

---

## 7. Next steps

**Preprocessing:** close the open items in 5.7 (orientation check, slab_00 QC, image budget, resolution,
class balance), then generate the full dataset with the chosen protocol.

**Data assembly:** prepare structured labels + reference reports; build a **manifest + labels →
training JSONL** bridge (excluding `axial_slab_00`) so images, coverage metadata, and labels are joined
into the format the training scripts expect.

**Modeling:** wire the shared reward into each algorithm (real GRPO reward, image-conditioning for
KTO/GRPO, fix SFT label masking), then run the DPO/KTO/ORPO/GRPO benchmark under the shared
clinically-verifiable reward, with cross-validation and confidence intervals.
