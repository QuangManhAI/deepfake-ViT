# Final EDA Report — Real-Data Analysis of `test_data_v3`

## Executive Summary

The EDA has been completed on the actual `test_data_v3` dataset (30,691 images). The `src/data/eda_deepfake_dataset.ipynb` notebook has been rewritten as a real-data EDA, executed successfully, and now contains all required sections with embedded visualizations.

**Most important findings:**

1. **247 exact-duplicate groups cross train/val/test splits.** This is a **serious leakage issue**.
2. **4,215 near-duplicate groups cross splits.** Source-level leakage is present.
3. **Video overlap across splits is large** (1,567 / 1,608 / 912 video IDs). The current identity-disjoint split does not protect against video/source-level leakage.
4. **Class imbalance is severe:** 25.08:1 fake:real.
5. **Weak methods are not just low-count:** `deepfacelab`, `heygen`, `e4s` are also low-resolution / lower-sharpness.
6. **DINOv3 feature similarity is blocked** because the published weights are not present locally.

---

## 1. What Was Done

| Step | Status |
|------|--------|
| Download `test_data_v3.zip` | ✓ done |
| Unzip and validate structure | ✓ done |
| Validate `manifest.csv` | ✓ done |
| Run `prepare_df40_splits.py --seed 42` | ✓ done |
| Validate generated CSVs | ✓ done |
| Full exact-duplicate analysis | ✓ done |
| Full near-duplicate analysis | ✓ done |
| Re-evaluate video/source leakage | ✓ done |
| Image quality analysis (2,000 sample) | ✓ done |
| Multi-dimensional weak data analysis | ✓ done |
| Update `eda_deepfake_dataset.ipynb` | ✓ done |
| Execute complete notebook | ✓ done |
| DINOv3 similarity | ✗ blocked |

---

## 2. Dataset Overview

| Metric | Value | Status |
|--------|-------|--------|
| Total images | 30,691 | FACT |
| Real images | 1,177 | FACT |
| Fake images | 29,514 | FACT |
| Fake:Real ratio | 25.08:1 | FACT |
| Unique identities | 23,237 | FACT |
| Unique videos | 9,554 | FACT |
| Unique methods | 41 (real + 40 fake) | FACT |
| Domains | `oth`, `ffc`, `efs`, `fe`, `cdc` | FACT |

---

## 3. Manifest Validation

- **Columns:** `method`, `video`, `path`, `identity`, `domain`, `label`
- **Missing files:** 0
- **Invalid labels:** 0
- **Empty identities:** 0
- **Duplicate paths in manifest:** 0
- **Manifest rows = actual files:** ✓

**Factual finding:** The manifest is internally consistent.

---

## 4. Split Validation

| Split | Images | Real | Fake | Identities | Videos |
|-------|--------|------|------|------------|--------|
| train | 21,459 | 828 | 20,631 | 16,265 | 7,587 |
| val | 4,586 | 171 | 4,415 | 3,485 | 2,616 |
| test | 4,646 | 178 | 4,468 | 3,487 | 2,597 |

- **Identity overlap:** 0 across all split pairs
- **Path overlap:** 0 across all split pairs
- **Video overlap:** see below

---

## 5. Leakage Re-evaluation

### 5.1 Identity leakage

| Pair | Overlap |
|------|---------|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

**Status:** No identity-level leakage.

### 5.2 Path/file leakage

| Pair | Overlap |
|------|---------|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

**Status:** No file-path-level leakage.

### 5.3 Exact-duplicate leakage

| Metric | Value |
|--------|-------|
| Exact duplicate groups | 438 |
| Extra images | 703 |
| **Cross-split exact duplicate groups** | **247** |

**This is a serious issue.** 247 groups of identical images exist across train/val/test. This violates the assumption that no image appears in more than one split.

### 5.4 Near-duplicate leakage

| Metric | Value |
|--------|-------|
| Near-duplicate groups | 5,340 |
| **Cross-split near-duplicate groups** | **4,215** |

**This is also serious.** Images that are visually similar (perceptual hash Hamming distance ≤ 8) appear across splits.

### 5.5 Video/source leakage

| Pair | Overlapping video IDs |
|------|----------------------|
| train ∩ val | 1,567 |
| train ∩ test | 1,608 |
| val ∩ test | 912 |

**This is expected given the split strategy, but it is a source of potential leakage.**

---

## 6. Leakage Taxonomy

### Identity leakage

- **Definition:** Same identity token appears in multiple splits.
- **Result:** 0 overlap.
- **Status:** SAFE.

### Video/source leakage

- **Definition:** Same `video` ID appears in multiple splits.
- **Result:** Large overlap.
- **Risk:** Shared background, camera, lighting, compression, scene, and source-video artifacts can leak information.
- **Status:** PRESENT — must be explicitly addressed if source-level generalization is the goal.

### Exact-duplicate leakage

- **Definition:** Byte-identical files appear in multiple splits.
- **Result:** 247 groups.
- **Risk:** Direct data leakage; identical training and testing images.
- **Status:** CRITICAL.

### Near-duplicate leakage

- **Definition:** Visually near-identical images appear in multiple splits.
- **Result:** 4,215 groups.
- **Risk:** Near-identical content across train/val/test allows the model to memorize rather than generalize.
- **Status:** CRITICAL.

---

## 7. Image Quality Findings

Sample: 2,000 images from all splits.

| Metric | Mean | Std |
|--------|------|-----|
| Width | 383.5 | — |
| Height | 383.5 | — |
| Aspect ratio | 1.0 | — |
| Brightness | 91.02 | 32.39 |
| Contrast | 47.29 | 12.84 |
| Edge magnitude (blur proxy) | 3.72 | 1.85 |

### Method-level quality (sample)

| Weak method | Avg edge | Avg width | Notes |
|-------------|----------|-----------|-------|
| `deepfacelab` | 2.78 | 256 | low sharpness, low resolution |
| `heygen` | 4.83 | 256 | low resolution |
| `e4s` | 3.97 | 256 | low resolution |
| `inswap` | 3.88 | 256 | low resolution |
| `one_shot_free` | 2.13 | 256 | very low sharpness |

**Factual finding:** Many weak methods are concentrated at 256×256 resolution. Others are higher resolution.

---

## 8. Weak vs Strong Data

### Weakest methods (multi-dimensional score: count + sharpness + resolution)

1. `deepfacelab`
2. `heygen`
3. `e4s`
4. `inswap`
5. `one_shot_free`

### Strongest methods

1. `sd2.1`
2. `mobileswap`
3. `DiT`
4. `SiT`
5. `StyleGAN3`

**Factual finding:** Weakness is not only low sample count; it is also low resolution and/or blur.

---

## 9. Visual Inspection

The updated notebook `src/data/eda_deepfake_dataset.ipynb` contains image grids for:

- Real faces
- Strong fake method (`sd2.1`)
- Weak fake methods (`deepfacelab`, `heygen`)

The visual difference confirms that weak methods are lower-resolution and sometimes blurrier.

---

## 10. Data Cleaning / Augmentation Recommendations

| Problem | Evidence | Impact | Recommended action | Priority |
|---------|----------|--------|--------------------|----------|
| **Cross-split exact duplicates** | 247 groups | Direct leakage | Remove/reassign; run `prepare_df40_splits.py` after deduplicating before re-splitting | **CRITICAL** |
| **Cross-split near duplicates** | 4,215 groups | Source leakage | Investigate; consider image grouping or stricter hashing before splitting | **CRITICAL** |
| **Video/source overlap** | > 900 video IDs per split pair | Background/camera artifact leakage | If strict source-disjoint is required, implement video-disjoint split and measure impact | **HIGH** |
| **Class imbalance** | 25:1 fake:real | Model bias toward fake | Class-weighted loss or targeted real oversampling | **HIGH** |
| **Weak methods** | `deepfacelab`, `heygen`, `e4s` | Poor generalization | Collect higher-resolution, more diverse samples for these methods | **HIGH** |
| **DINOv3 similarity** | Weights absent | Cannot find similar samples | Download `ManhQuangAI/dinov3-deepfake-detection` weights | **MEDIUM** |

---

## 11. Feature Similarity Status

- **Status:** BLOCKED
- **Reason:** DINOv3 pre-trained weights are not in `experiments/checkpoints/weights/`
- **Next step:** Download the published backbones from `ManhQuangAI/dinov3-deepfake-detection`

A placeholder section is in the EDA notebook explaining the block.

---

## 12. Notebook Status

- **Deliverable:** `src/data/eda_deepfake_dataset.ipynb`
- **Sections:** 11
- **Cells:** 24
- **Outputs:** stream + display_data (embedded plots)
- **Errors:** None
- **Execution:** Successful from first to last cell

The notebook now loads `manifest.csv` and the generated `*_detailed.csv` files as the source of truth, not stale JSON metadata.

---

## 13. Factual vs Inference vs Recommendation

### Factual findings

1. 30,691 images; 1,177 real, 29,514 fake.
2. 247 exact-duplicate groups cross splits.
3. 4,215 near-duplicate groups cross splits.
4. 1,567 / 1,608 / 912 video IDs cross train/val, train/test, val/test.
5. `deepfacelab`, `heygen`, `e4s` have the lowest composite weakness score.
6. DataLoader can load all splits with shape `(B, 3, 256, 256)`.

### Inferences

1. Identity-disjoint splitting is working, but it does not prevent video/source-level leakage.
2. The 247 cross-split exact duplicates are likely an error in the split generation process (they should not exist if the original data were deduplicated before splitting).
3. The 4,215 cross-split near duplicates are partly expected because the same real face is reused across many fake methods, but they still represent a leakage risk.
4. Weak methods need higher-resolution data, not just more samples.

### Recommendations

1. Remove/reassign the 247 cross-split exact duplicates.
2. Decide whether to implement video-disjoint splitting, and if so, re-run `prepare_df40_splits.py` with video-group shuffling.
3. Use class-weighted loss for the 25:1 imbalance.
4. Collect higher-resolution, more diverse data for `deepfacelab`, `heygen`, `e4s`.
5. Download DINOv3 weights to enable embedding-based weak-data similarity.

---

## 14. Remaining Blockers

1. **Cross-split exact duplicates must be fixed before training.**
2. **DINOv3 weights are needed for feature similarity.**
3. **Decision needed on video-disjoint vs identity-disjoint splitting.**

---

## 15. Key Files

| File | Purpose |
|------|---------|
| `src/data/eda_deepfake_dataset.ipynb` | Real-data EDA notebook (deliverable) |
| `src/data/duplicate_analysis_full.py` | Full exact/near-duplicate analysis |
| `src/data/eda_real_data.py` | Image quality, weak data, leakage analysis |
| `src/data/validate_splits.py` | Split CSV validation |
| `src/data/validate_test_data_v3.py` | Manifest validation |
| `experiments/results/eda_real_data/` | All CSV/JSON artifacts |
| `FINAL_DATASET_VALIDATION_REPORT.md` | Earlier full-pipeline report |

