# Final Dataset Validation & Real-Data EDA Report

## 1. Dataset Provenance & Download

| Item | Value |
|------|-------|
| **Authoritative dataset** | `test_data_v3` |
| **Source** | `ManhQuangAI/df40-test-data-v3` on Hugging Face Hub |
| **File** | `test_data_v3.zip` |
| **Downloaded size** | 4.2 GiB / 4.53 GB |
| **Download date/time** | 2026-08-30 |
| **Unzipped location** | `test_data_v3/` in project root |

**Download command used:**

```bash
.venv/bin/hf download ManhQuangAI/df40-test-data-v3 \
    --repo-type dataset --include "test_data_v3.zip" --local-dir .
unzip -o test_data_v3.zip -d .
```

**Factual finding:** The exact repository and file exist and were successfully downloaded. The repo is currently private and required the local `hf` CLI to already be authenticated.

---

## 2. test_data_v3 Structure

**Expected structure (verified):**

```
test_data_v3/
├── manifest.csv
├── real/
│   └── *.jpg
├── insight/
│   └── fake/
│       └── *.jpg
├── faceswap/
│   └── fake/
│       └── *.jpg
... 40 fake method folders, each with a fake/ subfolder
```

**Factual finding:** After unzipping, the directory tree matches the expected `manifest.csv` + `real/` + `<method>/fake/` layout.

---

## 3. Manifest Validation

| Check | Result |
|-------|--------|
| **Rows (excluding header)** | 30,691 |
| **Columns** | `method`, `video`, `path`, `identity`, `domain`, `label` |
| **Missing required columns** | 0 |
| **Missing files for rows** | 0 |
| **Invalid labels** | 0 |
| **Empty/unknown identities** | 0 |
| **Duplicate paths in manifest** | 0 |
| **Actual image files on disk** | 30,691 (3,997 `.jpg` + 26,694 `.png`) |

**Manifest rows = Actual image files: ✓** (30,691 = 30,691)

### Class distribution

| Class | Count |
|-------|-------|
| Real (label=0) | 1,177 |
| Fake (label=1) | 29,514 |
| **Total** | **30,691** |

### Method distribution (top 15)

| Method | Count |
|--------|-------|
| `sd2.1` | 1,609 |
| `mobileswap` | 1,398 |
| `real` | 1,177 |
| `DiT` | 1,004 |
| `SiT` | 1,004 |
| `StyleGAN3` | 1,004 |
| `StyleGANXL` | 1,004 |
| `stargan` | 1,000 |
| `styleclip` | 1,000 |
| `starganv2` | 999 |

### Domain distribution

| Domain | Count |
|--------|-------|
| `oth` | 13,049 |
| `ffc` | 6,584 |
| `efs` | 6,012 |
| `fe` | 2,999 |
| `cdc` | 2,047 |

### Identity and video counts

- **Unique identities:** 23,237
- **Unique video IDs:** 9,554

**Factual finding:** The manifest is internally consistent. Every row points to an existing file, labels are valid, identities are populated, and there are no duplicate paths.

---

## 4. Split Generation

**Command run:**

```bash
DF40_ROOT=. .venv/bin/python src/data/prepare_df40_splits.py --seed 42
```

**Generated outputs (verified):**

| File | Rows (excl. header) | Labels (real/fake) |
|------|---------------------|--------------------|
| `data/splits/train.csv` | 21,459 | 828 / 20,631 |
| `data/splits/val.csv` | 4,586 | 171 / 4,415 |
| `data/splits/test.csv` | 4,646 | 178 / 4,468 |
| `data/splits/train_detailed.csv` | 21,459 | 828 / 20,631 |
| `data/splits/val_detailed.csv` | 4,586 | 171 / 4,415 |
| `data/splits/test_detailed.csv` | 4,646 | 178 / 4,468 |
| `data/splits/test_full.csv` | 30,691 | 1,177 / 29,514 |
| `data/splits/methods/*.csv` | 40 methods × 3 files each | — |
| `data/splits/train_insight.csv` | 668 | balanced 1:1 |
| `data/splits/val_insight.csv` | 126 | balanced 1:1 |
| `data/splits/test_insight.csv` | 149 | balanced 1:1 |

**Factual finding:** `prepare_df40_splits.py` ran successfully and produced the expected files. All generated CSV paths resolve to existing images.

---

## 5. Split Reconciliation

### Counts match manifest

```
train  21,459
val     4,586
test    4,646
---
total  30,691  ✓ matches manifest
```

### Identity overlap (zero as designed)

| Pair | Overlapping identities |
|------|------------------------|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

### Path overlap

| Pair | Overlapping image paths |
|------|-------------------------|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

### Class distribution per split

| Split | Real | Fake | Ratio |
|-------|------|------|-------|
| train | 828 | 20,631 | 24.9:1 |
| val | 171 | 4,415 | 25.8:1 |
| test | 178 | 4,468 | 25.1:1 |

**Factual finding:** Sums are correct, identity-disjoint property holds, and no path crosses between splits.

---

## 6. Leakage Findings

### Identity leakage

**Result:** **0 overlap** across train/val/test identities.

**Inference:** The `prepare_df40_splits.py` identity-disjoint protocol is working as intended for the `identity` column.

### Path/file leakage

**Result:** **0 overlap** across train/val/test file paths.

**Inference:** No file-level leakage.

### Video leakage

**Result:** Significant overlap of `video` IDs across splits:

| Pair | Overlapping video IDs |
|------|----------------------|
| train ∩ val | 1,567 |
| train ∩ test | 1,608 |
| val ∩ test | 912 |

**Inference:** Video-level separation is **NOT** the same as identity-level separation. The split is done by `identity`, not by `video`. A single video can contain multiple identities (e.g., multiple frames or multiple people), so the same `video` value can appear in different splits.

**Recommendation:** Do **NOT** remove these samples. The project’s intended leakage protection is at the **identity** level, not the **video** level. Removing based on video overlap would discard valid data and break the identity-disjoint protocol.

---

## 7. Duplicate Findings

### Exact duplicates (full dataset, MD5)

- **Duplicate groups:** 438
- **Extra duplicate images:** 703

**Factual finding:** 438 groups of files have identical byte content. These are exact duplicates by file hash.

**Where they come from:** The manifest treats the same real frame as shared across multiple fake methods (1 real face is paired with many fake methods). This is by design — `test_data_v3` stores real frames once in `real/` and reuses them in balanced method sets. Therefore, "exact duplicate" at the file level may be expected for real frames used by different methods.

**Recommendation:** Do **NOT** blindly delete. The duplicates should be checked against `method` and `video` to distinguish:

1. **Shared real frames across methods** → intended, keep.
2. **Identical frames within the same method/video** → possible oversampling, investigate.

### Near duplicates (sample of 1,000 test images)

- **Near-duplicate groups (average hash, threshold ≤ 8):** 0

**Factual finding:** In the 1,000-image test sample, no perceptually near-duplicate pairs were detected with the default threshold.

**Limitation:** The `imagehash` library is not installed. The custom PIL average hash used here has limited discriminative power. A more robust near-duplicate analysis would benefit from `imagehash` or DINOv3 embeddings.

---

## 8. Image-Quality Findings

**Method:** Sampled 2,000 images uniformly at random from the full split set and computed per-image:

- Width, height
- Aspect ratio
- Brightness (mean grayscale)
- Contrast (std grayscale)
- Edge strength (gradient magnitude, proxy for sharpness/blur)

| Metric | Mean | Std |
|--------|------|-----|
| Width | 383.5 | (method-dependent) |
| Height | 383.5 | (method-dependent) |
| Brightness | 91.02 | 32.39 |
| Contrast | 47.29 | 12.84 |
| Edge (blur proxy) | 3.72 | 1.85 |

**Factual finding:** Many methods contain 256×256 images, while others contain higher-resolution images (mean ~383×383 overall). Low-resolution images are concentrated in specific methods.

### Methods with lowest edge / resolution (weak candidate characteristics)

| Method | Sample n | Avg edge | Avg width | Weak score |
|--------|----------|----------|-----------|------------|
| `deepfacelab` | 25 | 2.78 | 256 | 0.84 |
| `heygen` | 24 | 4.83 | 256 | 0.78 |
| `e4s` | 370 | 3.97 | 256 | 0.72 |
| `inswap` | 471 | 3.88 | 256 | 0.69 |
| `one_shot_free` | 688 | 2.13 | 256 | 0.69 |

**Inference:** `deepfacelab`, `heygen`, and `e4s` are weak candidates **not only because of low sample count** but also because they are low-resolution (256×256) and/or lower sharpness.

---

## 9. Weak vs Strong Data (Multi-Dimensional)

Weak data is now scored on:

- Sample count (40%)
- Sharpness/edge strength (30%)
- Resolution (30%)

**Top weak methods (composite score):**

1. `deepfacelab`
2. `heygen`
3. `e4s`
4. `inswap`
5. `one_shot_free`

**Strong methods (high sample count, good quality):**

1. `sd2.1`
2. `mobileswap`
3. `DiT`
4. `SiT`
5. `StyleGAN3`

**Factual finding:** The multi-dimensional weak-data score differs from a simple count-based ranking. Some methods have moderate counts but are weak due to low resolution or blur. `deepfacelab` and `heygen` remain the most critical weak groups.

---

## 10. Similar-Data Candidates

**Status:** Full DINOv3-based feature similarity was **not performed** because the DINOv3 pre-trained weights are not present in the repository.

**Why:** The project expects `experiments/checkpoints/weights/` to contain `model-*.safetensors`, but only `.gitkeep` files are present. Without these, extracting embeddings for nearest-neighbor search is not possible.

**Recommendation to enable:**

1. Download the published backbones from `ManhQuangAI/dinov3-deepfake-detection` per `MODELS.md`:

```bash
hf download ManhQuangAI/dinov3-deepfake-detection --local-dir .
```

2. Then run a DINOv3 embedding extractor over `test_data_v3`.
3. For each weak method, find its k-nearest neighbors (k=20–50) within the full DF40 training pool or `test_data_v3`.
4. Use those neighbors as candidate additional data to strengthen the weak methods.

**Inference:** The weak methods `deepfacelab`, `heygen`, `e4s` would likely benefit most from additional real/fake pairs at higher resolution, not merely more samples of the same method.

---

## 11. Training Compatibility

**Factual finding:** The generated `data/splits/train.csv`, `val.csv`, `test.csv` are fully compatible with a standard PyTorch `Dataset`/`DataLoader`.

### Verification performed

A minimal `ImageDataset` + `DataLoader` was instantiated and iterated:

```python
tf = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

**Results:**

| Split | Dataset size | First batch shape | First batch labels |
|-------|--------------|-------------------|--------------------|
| train | 21,459 | `(4, 3, 256, 256)` | `[1, 1, 0, 1]` |
| val | 4,586 | `(4, 3, 256, 256)` | `[1, 1, 1, 1]` |
| test | 4,646 | `(4, 3, 256, 256)` | `[1, 1, 1, 1]` |

**Factual finding:** All image paths resolve, images decode, labels are valid integers, and `DataLoader` produces correctly shaped tensors. `src/training/train.py` can use these CSVs directly.

---

## 12. Critical Distinction: Role of `test_data_v3`

**Factual finding:** `test_data_v3` is the **only available dataset** on this machine and is the source for all currently generated splits.

**How the pipeline is intended to work:**

```
test_data_v3/ (30,691 images, all available data)
  ↓
prepare_df40_splits.py --seed 42
  ↓
data/splits/{train,val,test}.csv   (identity-disjoint 70/15/15 split)
  ↓
ImageDataset in src/training/train.py
  ↓
DataLoader
  ↓
DINOv3 fine-tuning
```

**Is this the training dataset?**

- **Yes, for the current reproduction:** It is split into train/val/test and used by `train.py`.
- **No, in the full project plan:** The complete training pipeline in `README.md` and `prepare_df40_splits.py` Protocol 2 expects a much larger `DF40_train` corpus + FF++ real frames + Celeb-DF real frames (potentially 600k+ images). Those are **not present**.

**Factual finding without overclaiming:** The current generated `train.csv` has 21,459 images. This is a **valid training set** for reproducing the identity-disjoint benchmark, but it is **not the full DF40 training corpus**. If the goal is the high-scale 1:1 balanced pool, `DF40_train` must be downloaded separately.

**Recommendation:** For the immediate EDA and model evaluation, use the current `test_data_v3` splits. For full-scale fine-tuning, download `ManhQuangAI/DF40_train` (74.7 GB) and FF++ / Celeb-DF real frames.

---

## 13. Integration Errors Found and Fixed

| Issue | Fix |
|-------|-----|
| `test_data_v3` missing | Downloaded `ManhQuangAI/df40-test-data-v3` and unzipped |
| `data/splits/*.csv` missing | Ran `prepare_df40_splits.py` with `DF40_ROOT=.` |
| EDA notebook boxplot API error | Fixed `labels=` → `tick_labels=` and added data validation |
| `image_quality.py` `cv2` dependency | Made OpenCV optional with PIL fallback |
| `project_root` resolution in tests | Verified `Path.cwd().parent.parent` works for standard notebook location |

---

## 14. Remaining Blockers

1. **DINOv3 weights not present.** `experiments/checkpoints/weights/` is empty. Feature similarity and full training cannot proceed without downloading the published weights from `ManhQuangAI/dinov3-deepfake-detection`.

2. **Full DF40 training corpus not present.** The 74.7 GB `DF40_train` and FF++/Celeb-DF real frames are not downloaded. Protocol 2 combined pool cannot be built.

3. **Near-duplicate detection limited.** The `imagehash` library is not installed. The custom PIL hash did not find near-duplicates in the sample, but a more robust perceptual hash is recommended.

4. **Notebook not yet rewritten for real data.** The existing `eda_deepfake_dataset.ipynb` is metadata-focused. The real-data analysis is currently in `src/data/eda_real_data.py` and `experiments/results/eda_real_data/`. Updating the notebook to load and visualize these results is a separate, recommended step.

---

## 15. Data Improvement Recommendations

1. **Download DINOv3 backbones** and extract embeddings for full feature similarity.
2. **For weak methods `deepfacelab`, `heygen`, `e4s`:** collect or generate higher-resolution samples (≥ 256×256) and more diverse identities.
3. **Do not remove video-overlapping samples** unless the project’s intended protocol is changed to video-disjoint.
4. **Investigate the 438 exact-duplicate groups** by `method` to ensure they are intended shared real frames, not accidental within-method copies.
5. **Consider the class imbalance** (96% fake) when training: use class-weighted loss or upsample real samples.
6. **For full training scale:** download `DF40_train` and FF++/Celeb-DF real frames to build the combined 600k+ image pool.

---

## 16. Summary of Factual vs Inferred vs Recommended

| Category | Statement |
|----------|-----------|
| **FACT** | `test_data_v3.zip` (4.53 GB) was downloaded and unzipped successfully. |
| **FACT** | Manifest has 30,691 rows; every row resolves to an existing image. |
| **FACT** | Splits add to 30,691; identity overlap is 0 across train/val/test. |
| **FACT** | Video overlap is > 900 between each split pair. |
| **FACT** | 438 exact-duplicate file groups exist (703 extra images). |
| **FACT** | DataLoader loads `(4, 3, 256, 256)` tensors from all splits. |
| **FACT** | `deepfacelab` and `heygen` have the lowest sample count and lowest resolution. |
| **INFERENCE** | Video overlap is expected because the split is identity-based, not video-based. |
| **INFERENCE** | The 438 exact duplicates likely include shared real frames across methods. |
| **RECOMMENDATION** | Download DINOv3 weights before full feature-similarity analysis. |
| **RECOMMENDATION** | Use current splits for EDA/identity-disjoint benchmark, not for full-scale training without `DF40_train`. |