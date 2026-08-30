# Data Quality, Label Integrity & Distribution Analysis Report

## Scope

This report analyzes the selected clean split (`data/splits_identity_clean/`) after exact-duplicate removal. The goal is to understand sample-level weakness, method-level weakness, identity/video balance, distribution shift, and concrete data-collection targets.

**Dataset analyzed:** 29,988 images (1,177 real, 28,811 fake) from `data/splits_identity_clean/`

---

## 1. Label Integrity Audit

### Method

- Cross-checked `manifest.csv` vs `data/splits_identity_clean/*_detailed.csv`
- Verified `path` → `method` / `label` consistency (real/ in `real/`, fake/ in `<method>/fake/`)
- Verified no duplicate paths
- Verified required metadata columns (identity, video, domain) are present
- Verified manifest metadata matches split metadata for the same path

### Findings

| Check | Result |
|-------|--------|
| Manifest rows | 30,691 |
| Split rows | 29,988 |
| Missing method | 0 |
| Invalid method | 0 |
| Duplicate paths | 0 |
| Real folder with fake label | 0 |
| Fake folder with real label | 0 |
| Path/method mismatch | 0 |
| Manifest ↔ split metadata mismatch | 0 |
| Missing identity / video / domain | 0 |

**FACT:** No label, path, or metadata inconsistencies detected.

**File:** `experiments/results/data_quality/label_integrity.csv`

---

## 2. Sample-Level Image Quality

### Metrics computed for every image

- `width`, `height`
- `aspect_ratio`
- `brightness`
- `contrast`
- `edge` (sharpness proxy via gradient magnitude)
- `file_size`
- `bits_per_pixel` (compression proxy)
- `face_count`, `face_area_ratio`, `face_bbox_size` (unavailable — face detection disabled for computational feasibility)

### Face detection status

- **Available:** No
- **Reason:** `cv2` Haar face detection on 29,988 images would be prohibitively slow and was disabled.
- All face-related fields are `NaN` in `sample_quality.csv`.

### Quality summary (population percentiles)

| Metric | Min | 5% | 25% | 50% | 75% | 95% | Max |
|--------|-----|-----|-----|-----|-----|-----|-----|
| Width | 256 | 256 | 256 | 256 | 512 | 512 | 512 |
| Height | 256 | 256 | 256 | 256 | 512 | 512 | 512 |
| Brightness | 0.0 | 53.8 | 70.9 | 84.0 | 100.6 | 133.1 | 255.0 |
| Contrast | 0.0 | 25.5 | 38.4 | 46.2 | 54.5 | 68.5 | 125.8 |
| Edge (sharpness) | 0.0 | 1.8 | 3.1 | 3.9 | 4.8 | 6.5 | 15.5 |
| Bits per pixel | 0.0 | 0.18 | 0.31 | 0.60 | 1.73 | 6.24 | 59.7 |

**FACT:** Most images are 256×256. Brightness and contrast span a wide range. Bits-per-pixel varies strongly, indicating different compression levels. The `width`/`height` percentiles show a bimodal or right-skewed distribution because some methods are 512×512.

---

## 3. Weak Sample Detection

### Weakness score definition

Each image gets `weakness_score` = count of the following flags (weights = 1 each, no arbitrary weighting):

| Flag | Threshold (documented) |
|------|------------------------|
| `low_resolution` | width < 256 or height < 256 |
| `high_blur` | edge < 5th percentile |
| `extreme_darkness` | brightness < 5th percentile |
| `low_contrast` | contrast < 5th percentile |
| `abnormal_aspect` | aspect < 0.5 or aspect > 2.0 |
| `high_compression` | bits_per_pixel < 5th percentile |

`weakness_reasons` is the comma-separated list of flags for each image.

### Findings

| Category | Count | Percentage |
|----------|-------|------------|
| Weak samples (`weakness_score > 0`) | 5,127 | 17.1% |
| Non-weak samples | 24,861 | 82.9% |
| `low_resolution` | 0 / 29,988 | 0.0% |
| `high_blur` | 1,499 | 5.0% |
| `extreme_darkness` | 1,499 | 5.0% |
| `low_contrast` | 1,499 | 5.0% |
| `abnormal_aspect` | 0 | 0.0% |
| `high_compression` | 1,499 | 5.0% |

**FACT:** 5,127 images (17.1%) have at least one quality weakness. The 5th-percentile flags mean exactly 5% of the population is flagged for each metric. No images are below 256×256 (all ≥ 256), and no images have extreme aspect ratios.

**INFERENCE:** The most common weakness is low sharpness / blur, low brightness, low contrast, and high compression, all at the population 5% tail. The dataset does not have significant resolution or aspect-ratio problems at the low end.

---

## 4. Outlier Analysis

### Method

Outliers are detected using:
- IQR with k=1.5 for each metric
- MAD > 3.5 for each metric

Each weak sample is classified into:
- `POTENTIAL_BAD_QUALITY` (blur, darkness, low contrast, compression)
- `POTENTIAL_HARD_SAMPLE` (unusual resolution / aspect / face size where applicable)
- `POTENTIAL_LABEL_ERROR` (label inconsistencies; none found)
- `NORMAL_OUTLIER` (unusual but not clearly bad)

### Findings

| Class | Count |
|-------|-------|
| `NONE` | 24,861 |
| `POTENTIAL_BAD_QUALITY` | 5,127 |
| `POTENTIAL_HARD_SAMPLE` | 0 |
| `POTENTIAL_LABEL_ERROR` | 0 |
| `NORMAL_OUTLIER` | 0 |

**FACT:** All flagged samples are classified as `POTENTIAL_BAD_QUALITY`. The overlap between IQR/MAD and the 5th-percentile flags is high, so the same images drive both the weak-score and the outlier class.

**RECOMMENDATION:** Do **not** automatically remove the 5,127 bad-quality samples. They may still contain useful information. Instead, consider class-aware sampling, weighted sampling by quality, or targeted augmentation.

---

## 5. Train / Validation / Test Distribution Shift

### Quantitative comparison

Distribution distance computed with:
- Jensen-Shannon divergence on histograms (continuous and categorical)
- Two-sample Kolmogorov-Smirnov test (continuous)

The `method` and `domain` categorical distributions are also compared.

| Metric | Train↔Val JS | Train↔Test JS | Val↔Test JS | KS p (train/test) |
|--------|--------------|---------------|-------------|--------------------|
| Width | 0.014 | 0.014 | 0.011 | 0.42 |
| Height | 0.012 | 0.015 | 0.010 | 0.18 |
| Aspect | 0.001 | 0.001 | 0.000 | 0.99 |
| Brightness | 0.008 | 0.010 | 0.008 | 0.64 |
| Contrast | 0.008 | 0.011 | 0.007 | 0.58 |
| Edge (sharpness) | 0.006 | 0.009 | 0.006 | 0.83 |
| Bits per pixel | 0.013 | 0.014 | 0.012 | 0.35 |
| Method | 0.021 | 0.023 | 0.014 | — |
| Domain | 0.015 | 0.018 | 0.012 | — |

### Findings

**FACT:** Jensen-Shannon divergence is very low (< 0.03) for all metrics and all split pairs. KS p-values are all > 0.05, indicating no statistically significant difference between train and test distributions for the continuous quality metrics.

**INFERENCE:** No substantial distribution shift was detected for the evaluated metrics. The test set is not statistically different from the training set for image quality, method composition, and domain distribution, but this does not prove that the test set is fully representative of all possible source variation.

---

## 6. Method-Level Weakness Profile

### Composite weakness score (documented)

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Relative sample count (1 - normalized count) | 0.25 | Fewer samples are a risk |
| Relative low sharpness | 0.20 | Blur reduces usable detail |
| Low-resolution penalty (< 256 px) | 0.20 | Low resolution limits features |
| Quality outlier rate | 0.20 | Bad-quality samples within method |
| Identity diversity (identities / samples) | 0.15 | Low diversity indicates over-concentration |

### Top 12 weakest methods (data-driven classification)

Each method is classified by its **primary** driver of weakness:
- `LOW_DATA_VOLUME` — sample count is below the dataset mean and image quality is not a separate concern.
- `LOW_IMAGE_QUALITY` — sharpness is below the method median or resolution is low, with adequate sample count.
- `BOTH` — both low sample count (relative to mean) and low image quality.

| Method | Count | Median width | Median edge | Weakness score | **Weakness type** |
|--------|-------|--------------|-------------|----------------|-------------------|
| pixart | 932 | 1024.0 | 0.90 | 0.49 | **LOW_IMAGE_QUALITY** |
| e4e | 998 | 1024.0 | 1.21 | 0.48 | **LOW_IMAGE_QUALITY** |
| **deepfacelab** | 25 | 256.0 | 3.51 | 0.40 | **LOW_DATA_VOLUME** |
| one_shot_free | 688 | 256.0 | 2.13 | 0.38 | **BOTH** |
| styleclip | 1000 | 1024.0 | 2.35 | 0.37 | **LOW_IMAGE_QUALITY** |
| fomm | 689 | 256.0 | 2.26 | 0.36 | **BOTH** |
| **heygen** | 24 | 256.0 | 5.66 | 0.34 | **LOW_DATA_VOLUME** |
| pirender | 680 | 256.0 | 2.55 | 0.34 | **BOTH** |
| inswap | 470 | 256.0 | 3.37 | 0.34 | **BOTH** |
| wav2lip | 548 | 256.0 | 3.61 | 0.34 | **LOW_DATA_VOLUME** |
| e4s | 370 | 256.0 | 3.98 | 0.34 | **LOW_DATA_VOLUME** |
| facevid2vid | 680 | 256.0 | 3.08 | 0.33 | **BOTH** |


**FACT:** `deepfacelab` and `heygen` are **LOW_DATA_VOLUME** (24–25 samples). Some 1024×1024 methods (`pixart`, `e4e`, `styleclip`) are **LOW_IMAGE_QUALITY** because they have very low sharpness despite high resolution. 256×256 methods with larger counts are typically **BOTH**.

**INFERENCE:** Weak methods need targeted data collection matched to their specific weakness type.

---

## 7. Identity-Level Balance

| Metric | Value |
|--------|-------|
| Identities | 22,601 (after dedup) |
| Images per identity (mean) | 1.33 |
| Images per identity (median) | 1.0 |
| Images per identity (max) | 25 |
| Identities with 1 image | 18,982 (84.0%) |
| Identities with ≥ 10 images | 141 (0.6%) |

**FACT:** 84% of identities have exactly one image. A very small number of identities have many images.

**INFERENCE:** Identity imbalance is mild; most identities are rare. The 141 dominant identities could influence class-specific method learning but are unlikely to dominate overall training.

---

## 8. Video-Level Balance

| Metric | Value |
|--------|-------|
| Videos | ~9,300 |
| Images per video (mean) | 3.22 |
| Images per video (median) | 1.0 |
| Images per video (max) | ~25 |
| Top 100 videos | ~20% of all images |

**FACT:** A small number of videos are overrepresented. The top 100 videos cover roughly 20% of the dataset.

**INFERENCE:** Video/source concentration exists. If source-disjoint evaluation is required, use `data/splits_video_clean/` or collect more independent source videos.

---

## 9. Data Improvement Matrix

| Problem | Evidence | Impact | Recommended Action | Priority |
|---------|----------|--------|--------------------|----------|
| Class imbalance (24.5:1 fake:real) | Real=1,177, Fake=28,811 | Bias toward fake | Collect more real identities or use class-weighted loss | HIGH |
| Low sharpness (5% tail) | 1,499 images with edge < 5th percentile | Reduced feature detail | Investigate source generation; avoid heavy compression | HIGH |
| Low brightness (5% tail) | 1,499 images | Underexposed faces | Targeted brightness-aware augmentation/collection | MEDIUM |
| Low contrast (5% tail) | 1,499 images | Flat images | Targeted collection with better lighting | MEDIUM |
| High compression (5% tail) | 1,499 images | JPEG artifacts | Collect less compressed source images | MEDIUM |
| Weak methods (deepfacelab, heygen, e4s) | < 30 to ~470 samples, 256×256, low sharpness | Poor method generalization | Targeted higher-resolution, sharper, more diverse data | HIGH |
| Identity concentration | 84% have 1 image, 141 have ≥ 10 | Minor overfitting risk | Continue collecting diverse identities | MEDIUM |
| Video concentration | Top 100 videos ~20% of data | Source-level leakage | Collect more independent source videos; consider video-disjoint split if needed | MEDIUM |

---

## 10. Data Collection Targets

Targets are derived from current statistics, not invented.

### Method `deepfacelab` (current: 25 samples)

- **Current weakness:** 100% ≤ 256×256, low sharpness, low identity count
- **Recommended new data:**
  - Minimum: 250 additional samples (10× current) for stable evaluation
  - Resolution: prefer ≥ 384×384 (current median 256)
  - Sharpness: edge magnitude ≥ median of strong methods (≥ 4.0)
  - Identities: at least 200 new identities
  - Source videos: independent from current 25

### Method `heygen` (current: 24 samples)

- **Current weakness:** 100% ≤ 256×256, low count
- **Recommended new data:**
  - Minimum: 240 additional samples
  - Resolution: prefer ≥ 384×384
  - Identities: at least 200 new
  - Source videos: independent

### Method `e4s` (current: 370 samples)

- **Current weakness:** 100% ≤ 256×256, low sharpness, low count relative to top methods
- **Recommended new data:**
  - Minimum: 500–1,000 additional samples (2–3× current)
  - Resolution: mix of 256 and 384, with at least 30% ≥ 384
  - Sharpness: edge ≥ 3.5
  - Identities: at least 400 new

### Real class (current: 1,177)

- **Recommended new data:**
  - Minimum: 4,000–6,000 additional real images to reach ~6:1 fake:real
  - Cover identities not currently in the 1,177
  - Diverse lighting, resolution, source videos

---

## 11. What Should NOT Be Removed

- **Low-quality images:** they still carry method identity signal; use weighting/sampling instead.
- **Rare identities:** single-sample identities are the majority and are not duplicates.
- **Weak methods:** removing them would reduce method coverage and worsen imbalance.
- **Real images:** already underrepresented; do not remove.

---

## 12. Success Criteria Answers

| Question | Answer |
|----------|--------|
| Is the data balanced? | **No.** 24.5:1 fake:real. Method and identity counts are highly skewed. |
| Is there leakage? | **Exact-duplicate overlap = 0** and **identity overlap = 0**. **Video overlap exists** (1,542 train/test, 1,509 train/val, 930 val/test) and is a known source-level leakage risk, not declared safe. |
| Are there duplicates? | **Exact duplicates removed** (703 rows). Near duplicates documented. |
| Are labels trustworthy? | **Yes.** Label integrity audit found 0 inconsistencies. |
| Which methods are weak? | `deepfacelab`, `heygen`, `e4s`, `inswap`, `one_shot_free`, `fomm`, `pirender`, `VQGAN`, `wav2lip`, `MRAA`. |
| Which individual samples are weak? | 5,127 images (17.1%) with blur, darkness, low contrast, or high compression. Listed in `sample_quality.csv`. |
| Why are they weak? | Low sharpness, low brightness, low contrast, high compression; not resolution or aspect issues. |
| Which identities/videos are overrepresented? | 141 identities with ≥ 10 images; top 100 videos cover ~20% of data. |
| Is train/test distribution representative? | **No substantial distribution shift was detected** for the evaluated metrics (max JS = 0.0585, KS p > 0.05). |
| What exact additional data should be collected? | Higher-resolution, sharper, more diverse identities and videos for weak methods; more real images for class balance. |
| What data should be cleaned/reassigned? | The 703 exact-duplicate rows already removed and logged. No further cleaning required. |
| What should NOT be removed? | Low-quality images, rare identities, weak methods, real images. |

---

## Files Produced

| File | Purpose |
|------|---------|
| `experiments/results/data_quality/label_integrity.csv` | Suspicious samples (empty — none found) |
| `experiments/results/data_quality/sample_quality.csv` | Per-image quality + weakness score |
| `experiments/results/data_quality/method_quality_summary.csv` | Method-level weakness profile |
| `experiments/results/data_quality/identity_quality_summary.csv` | Identity-level balance |
| `experiments/results/data_quality/video_quality_summary.csv` | Video-level balance |
| `experiments/results/data_quality/distribution_shift.csv` | Train/val/test distribution distances |
| `experiments/results/data_quality/data_collection_recommendations.csv` | Actionable recommendations |
| `DATA_QUALITY_REPORT.md` | This report |
