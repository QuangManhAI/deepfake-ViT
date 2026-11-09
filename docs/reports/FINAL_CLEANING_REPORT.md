# Final Dataset Cleaning & Training-Readiness Report

## Summary

The dataset is now **training-ready for the identity-disjoint + exact-duplicate-aware protocol**.

- Original split preserved at `data/splits_original/`
- Clean candidate A (identity-disjoint + duplicate-aware) at `data/splits_identity_clean/`
- Clean candidate B (video-disjoint + duplicate-aware) at `data/splits_video_clean/`
- **Recommended:** Strategy A
- **Validator:** `src/data/validate_clean_split.py` passes for Strategy A
- **Training integration:** `src/training/train.py` loads Strategy A CSVs; it currently fails only because DINOv3 weights are missing.

---

## 1. Exact Duplicate Classification

### Source data

- 438 exact-duplicate groups, 703 extra images
- 247 cross-split exact-duplicate groups, 699 images

### Classification of the 247 cross-split groups

| Classification | Count | Definition |
|----------------|-------|------------|
| **LEAKAGE** | 242 | Same physical image reused with different `identity` and `video` metadata. These are identical pixels appearing in multiple splits. |
| **AMBIGUOUS** | 5 | Mixed or unclear identity/video pattern; treated as leakage for safety. |
| **SAFE_INTENTIONAL** | 0 | No safe, intentional cross-split exact duplicates. The 247 are not legitimate shared real frames. |

### Finding

**242 of 247 cross-split exact duplicates are the same physical image assigned different `identity` and `video` values in the manifest.** This is a data-construction bug, not an intentional DF40 design. The same fake image (e.g., from `starganv2`) has been recorded multiple times, each time with a different `identity` and `video` field, and then split by `identity`, which places identical pixels in different splits.

**Action:** For each duplicate MD5 group, keep one canonical row and remove the other rows. Canonical selection: prefer `real` if present, otherwise the lexicographically first path. This removed 703 rows, leaving 29,988 unique images.

**Traceability:** Every removed row is recorded in:

```text
experiments/results/eda_real_data/removed_exact_duplicates.csv
```

with: `md5`, `kept_path`, `removed_path`, `removed_split`, `kept_split`, `method`, `identity`, `video`.

---

## 2. Near-Duplicate Analysis

### Method

- 8×8 average hash (pure PIL/numpy)
- Hamming distance threshold ≤ 8
- LSH bucketing by 4 quadrants
- Connected-component grouping

### Results (from full 30,691 images)

| Metric | Value |
|--------|-------|
| Near-duplicate groups | 5,340 |
| Cross-split near-duplicate groups | 4,215 |

### Strength classification by `max_hamming` within groups

| Class | Definition | Groups | Cross-split groups |
|-------|------------|--------|--------------------|
| Strong | max_hamming ≤ 5 | 254 | 63 |
| Moderate | 6–8 | 1,539 | 888 |
| Weak | 9–12 | 1,284 | 1,097 |
| Likely false positive / hash collision | > 12 | 2,263 | 2,167 |

### Caveat

The `max_hamming` within a connected component can exceed the pairwise threshold of 8 due to transitive chains. Groups with `max_hamming` > 8 are therefore not guaranteed to be pairwise near-duplicates; many are likely false positives or hash collisions. **Strong near duplicates (≤ 5) are 254 groups, with 63 crossing splits.**

### Recommendation on near duplicates

- Do **NOT** automatically remove the 5,340 groups.
- The 254 strong groups can be reviewed for source-level leakage.
- The 2,263 weak/FP groups should not drive split decisions; they need DINOv3 embeddings or a stricter hash threshold to validate.
- For now, near duplicates are quantified and documented, but the split is based on exact duplicates and identity/video grouping.

---

## 3. Strategy A — Identity-Disjoint + Duplicate-Aware

### Procedure

1. Deduplicate exact duplicates (keep canonical, remove 703 extras).
2. Group remaining 29,988 rows by `identity`.
3. Shuffle identities with seed 42.
4. Assign entire identity groups to train/val/test in a 70/15/15 split by accumulating until target sizes.

### Statistics

| Split | Images | Real | Fake | Fake:Real | Identities | Videos | Methods |
|-------|--------|------|------|-----------|------------|--------|---------|
| train | 20,991 | 826 | 20,165 | 24.41:1 | 15,736 | 7,535 | 41 |
| val | 4,498 | 180 | 4,318 | 23.99:1 | 3,356 | 2,462 | 41 |
| test | 4,499 | 171 | 4,328 | 25.31:1 | 3,511 | 2,591 | 41 |

### Validation

```text
Exact duplicate leakage: 0
Identity overlap: 0
Video overlap: 1,509 / 1,542 / 930 (allowed by design)
Missing files: 0
Duplicate paths: 0
```

**Status:** `PASS`

---

## 4. Strategy B — Video-Disjoint + Duplicate-Aware

### Procedure

1. Same deduplication as Strategy A.
2. Group remaining 29,988 rows by `video`.
3. Shuffle videos with seed 42.
4. Assign entire video groups to train/val/test in a 70/15/15 split by accumulating until target sizes.

### Statistics

| Split | Images | Real | Fake | Fake:Real | Identities | Videos | Methods |
|-------|--------|------|------|-----------|------------|--------|---------|
| train | 20,991 | 816 | 20,175 | 24.72:1 | 16,214 | 6,600 | 41 |
| val | 4,499 | 178 | 4,321 | 24.28:1 | 3,558 | 1,450 | 41 |
| test | 4,498 | 183 | 4,315 | 23.58:1 | 3,655 | 1,409 | 41 |

### Validation

```text
Exact duplicate leakage: 0
Video overlap: 0
Identity overlap: 393 / 403 / 124
Missing files: 0
Duplicate paths: 0
```

**Status:** `PASS` on exact/video, but `FAIL` on identity-disjoint.

---

## 5. Comparison and Recommended Protocol

| Criterion | Strategy A (identity) | Strategy B (video) |
|-----------|----------------------|---------------------|
| Exact-duplicate leakage | 0 ✅ | 0 ✅ |
| Identity leakage | 0 ✅ | 393+ ❌ |
| Video leakage | present | 0 ✅ |
| Class balance | 24.4:1 fake:real | 24.2:1 fake:real |
| Method coverage | 41 methods all splits | 41 methods all splits |
| Split sizes | 20,991 / 4,498 / 4,499 | 20,991 / 4,499 / 4,498 |
| Data lost vs original | 703 images | 703 images |
| Evaluation reliability | Higher: aligns with original project protocol | Lower for identity-disjoint tasks; better for source-disjoint tasks |

### Recommendation: Strategy A

**Rationale:**

- The original `prepare_df40_splits.py` and project documentation are built around **identity-disjoint** splits.
- Strategy A preserves identity-disjointness while removing the exact-duplicate leakage.
- Strategy B prevents video leakage but introduces identity overlap, which breaks the project’s stated protocol.
- Video/source-level leakage is real, but the project’s intended safety mechanism is at the identity level. If source-disjoint evaluation becomes a hard requirement, a dedicated video-disjoint split can be generated later.

**What it prevents:**
- Exact-duplicate leakage across splits.
- Identity-level leakage across splits.

**What it does not prevent:**
- Video/source-level leakage.

**How class/method balance changes:**
- Negligible. Fake:real ratio remains ~24.5:1.
- All 41 methods still present in all splits.

---

## 6. Data Removed / Reassigned

- **703 rows removed** (one canonical kept per MD5 group).
- **29,988 images remain** (from 30,691).
- **No images silently deleted.** Removed paths are recorded in `removed_exact_duplicates.csv`.
- **All removals are traceable by MD5 and original split.**

---

## 7. Final Leakage Validation

Executable validator: `src/data/validate_clean_split.py`

Command used:

```bash
.venv/bin/python src/data/validate_clean_split.py \
    --split-dir data/splits_identity_clean \
    --identity-disjoint
```

Output: `PASS`

### Validator checks

- Exact duplicate overlap across splits = 0
- Identity overlap across splits = 0 (when `--identity-disjoint`)
- Video overlap across splits = 0 (when `--video-disjoint`)
- Missing files = 0
- Duplicate paths within splits = 0
- Class / method / identity / video balance printed

**Fails loudly** if any leakage condition is violated.

---

## 8. Training Integration Status

### DataLoader verification

`src/data/verify_training_dataloaders.py` confirms all splits load `(B, 3, 256, 256)` tensors.

### `train.py` with Strategy A

Command tested:

```bash
.venv/bin/python src/training/train.py \
    --train-csv data/splits_identity_clean/train.csv \
    --val-csv data/splits_identity_clean/val.csv \
    --test-csv data/splits_identity_clean/test.csv \
    --max-train 2 --epochs 1
```

Output:

```text
[14:40:38] [INFO] RunLogger started: dinov3_finetuned
[14:40:39] [INFO] Train: 2 | Val: 4498 | Test: 4499
FileNotFoundError: .../model.safetensors
```

**Interpretation:** `train.py` correctly reads the Strategy A CSVs and counts the samples. It fails only at the DINOv3 model weight loading step.

### To train with the cleaned split

```bash
.venv/bin/python src/training/train.py \
    --train-csv data/splits_identity_clean/train.csv \
    --val-csv data/splits_identity_clean/val.csv \
    --test-csv data/splits_identity_clean/test.csv \
    --model <path-to-dinov3-weights.safetensors> \
    --class-weight \
    --seed 42
```

The original `data/splits/` is preserved; no default paths were overwritten. The user must explicitly pass the cleaned CSVs to `train.py`, or symlink/copy `data/splits_identity_clean/` to `data/splits/` when ready.

---

## 9. DINOv3 Status

- **Status:** BLOCKED
- **Reason:** No `model.safetensors` in `experiments/checkpoints/weights/`
- **Impact:** Cannot run full `train.py` or embedding-based feature similarity
- **Action:** Download from `ManhQuangAI/dinov3-deepfake-detection` when convenient

---

## 10. Remaining Blockers

1. **DINOv3 weights missing** — blocks actual fine-tuning and DINOv3 similarity.
2. **Video/source-level leakage** — still present in Strategy A by design. If this must be eliminated, switch to Strategy B or a hybrid, accepting the impact on identity-disjointness.
3. **Class imbalance** — still ~24:1. Consider `--class-weight` or additional real data.

---

## Files Produced

| File | Purpose |
|------|---------|
| `data/splits_original/` | Immutable original splits |
| `data/splits_identity_clean/` | Strategy A: recommended cleaned split |
| `data/splits_video_clean/` | Strategy B: video-disjoint cleaned split |
| `src/data/build_clean_splits.py` | Builds both strategies |
| `src/data/validate_clean_split.py` | Executable validator |
| `src/data/classify_exact_duplicates.py` | Classifies cross-split exact duplicates |
| `experiments/results/eda_real_data/removed_exact_duplicates.csv` | Traceable removal log |
| `experiments/results/eda_real_data/clean_split_comparison.json` | Numerical comparison |
| `FINAL_CLEANING_REPORT.md` | This report |
