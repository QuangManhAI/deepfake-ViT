# Data / EDA Phase — Final Consistency Report

**Status:** `DATA / EDA PHASE = COMPLETE`

---

## 1. Final Dataset Size

| Source | Rows |
|--------|------|
| `manifest.csv` | 30,691 |
| Removed / reassigned exact duplicates | 703 |
| **Clean split total** | **29,988** |

**Consistency check:** `30,691 - 703 = 29,988` ✓

---

## 2. Final Split Sizes

| Split | Images | Real | Fake | Fake:Real |
|-------|--------|------|------|-----------|
| Train | 20,991 | 826 | 20,165 | 24.41:1 |
| Val | 4,498 | 180 | 4,318 | 23.99:1 |
| Test | 4,499 | 171 | 4,328 | 25.31:1 |

All split files load, reference valid paths, and contain no duplicate paths.

---

## 3. Leakage Status

| Leakage Type | Train↔Val | Train↔Test | Val↔Test | Status |
|--------------|-----------|------------|----------|--------|
| **Exact-duplicate overlap** | 0 | 0 | 0 | ✓ Clean |
| **Identity overlap** | 0 | 0 | 0 | ✓ Clean |
| **Video overlap** | 1,509 | 1,542 | 930 | ⚠ Exists (source-level risk) |
| **Near-duplicate overlap groups** | — | — | — | 4,215 cross-split groups |

**Video overlap is not described as safe.** It is a known, quantified source-level leakage risk in the identity-disjoint split.

---

## 4. Exact Duplicate Cleanup

- 438 exact-duplicate groups
- 703 extra images removed/reassigned
- One canonical image kept per group
- Every removed sample is logged: `experiments/results/eda_real_data/removed_exact_duplicates.csv`
- No data was silently deleted.

---

## 5. Near-Duplicate Status

- Total near-duplicate groups (threshold 8): 5,340
- Cross-split near-duplicate groups: 4,215
- Not automatically removed
- Strong (≤ 5), moderate (6–8), weak (9–12), likely false-positive (> 12) quantified in `experiments/results/eda_real_data/near_duplicate_threshold_report.json`

---

## 6. Class / Method / Identity Imbalance

| Metric | Finding |
|--------|---------|
| **Class balance** | 24.5:1 fake:real (1,177 real; 28,811 fake) |
| **Method balance** | Highly skewed; `deepfacelab` (25) and `heygen` (24) are the lowest-volume |
| **Identity balance** | 22,601 identities; 84% have only 1 image; 141 have ≥ 10 images |
| **Video balance** | Top 100 videos cover ~20% of images |

---

## 7. Weak-Data Findings

| Method | Weakness Cause |
|--------|----------------|
| `deepfacelab` (25) | **LOW_DATA_VOLUME** |
| `heygen` (24) | **LOW_DATA_VOLUME** |
| `one_shot_free`, `fomm`, `pirender`, `inswap` | **BOTH** |
| `pixart`, `e4e`, `styleclip` | **LOW_IMAGE_QUALITY** (low sharpness despite 1024×1024) |

Sample-level weak images: 5,127 / 29,988 (17.1%) with blur, darkness, low contrast, or high compression.

---

## 8. Distribution-Shift Conclusion

**No substantial distribution shift was detected for the evaluated metrics.**

- Maximum Jensen-Shannon divergence: 0.0585
- All KS tests p > 0.05 for continuous image-quality metrics
- Does **not** prove the test set is fully representative of all future source variation; only that the measured metrics are similar.

---

## 9. Remaining Known Limitations

1. **DINOv3 weights** unavailable — model training/fine-tuning blocked.
2. **Video/source-level overlap** persists in the identity-disjoint split.
3. **Class imbalance** remains ~24.5:1; class weighting or more real data needed.
4. **Face detection** not run for the full dataset (computationally infeasible).

---

## 10. Internal Consistency Confirmation

| Artifact | Status |
|----------|--------|
| `manifest.csv` rows | 30,691 ✓ |
| `removed_exact_duplicates.csv` rows | 703 ✓ |
| `data/splits_identity_clean/*` total | 29,988 ✓ |
| `experiments/results/data_quality/sample_quality.csv` rows | 29,988 ✓ |
| Exact-duplicate overlap (clean split) | 0 ✓ |
| Identity overlap (clean split) | 0 ✓ |
| `DATA_QUALITY_REPORT.md` numbers | reconciled to CSVs ✓ |
| `src/data/eda_deepfake_dataset.ipynb` | 30 cells, executed, no errors ✓ |

**All checked numbers agree across manifest, cleaned splits, report, CSVs, and notebook.**

---

## Final Status

```text
DATA / EDA PHASE = COMPLETE
```

No model training or DINOv3 analysis was performed in this task.
