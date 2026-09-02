# Dataset Protocol

## 1. Dataset

| Property | Value |
|----------|-------|
| **Source** | Hugging Face `ManhQuangAI/df40-test-data-v3` |
| **Local root** | `test_data_v3/` |
| **Original manifest rows** | 30,691 |
| **Clean rows** | 29,988 |
| **Real** | 1,177 |
| **Fake** | 28,811 |
| **Identities** | 22,601 |
| **Videos** | ~9,300 |
| **Methods** | 41 |

The raw dataset is **frozen**. No raw images are modified. EDA artifacts are preserved as immutable evidence.

---

## 2. Primary Split Strategy

**Selected protocol:** `identity_clean_v1`

**Location:** `data/protocol/`

**Strategy:** identity-disjoint + exact-duplicate-aware

**Random seed:** 42

**Files:**

- `data/protocol/train.csv`
- `data/protocol/val.csv`
- `data/protocol/test.csv`
- `data/protocol/train_detailed.csv`
- `data/protocol/val_detailed.csv`
- `data/protocol/test_detailed.csv`
- `data/protocol/protocol_metadata.json`
- `data/protocol/protocol_config.json`
- `data/protocol/README.md`

### Split sizes

| Split | Images | Real | Fake | Fake:Real |
|-------|--------|------|------|-----------|
| Train | 20,991 | 826 | 20,165 | 24.41:1 |
| Val   | 4,498  | 180 | 4,318  | 23.99:1 |
| Test  | 4,499  | 171 | 4,328  | 25.31:1 |

### Constraints

- `identity(train) ∩ identity(val) = ∅`
- `identity(train) ∩ identity(test) = ∅`
- `identity(val) ∩ identity(test) = ∅`
- `exact_duplicate(train, val) = 0`
- `exact_duplicate(train, test) = 0`
- `exact_duplicate(val, test) = 0`

---

## 3. Alternative Benchmark: Video-Disjoint Split

**Location:** `data/splits_video_clean/`

**Strategy:** video-disjoint + exact-duplicate-aware

### Split sizes

| Split | Images | Real | Fake |
|-------|--------|------|------|
| Train | 20,991 | 816 | 20,175 |
| Val   | 4,499  | 178 | 4,321  |
| Test  | 4,498  | 183 | 4,315  |

### Constraints

- `video(train) ∩ video(val) = ∅`
- `video(train) ∩ video(test) = ∅`
- `video(val) ∩ video(test) = ∅`
- exact duplicate overlap = 0

**Trade-off:** Identity overlap emerges in the video-disjoint split (393 train/val, 403 train/test, 124 val/test). This breaks the identity-disjoint objective.

---

## 4. Protocol Comparison

| Metric | Identity Clean | Video Clean |
|--------|----------------|-------------|
| Train samples | 20,991 | 20,991 |
| Val samples | 4,498 | 4,499 |
| Test samples | 4,499 | 4,498 |
| Real | 1,177 | 1,177 |
| Fake | 28,811 | 28,811 |
| # identities | 22,601 | 22,601 |
| # videos | ~9,300 | ~9,300 |
| # methods | 41 | 41 |
| Exact duplicate leakage | 0 | 0 |
| Identity overlap | 0 | 920 |
| Video overlap | 3,981 | 0 |
| Near-duplicate overlap groups | 4,215 | 4,215 |
| Weak-method coverage (< 500 samples) | 10 | 10 |

---

## 5. Why Identity-Disjoint Is Primary

1. **Measures identity generalization.** The model must classify fake faces on identities never seen during training, which is the core generalization task for deepfake detection.
2. **Prevents exact-duplicate and identity-level leakage.** All cross-split exact and identity overlaps are zero.
3. **Preserves method coverage and class distribution.** All 41 methods remain in all splits; class imbalance is essentially unchanged.
4. **Matches the project objective and existing tooling.** The original `prepare_df40_splits.py` and the EDA pipeline are built around identity-disjoint splits.
5. **Known limitation is explicit.** Video/source overlap is not treated as safe; it is quantified and documented as a known source-level leakage risk.

**Video-disjoint is not primary** because it introduces identity overlap, which directly contradicts the stated project protocol. It is kept as an alternative benchmark for source-disjoint analysis if needed later.

---

## 6. Leakage Controls

### Exact duplicates

- 438 exact-duplicate groups
- 703 extra images removed/reassigned
- One canonical image kept per group
- Log: `experiments/results/eda_real_data/removed_exact_duplicates.csv`
- Cross-split exact duplicate overlap: **0**

### Identity separation

- Cross-split identity overlap: **0**

### Video separation

- **Primary protocol:** no video separation. Video overlap exists (1,509 train/val, 1,542 train/test, 930 val/test) and is documented as a known limitation.
- **Alternative protocol:** cross-split video overlap = **0**, but identity overlap emerges.

### Near duplicates

- 5,340 near-duplicate groups
- 4,215 cross-split near-duplicate groups
- **Not removed**; quantified and documented for downstream analysis.

---

## 7. Known Limitations

1. **Video/source overlap** in the primary protocol. Background, lighting, compression, and source artifacts may leak across splits.
2. **Near-duplicate overlap.** 4,215 near-duplicate groups cross splits; only exact duplicates were removed.
3. **Class imbalance.** ~24.5:1 fake:real. Class weighting or additional real data is recommended.
4. **Method imbalance.** Several methods have < 500 samples.
5. **Face detection unavailable** in the quality profile.
6. **DINOv3 weights unavailable** — model training is still blocked.

---

## 8. Single Source of Truth

The active protocol is configured in:

- `data/protocol/protocol_config.json`

Example:

```json
{
  "DATA_PROTOCOL": "identity_clean_v1",
  "protocol_dir": "data/protocol",
  "train_csv": "data/protocol/train.csv",
  "val_csv": "data/protocol/val.csv",
  "test_csv": "data/protocol/test.csv"
}
```

Downstream code should load CSVs from this config:

```python
from src.data.get_protocol_paths import get_protocol_csvs
train_csv, val_csv, test_csv = get_protocol_csvs()
```

`src/training/train.py` also falls back to the protocol config if `--train-csv`, `--val-csv`, and `--test-csv` are not provided.

---

## 9. Reproducibility

The protocol is generated by:

- `src/data/build_clean_splits.py`
- `src/data/validate_protocol.py`

Metadata and validation reports:

- `data/protocol/protocol_metadata.json`
- `experiments/results/dataset_protocol/protocol_validation.json`
- `experiments/results/dataset_protocol/protocol_comparison.csv`

All numbers are internally consistent with `manifest.csv`, `sample_quality.csv`, and `FINAL_DATA_EDA_PHASE_REPORT.md`.

---

## 10. Validation Gate

| Check | Status |
|-------|--------|
| Primary protocol reproducible | ✓ `data/protocol/` frozen |
| Train/val/test files exist | ✓ |
| All paths valid | ✓ smoke test |
| No exact duplicate leakage | ✓ |
| Identity constraint satisfied | ✓ |
| Video constraint documented | ✓ (known limitation) |
| Method coverage documented | ✓ 41 methods |
| Class imbalance documented | ✓ ~24.5:1 fake:real |
| Near-duplicate limitation documented | ✓ 4,215 cross-split groups |
| DataLoader smoke test passes | ✓ |
| Training code points to selected protocol | ✓ `train.py` uses `data/protocol/protocol_config.json` |
| No hardcoded external dataset paths | ✓ config-driven |

---

## 11. Final Status

```text
DATASET PROTOCOL PHASE = COMPLETE
```

Ready to proceed to **Phase 3 — Baseline Training** once DINOv3 weights are available.
