# Dataset & Pipeline Reconciliation Report

## Executive Determination

**Option C: The repository contains multiple pipelines that must be separated.**

There is **no single authoritative dataset** currently present. The repo contains artifacts and scripts for at least three distinct pipelines:

1. **DF40 test_data_v3 benchmark pipeline** (`prepare_df40_splits.py`) — intended for the main DINOv3 ViT training
2. **v5_weakfix experiment pipeline** (ConvNeXt weak-method repair) — 121,884 training samples
3. **Legacy `data/hug` pipeline** (`split_dataset.py`) — old DeepFakeFace structure

The EDA must target exactly one of these. The current `split_info.json` belongs to the DF40 pipeline but is **stale metadata** for CSV files that do not exist.

---

## 1. Identified Datasets / Pipelines

### Pipeline A: DF40 test_data_v3 (Main Pipeline)

| Attribute | Value |
|-----------|-------|
| **Generator script** | `src/data/prepare_df40_splits.py` |
| **Source data** | `test_data_v3/` (and `DF40_train_manifest.csv` for Protocol 2) |
| **Default root** | `data/raw` (or `$DF40_ROOT`) |
| **Outputs location** | `data/splits/` and `data/processed/` |
| **Metadata file** | `data/splits/split_info.json` |
| **Method summary** | `data/splits/methods_summary.json` |
| **Sample count** | 30,691 (from `split_info.json`) |
| **Split mechanism** | Identity-disjoint on `test_data_v3` (seed 42) |
| **Primary use** | `src/training/train.py` default args |

**Evidence:**
- `prepare_df40_splits.py` docstring explicitly says it generates `data/splits/split_info.json` and `data/splits/methods_summary.json`.
- `split_info.json` `dataset_name` is `"DF40 Deepfake Benchmark"`.
- `split_info.json` contains `identity_disjoint_splits` with 30,691 samples.
- `train.py` default command in docstring uses `data/splits/train_insight.csv`, which `prepare_df40_splits.py` is documented to produce.
- Git commit `61ae0ca` added `prepare_df40_splits.py`, `split_info.json`, and `methods_summary.json` together.

**Current state:**
- `data/splits/*.csv` files are **missing**.
- `test_data_v3/` is **missing**.
- `data/raw/DF40` is **missing**.
- Therefore `split_info.json` is **stale metadata** for non-existent CSVs.

---

### Pipeline B: v5_weakfix (ConvNeXt Experiment)

| Attribute | Value |
|-----------|-------|
| **Generator script** | `scripts/build_finetune_v5_weakfix.py` (referenced but **NOT PRESENT** in repo) |
| **Source data** | `DF40_train_extracted` (absolute paths to `/workspace/data/`) |
| **Train CSV** | `data/splits/train_v5_weakfix.csv` (121,884 rows) |
| **Train v3 CSV** | `data/splits/train_v5_weakfix_v3.csv` (129,884 rows) |
| **Metadata** | `data/splits/v5_weakfix_dataset_summary.json` (v2), `v5_weakfix_v3_dataset_summary.json` (v3) |
| **Sample count** | 121,884 (v2) / 129,884 (v3) |
| **Split mechanism** | Identity-disjoint, dedup vs v5, frame/video tokens |
| **Primary use** | `scripts/finetune_convnext_weakfix.py` |

**Evidence:**
- `v5_weakfix_dataset_summary.json` `out_csv` is `/workspace/hoangtuan/deepfake-ViT/data/splits/train_v5_weakfix.csv`.
- `finetune_convnext_weakfix.py` uses absolute `PROJECT_ROOT = Path("/workspace/quangmanh/deepfake")` and hardcoded CSVs.
- `doc/v5_weakfix/` and `doc/convnext_weakfix/` describe the v2/v3 CSVs.
- Git commit `f2fb8fd` added `v5_weakfix_dataset_summary.json` and `finetune_convnext_weakfix.py` together.

**Current state:**
- `data/splits/train_v5_weakfix.csv` is **missing**.
- `data/splits/train_v5_weakfix_v3.csv` is **missing**.
- The generator `build_finetune_v5_weakfix.py` is **missing** from `scripts/`.
- CSVs reference absolute `/workspace/` paths from another machine.
- This pipeline is an **external/replicated experiment**, not the main training pipeline.

---

### Pipeline C: Legacy `data/hug` (DeepFakeFace)

| Attribute | Value |
|-----------|-------|
| **Generator script** | `src/data/split_dataset.py` |
| **Source data** | `data/hug` with `wiki/insight/inpainting/text2img` subfolders |
| **Outputs** | `data/splits/train.csv`, `val.csv`, `test.csv` |
| **Split mechanism** | 100 identities (folders 00-99) → 72/8/20 train/val/test |
| **Sample count** | Unknown; would need `data/hug` |
| **Primary use** | None currently (orphan script) |

**Evidence:**
- `split_dataset.py` docstring describes `data/hug` and category mapping `wiki=real`, `insight/inpainting/text2img=fake`.
- It writes a different `split_info.json` with only `seed` and `splits` keys, **not** the current `split_info.json`.
- `data/hug` does **not exist**.

**Current state:**
- Legacy/orphan. Not connected to DF40 or v5_weakfix.
- Should not be used for the current project unless intentionally reverting to DeepFakeFace.

---

## 2. Training Pipeline Trace

### Entry point
`src/training/train.py`

### CSV expected
```python
--train-csv data/splits/train_insight.csv
--val-csv data/splits/val_insight.csv
--test-csv data/splits/test_insight.csv
```

### Dataset class
`ImageDataset` inside `train.py` reads `path,label` columns from CSV.

### DataLoader
`DataLoader(train_ds, ...)` with `num_workers` and `pin_memory`.

### Transform
- `TRAIN_TF`: Resize, RandomHorizontalFlip, ToTensor, Normalize
- `EVAL_TF`: Resize, ToTensor, Normalize

### CSV source
`train_insight.csv` is intended to be generated by `prepare_df40_splits.py` (Protocol 3 method convenience split). It does **not** exist currently.

### Conclusion
The training pipeline is **designed for DF40** but cannot run because the CSVs from `prepare_df40_splits.py` are missing.

---

## 3. DF40 Pipeline Trace

```
test_data_v3/
  manifest.csv
  real/
  <method>/
    
  ↓

src/data/prepare_df40_splits.py
  --seed 42
  
  ↓

  data/splits/split_info.json
  data/splits/methods_summary.json
  data/splits/train.csv, val.csv, test.csv
  data/splits/train_balanced.csv, ...
  data/splits/train_insight.csv, train_faceswap.csv, ...
  data/splits/test_full.csv, test_full_detailed.csv
  data/splits/methods/*.csv
  
  ↓

  ImageDataset (src/training/train.py)
  
  ↓

  DataLoader
  
  ↓

  DinoViTClassifier
```

**Current blockage:** `test_data_v3/` is not present, so `prepare_df40_splits.py` cannot be run.

---

## 4. Investigation: `split_info.json`

| Question | Answer |
|----------|--------|
| Who generated it? | `src/data/prepare_df40_splits.py` |
| Which commit? | `61ae0ca` ("fix(audit): resolve all codebase audit findings from 08d20d7 re-audit") |
| When? | 2026-08-22 04:34:49 UTC |
| For what pipeline? | DF40 `test_data_v3` identity-disjoint benchmark |
| Dataset it describes | `DF40 Deepfake Benchmark` (30,691 samples) |
| Why 37 split CSVs? | `prepare_df40_splits.py` generates them; they were not committed |
| Why are CSVs missing? | CSVs are data outputs, likely gitignored or too large for git |
| Why 30,691 samples? | This is the size of `test_data_v3` after identity-disjoint split |
| Is it valid? | It accurately describes the *intended* output of `prepare_df40_splits.py` but is stale without the CSVs |
| Is it stale? | **Yes** — metadata for files that no longer exist locally |

---

## 5. Investigation: `v5_weakfix_dataset_summary.json`

| Question | Answer |
|----------|--------|
| Who generated it? | `scripts/build_finetune_v5_weakfix.py` (not present in this repo) |
| Which commit? | `f2fb8fd` ("Add evaluation and finetuning scripts for ConvNeXt weakfix models") |
| When? | 2026-08-24 04:23:52 UTC |
| What dataset? | `train_v5_weakfix.csv` (121,884 samples) for v5_weakfix v2 |
| What does v5 mean? | "v5" refers to the 5th version/experiment of a weak-method fix; documented in `doc/v5_weakfix/` |
| Why 121,884? | `v5_weakfix_dataset_summary.json` confirms 121,884 rows → `train_v5_weakfix.csv` |
| Which scripts consume it? | `scripts/finetune_convnext_weakfix.py` |
| Is it related to DF40? | Yes, it uses DF40 training data, but it is a **separate experiment**, not the main pipeline |
| Is it related to current training? | No — `train.py` expects `train_insight.csv` (DF40 pipeline) |

---

## 6. Authoritative Dataset Determination

### Verdict: Option C — Multiple pipelines coexist

There is **no single authoritative dataset** in the current repository because:

1. The **main DF40 pipeline** has metadata (`split_info.json`, `methods_summary.json`) but no CSVs and no `test_data_v3`.
2. The **v5_weakfix pipeline** has metadata (`v5_weakfix_dataset_summary.json`) but no CSVs and no generator script.
3. The **legacy `data/hug` pipeline** has a generator script but no data.

### Recommended authoritative pipeline

For the main DINOv3 ViT project described in `README.md`, the authoritative pipeline is:

**DF40 test_data_v3 → `prepare_df40_splits.py` → `data/splits/*.csv` → `src/training/train.py`**

All other pipelines (v5_weakfix, data/hug) are either experiments or legacy and should not be mixed into the EDA.

---

## 7. Cross-Pipeline Contamination Detected

| Contamination | Evidence |
|---------------|----------|
| EDA analyzed `split_info.json` as if it were the active dataset | EDA used 30,691 samples from DF40, but no CSVs exist |
| Training docs recommend `split_dataset.py` | README lists it, but it is for `data/hug`, not DF40 |
| `v5_weakfix_dataset_summary.json` uses absolute `/workspace/` paths | Refers to another machine (`hoangtuan`, `quangmanh`) |
| `finetune_convnext_weakfix.py` uses hardcoded `PROJECT_ROOT` | `/workspace/quangmanh/deepfake` — will not work on this machine |
| `v5_weakfix` CSVs expected by scripts but missing | `train_v5_weakfix.csv` referenced but absent |
| `split_info.json` references 242 total CSVs that are absent | 37 split files + 205 method files missing from `data/splits/` |

**Critical rule for EDA:** The EDA must analyze the **same** dataset used by `train.py`.

---

## 8. Git History Findings

Relevant commits:

| Commit | Date | Significance |
|--------|------|--------------|
| `61ae0ca` | 2026-08-22 04:34 | Added `prepare_df40_splits.py`, `split_info.json`, `methods_summary.json` — main DF40 pipeline |
| `86f0be6` | 2026-08-22 14:44 | Updated `split_info.json` and `methods_summary.json`; added `train.py` freeze-backbone, error analysis |
| `f2fb8fd` | 2026-08-24 04:23 | Added `v5_weakfix_dataset_summary.json` and `finetune_convnext_weakfix.py` — separate ConvNeXt experiment |

No evidence that CSV files were ever committed. They were generated locally and not tracked.

---

## 9. Pipeline Map

```
                    PROJECT
                       │
          ┌────────────┼────────────┐
          │            │            │
       DF40        v5_weakfix    data/hug
          │            │            │
    test_data_v3   DF40_train     wiki/insight/...
          │            │            │
    prepare_df40_   build_finetune_  split_dataset.py
        splits.py    v5_weakfix.py    │
          │            │            │
    data/splits/   data/splits/   data/splits/
    split_info.json  v5_*_summary.json  split_info.json
          │            │            │
    train_insight.   train_v5_      train.csv
         csv         weakfix.csv    │
          │            │            │
    src/training/    scripts/       (unused)
       train.py    finetune_convnext_  │
                     weakfix.py    │
```

---

## 10. File Classification

| File | Pipeline | Status | Reason |
|------|----------|--------|--------|
| `src/data/prepare_df40_splits.py` | DF40 | **KEEP** | Main, authoritative split generator |
| `src/data/split_dataset.py` | data/hug | **ARCHIVE / LEGACY** | Orphaned, incompatible with DF40 |
| `src/data/build_df40_balanced.py` | DF40 | **KEEP** | DF40 subset builder |
| `src/data/build_test_data.py` | DF40 | **KEEP** | DF40 test set builder |
| `scripts/finetune_convnext_weakfix.py` | v5_weakfix | **KEEP but fix paths** | Valid experiment but hardcoded workspace paths |
| `data/splits/split_info.json` | DF40 | **STALE / KEEP** | Valid metadata for missing CSVs |
| `data/splits/methods_summary.json` | DF40 | **STALE / KEEP** | Valid metadata for missing CSVs |
| `data/splits/v5_weakfix_dataset_summary.json` | v5_weakfix | **KEEP** | Experiment metadata only |
| `data/splits/v5_weakfix_v3_dataset_summary.json` | v5_weakfix | **KEEP** | Experiment metadata only |
| `notebooks/courseWorkCheck.ipynb` | v5_weakfix / external | **ARCHIVE / UNKNOWN** | References absolute external paths |

**Do not delete any files.** They all have diagnostic value.

---

## 11. Safe Integration Fixes Applied

1. **None requiring dataset changes.**
2. **Recommendation only:** update `finetune_convnext_weakfix.py` `PROJECT_ROOT` to be repo-relative if that experiment is to be run.
3. **No labels, splits, or train/val/test assignments were modified.**

---

## Final Report

### 1. Authoritative Dataset

For the main project, the authoritative dataset is **DF40 test_data_v3**, processed by `src/data/prepare_df40_splits.py`.

### 2. Evidence

- `README.md` and `prepare_df40_splits.py` both describe DF40.
- `train.py` default CSVs (`train_insight.csv`, etc.) are produced by `prepare_df40_splits.py`.
- `split_info.json` and `methods_summary.json` match the `prepare_df40_splits.py` output format.
- Git commit `61ae0ca` added all three together.

### 3. Pipeline Map

```
DF40 test_data_v3
      ↓
src/data/prepare_df40_splits.py
      ↓
data/splits/*.csv (including train_insight.csv)
      ↓
ImageDataset (src/training/train.py)
      ↓
DataLoader
      ↓
DinoViTClassifier
```

### 4. Conflicting Artifacts

- `split_info.json` = 30,691 samples (DF40 test_data_v3)
- `v5_weakfix_dataset_summary.json` = 121,884 samples (ConvNeXt experiment v2)
- **Conflict resolved:** They describe different pipelines.

### 5. Legacy / Broken Files

- `src/data/split_dataset.py` — legacy, expects `data/hug`
- `notebooks/courseWorkCheck.ipynb` — hardcoded external paths
- `scripts/finetune_convnext_weakfix.py` — hardcoded `PROJECT_ROOT`

### 6. Integration Problems

- 242 CSV files referenced by metadata are missing.
- `train.py` cannot run without `data/splits/train_insight.csv`.
- `test_data_v3/` and `data/raw/DF40` are absent.
- v5_weakfix scripts reference missing CSVs and wrong project root.

### 7. Required Next Step

Before continuing the EDA, **obtain or generate `test_data_v3/` and run `src/data/prepare_df40_splits.py`** to produce the actual CSV split files. Only then can the EDA and training pipeline point to the same authoritative dataset.

**Recommended exact command once data is present:**

```bash
.venv/bin/python src/data/prepare_df40_splits.py --seed 42
```

This will populate `data/splits/` with the correct CSVs, including `train_insight.csv`, `val_insight.csv`, `test_insight.csv`, and `split_info.json` will no longer be stale.