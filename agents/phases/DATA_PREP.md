# DATA_PREP.md — Data Preparation

- **Title:** Data Preparation (DF40)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Build, split, and transform the DF40 dataset into binary
  real/fake train/val/test splits.
- **Status:** In Progress

## Background

DF40 is a 40-method deepfake benchmark; real faces come from FF++ and
Celeb-DF. The project needs a deterministic, one-face-per-image binary
(real/fake) split for training and a held-out identity-unique test set.

## Goals / Purpose

- Produce reproducible train/val/test splits at 256×256 with the standard
  ImageNet normalization.
- Keep raw data immutable under `data/raw/`; outputs under `data/processed/`.
- Excludes: multi-class/per-method classification (binary only).

## Input / Output

- **Input:** HF `ManhQuangAI/DF40_train` (~74.7 GB) and real faces (FF++,
  Celeb-DF); test set `ManhQuangAI/df40-test-data-v3` (30,692 images). Source
  root via `DF40_ROOT`.
- **Output:** split CSVs / processed image dirs, manifest files
  (`method,video,path`).

## How to do it (general plan)

1. Download/extract DF40 + real faces. Prefer Hugging Face
   (`hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40`);
   `download_df40.py` (Google Drive/gdown) is unreliable and not recommended.
2. Build balanced real/fake subsets (`build_df40_balanced.py`,
   `build_df40_subset.py`).
3. Build test sets and manifests (`build_test_data.py`, `build_test_data_v2.py`,
   `restructure_test_data_v3.py`).
4. Split train/val/test and apply transforms (`split_dataset.py`,
   `make_balanced_split.py`).

## Pipeline

```
HF/DF40 + real (FF++/Celeb-DF) → data/raw → build_* → split_* → data/processed
```

## Detailed plan / gotchas

- Image size **256×256**; transforms `Resize→ToTensor→Normalize`
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) — see
  [src/training/train.py](../src/training/train.py#L40).
- Source root is configurable via `DF40_ROOT` — do not hardcode machine paths
  (CQ-1).
- Data leakage was analyzed in
  [experiments/results/eval/test_data_v3-build-and-leakage.md](../experiments/results/eval/test_data_v3-build-and-leakage.md);
  keep identity-disjoint splits for eval.

## Links

- Progress: [../progress/DATA_PREP_STATUS.md](../progress/DATA_PREP_STATUS.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
