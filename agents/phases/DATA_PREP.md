# DATA_PREP.md — Data Preparation

- **Title:** Data Preparation (DF40 Benchmark & Multi-Method Evaluation)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-22
- **Description:** Build, split, and transform the DF40 dataset into binary
  real/fake train/val/test splits, including 1:1 balanced and per-method evaluation sets.
- **Status:** Done

## Background

DF40 is a 40-method deepfake benchmark; real faces come from FaceForensics++ (FF++) and
Celeb-DF. The project requires a deterministic, identity-disjoint binary (real/fake) split
for model training/validation, as well as balanced, zero-leakage evaluation test sets for
all 40 deepfake generation methods.

## Goals / Purpose

- Produce reproducible train/val/test splits at 256×256 with standard ImageNet normalization.
- Enforce strict **Zero Data Leakage** at the Identity/Subject/Video level.
- Generate **1:1 Balanced** and **Full** test sets for each of the 40 fake generation methods.
- Keep raw external data strictly immutable under `/workspace/data/`; all outputs stored in `data/splits/` and `data/splits/methods/`.

## Input / Output

- **Input:**
  - DF40 Fake Training Pool: `DF40_train_manifest.csv` / `DF40_train_extracted` (692,158 fake frames across 31 methods)
  - Real Face Datasets: FaceForensics++ (`original_sequences/youtube/c23/frames` - 999 videos, 31,968 frames) and Celeb-DF-v2 (`Celeb-real`, `YouTube-real`)
  - Benchmark Test Suite: `test_data_v3` (29,691 verified images across 40 fake methods and 1,177 real face identities)
- **Output:**
  - Standard Identity-Disjoint: `data/splits/train.csv`, `val.csv`, `test.csv` (70/15/15 ratio, 0% ID overlap)
  - 1:1 Balanced Identity-Disjoint: `data/splits/train_balanced.csv`, `val_balanced.csv`, `test_balanced.csv`
  - High-Scale Training Pool: `data/splits/train_combined_balanced.csv` (58,958 imgs, 1:1 balanced: FF++ & Celeb-DF Real + DF40 Fake), `train_pool_693k.csv` (652,421 imgs)
  - Method-Specific Test Suites: `data/splits/methods/test_<method>_balanced.csv`, `test_<method>_full.csv`, `test_<method>_detailed.csv`, and `benchmark_test_<method>_balanced.csv` (195 files total)
  - Metadata: `data/splits/split_info.json`, `data/splits/methods_summary.json`, `data/processed/data_prep_manifest.json`

## Evaluation Protocols

1. **Protocol 1: Identity-Disjoint Splits (Zero Leakage Protocol - Primary)**
   - All 22,237 unique identities partitioned deterministically (seed 42) into 70% Train / 15% Val / 15% Test.
   - Exact 0% identity, subject, or video overlap between Train, Val, and Test.
2. **Protocol 2: High-Scale Combined Training Pool**
   - Combines 20,219 FaceForensics++ Real frames and 9,268 Celeb-DF-v2 Real frames with DF40 fake training frames.
   - Any video ID or identity belonging to Test or Val is strictly excluded from training (0% leakage).
3. **Protocol 3: 1:1 Balanced Splits**
   - Equal 50:50 Real:Fake class balance (58,958 train / 6,550 val) for rapid training convergence and unbiased threshold tuning.
4. **Protocol 4: Full Benchmark Suite**
   - All 32,281 benchmark samples in `test_full.csv` covering all 40 DF40 methods and Celeb-DF-v2 test benchmark.
5. **Protocol 5: Per-Method Evaluation Benchmark**
   - Dedicated balanced and full test sets for all 40 deepfake generation techniques (GANs, Diffusion models, FaceSwap, Expression/Pose reenactment, Audio-driven synthesis).

## Links

- Master Summary: [../DATA_SPLIT_SUMMARIZE.md](../DATA_SPLIT_SUMMARIZE.md)
- Technical Report: [../DATA_PREP_SUMMARY_REPORT.md](../DATA_PREP_SUMMARY_REPORT.md)
- Progress: [../progress/DATA_PREP_STATUS.md](../progress/DATA_PREP_STATUS.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
- Unit Tests: [../../tests/test_data_prep.py](../../tests/test_data_prep.py)


