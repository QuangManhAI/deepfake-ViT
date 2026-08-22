# OVERVIEW.md — Project Overview & Roadmap

- **Motivation/Background**: A single living overview keeps the project's
  plan, phases, and status visible without re-reading every phase doc.
- **Purpose**: Summarize the project: dataset, models, approach, phases,
  known constraints, and pointers to status docs.
- **Overview Pipeline**: Derived from the locked `PURPOSE.md` and the roadmap
  template; updated at the start of every session/phase.
- **Detailed Plan**: §1 project; §2 plan; §3 phases; §4 known constraints;
  §5 progress pointers.
- **References**: `PURPOSE.md`, `templates/PROJECT_ROADMAP_TEMPLATE.md`,
  `phases/<PHASE>.md`.

---

## Table of Contents

- [1. Project](#1-project)
- [2. Plan](#2-plan)
- [3. Phases](#3-phases)
- [4. Known Constraints](#4-known-constraints)
- [5. Progress](#5-progress)

---

## 1. Project

DINOv3 ViT-S/16 face-deepfake detector on the **DF40** benchmark (binary:
real vs. fake), with a matched-parameter ConvNeXt CNN baseline, LoRA
variants, and attention visualization. Full brief in
[PURPOSE.md](PURPOSE.md).

## 2. Plan
 
- **Dataset:** **DF40** (40 deepfake methods; real from FF++ & Celeb-DF; one
  face per image; **binary real/fake**). Shared raw data from `/workspace/data/test_data_v3/` (30,692 images). All splits and outputs isolated in `/workspace/hoangtuan/deepfake-ViT/`.
- **Models:** DINOv3 ViT-S/16 (pretrained, `src/models/dinov3_vit.py`),
  ConvNeXt-Tiny CNN baseline (`src/models/dinov3_convnext.py`), LoRA adapters
  (`src/models/lora.py`).
- **Optimization Roadmap (EXP-01):**
  1. 50:50 Real/Fake Batch Balancing via `WeightedRandomSampler`.
  2. Layer-wise Learning Rate Decay (LLRD $\gamma = 0.80$, Layer 11 $\rightarrow$ Layer 0).
  3. Artifact-preserving facial augmentations (ColorJitter, Blur, Flip).
  4. Label-Smoothed Loss ($\epsilon = 0.05$).
  5. Validation Cutoff Optimization ($\tau^*$).
  6. Test-Time Augmentation (TTA) & ViT+CNN probability ensembling.
- **Notebooks:**
  - Comprehensive EDA: [`notebooks/00_comprehensive_dataset_eda.ipynb`](../notebooks/00_comprehensive_dataset_eda.ipynb)
  - Standard Pipeline: [`notebooks/01_full_pipeline.ipynb`](../notebooks/01_full_pipeline.ipynb)
- **Success criteria:** **Test accuracy > 97.5%** (exceeding > 95% rubric requirement).

## 3. Phases

1. [DATA_PREP.md](phases/DATA_PREP.md) — data loading, 50:50 batch balancing, transforms
2. [MODEL.md](phases/MODEL.md) — DINOv3 ViT / ConvNeXt / LoRA model definitions
3. [TRAINING_INFO.md](phases/TRAINING_INFO.md) — LLRD training loops, hyperparameters
4. [EVAL.md](phases/EVAL.md) — evaluation, TTA, threshold tuning, ViT-vs-CNN comparison
5. [EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md) — dedicated optimization strategy

## 4. Known Constraints

- **Compute:** NVIDIA GeForce RTX 4060, **8 GB VRAM** — batch size, input
  resolution, and model size must fit this budget.
- **Task:** binary classification only (real/fake); one face per image.
- **Python:** 3.11 standard (setup via `src/utils/setup_ubuntu.sh`).
- **Weights:** pretrained `model.safetensors` are gitignored and live in
  `experiments/checkpoints/weights/` (not committed).
- **Deadline:** ≥ 2026-09-01.

## 5. Progress

- [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- [DATA_SPLIT_SUMMARIZE.md](DATA_SPLIT_SUMMARIZE.md) — Báo cáo tổng hợp dữ liệu Train, Val, Test đa nguồn (FF++, Celeb-DF, DF40)
- [DATA_PREP_SUMMARY_REPORT.md](DATA_PREP_SUMMARY_REPORT.md) — Comprehensive technical data report
- [progress/MODEL_STATUS.md](progress/MODEL_STATUS.md)
- [progress/TRAINING_STATUS.md](progress/TRAINING_STATUS.md)
- [progress/EVAL_STATUS.md](progress/EVAL_STATUS.md)



---

## References

- [PURPOSE.md](PURPOSE.md) — locked project brief
- [templates/PROJECT_ROADMAP_TEMPLATE.md](templates/PROJECT_ROADMAP_TEMPLATE.md)
- [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md)
