# TRAINING_INFO.md — Training

- **Title:** Training Loops (DINOv3 fine-tune / LoRA / ViT-vs-CNN)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Script-only training entry points and hyperparameters.
- **Status:** In Progress

## Background

Training must run only from scripts (never notebooks) and be reproducible and
resumable.

## Goals / Purpose

- Fine-tune the DINOv3 backbone + head; provide LoRA and ViT-vs-CNN variants.
- Reproducible runs via full RNG seeding (see `src/utils/seeding.py`).
- Record metrics per epoch for reporting.

## Input / Output

- **Input:** train/val/test CSVs (from DATA_PREP) + pretrained weights.
- **Output:** best checkpoints + metrics reports in `experiments/`.

## How to do it (general plan)

## How to do it (general plan)

- [src/training/train.py](../src/training/train.py) — standard differential LR fine-tuning.
- [src/training/finetune_lora.py](../src/training/finetune_lora.py) — LoRA adaptation.
- [src/training/finetune_compare.py](../src/training/finetune_compare.py) — ViT vs CNN comparison.
- **Advanced Execution Notebook:** [`notebooks/02_advanced_accuracy_finetuning.ipynb`](../../notebooks/02_advanced_accuracy_finetuning.ipynb) — Layer-wise LR Decay (LLRD) + 50:50 Balanced Batches via `WeightedRandomSampler`.

## Pipeline

```
Dataset (25:1 imbalance) → WeightedRandomSampler (50:50 Batches) → LLRD AdamW (gamma=0.80) → Label-Smoothed Loss (eps=0.05) → CosineAnnealingLR → Validation Checkpointing (best AUC)
```

## Detailed plan / gotchas

- **Imbalance Solution:** `WeightedRandomSampler` assigns $w_{real} = 1 / N_{real}$ and $w_{fake} = 1 / N_{fake}$ to ensure every batch contains 50% Real and 50% Fake images.
- **LLRD Schedule:** $\eta_l = 10^{-5} \cdot (0.80)^{11-l}$ (Layer 11: $10^{-5}$ down to Layer 0: $1.07 \times 10^{-6}$; Head: $10^{-3}$).
- **Regularization:** Label Smoothing $\epsilon = 0.05$ prevents overconfident logit saturation.
- **Checkpoints:** Local storage in `experiments/checkpoints/dinov3_vit_max_acc.pt`.

## Links

- Phase doc: [TRAINING_INFO.md](TRAINING_INFO.md)
- Experiment plan: [../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md)
- Progress: [../progress/TRAINING_STATUS.md](../progress/TRAINING_STATUS.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
