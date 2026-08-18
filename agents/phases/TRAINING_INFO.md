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

- [src/training/train.py](../src/training/train.py) — full fine-tune.
- [src/training/finetune_lora.py](../src/training/finetune_lora.py) — LoRA.
- [src/training/finetune_compare.py](../src/training/finetune_compare.py) — ViT vs CNN.

## Pipeline

```
split CSVs → set_seed → AdamW(2 LR groups) → CosineAnnealingLR → evaluate → save best
```

## Detailed plan / gotchas

- Hyperparameters: `img_size=256`, `batch_size=32`, `lr_backbone=1e-5`,
  `lr_head=1e-3`, `weight_decay=0.05`, `epochs=5`, `seed=42`.
- **Open item (ARCH-1):** checkpointing currently saves best-state only; full
  state + resume + `_last.pt` + JSONL history is deferred pending decision on
  [rules/LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md).

## Links

- Progress: [../progress/TRAINING_STATUS.md](../progress/TRAINING_STATUS.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
