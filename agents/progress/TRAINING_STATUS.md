# TRAINING_STATUS.md — Training

- **Title:** Training Loops (DINOv3 fine-tune / LoRA / ViT-vs-CNN)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Status of the training entry points and reproducibility.
- **Status:** In Progress
- **Phase doc:** [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)

## Log

- 2026-08-18: Full RNG seeding via `src/utils/seeding.py` (CQ-2).
- 2026-08-21: Strategic optimization plan EXP-01 created for maximizing accuracy (>97.5%) with LLRD, Balanced Label-Smoothed Loss, TTA, and Ensembling.

## Blockers (if any)

- ARCH-1 (open P0, deferred): full-state checkpointing + resume not yet
  implemented in `train.py`; decision pending.

## Decisions

- Script-only training; AdamW with Layer-wise LR Decay (LLRD) + cosine schedule; seed 42.
- Dedicated optimization strategy: `agents/experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md` (execution notebook planned, not yet created).

## Next step

- Run EXP-01 advanced fine-tuning pipeline on GPU to push Test Accuracy > 97.5%.

## Links

- Phase doc: [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)
- Experiment plan: [../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
