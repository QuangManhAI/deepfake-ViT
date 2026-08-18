# TRAINING_STATUS.md — Training

- **Title:** Training Loops (DINOv3 fine-tune / LoRA / ViT-vs-CNN)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Status of the training entry points and reproducibility.
- **Status:** In Progress
- **Phase doc:** [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)

## Log

- 2026-08-18: Full RNG seeding via `src/utils/seeding.py` (CQ-2).

## Blockers (if any)

- ARCH-1 (open P0, deferred): full-state checkpointing + resume not yet
  implemented in `train.py`; decision pending.

## Decisions

- Script-only training; AdamW 2-LR-groups + cosine schedule; seed 42.

## Next step

- Re-run `train.py` in the GPU env; then resolve ARCH-1 checkpointing.

## Links

- Phase doc: [../phases/TRAINING_INFO.md](../phases/TRAINING_INFO.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
