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
  face per image; **binary real/fake**). Train from HF
  `ManhQuangAI/DF40_train` (~74.7 GB); test from HF
  `ManhQuangAI/df40-test-data-v3` (30,692 images, imagefolder). Source root
  configurable via `DF40_ROOT`.
- **Models:** DINOv3 ViT-S/16 (pretrained, `src/models/dinov3_vit.py`),
  ConvNeXt-Tiny CNN baseline (`src/models/dinov3_convnext.py`), LoRA adapters
  (`src/models/lora.py`).
- **Approach:** pretrained DINOv3 → fine-tune (full / LoRA) → linear-probe
  evals; ViT-vs-CNN comparison at matched params; attention visualization.
- **Monitoring:** script-only runs (`src/training/`, `src/eval/`); JSON
  reports + `experiments/results/`; see
  [rules/LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) and
  [rules/RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md).
- **Success criteria:** **test accuracy > 95%** on the held-out DF40 test
  split; ViT-vs-CNN comparison and attention viz are required deliverables.

## 3. Phases

1. [DATA_PREP.md](phases/DATA_PREP.md) — data loading, build/split, transforms
2. [MODEL.md](phases/MODEL.md) — DINOv3 ViT / ConvNeXt / LoRA model definitions
3. [TRAINING_INFO.md](phases/TRAINING_INFO.md) — training loops, hyperparameters
4. [EVAL.md](phases/EVAL.md) — evaluation, metrics, ViT-vs-CNN comparison

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
- [progress/MODEL_STATUS.md](progress/MODEL_STATUS.md)
- [progress/TRAINING_STATUS.md](progress/TRAINING_STATUS.md)
- [progress/EVAL_STATUS.md](progress/EVAL_STATUS.md)

---

## References

- [PURPOSE.md](PURPOSE.md) — locked project brief
- [templates/PROJECT_ROADMAP_TEMPLATE.md](templates/PROJECT_ROADMAP_TEMPLATE.md)
- [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md)
