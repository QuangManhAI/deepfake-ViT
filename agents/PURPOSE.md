# PURPOSE.md — Project Brief & Requirements

- **Motivation/Background**: This file is the source of truth for *why* the
  project exists. Every phase doc, experiment, and progress entry derives from
  it, so a vague or wrong PURPOSE.md propagates into everything downstream.
- **Purpose**: Document the original brief: the problem, success criteria,
  scope boundaries, constraints, and audience.
- **Overview Pipeline**: Write/import the brief here → the agent runs a
  clarifying interview → you review and lock the final version → the roadmap
  is generated from it (see
  [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) Step 2–3).
- **Detailed Plan**: §1 brief; §2 clarifying answers; §3 locked objective.
- **References**: `HOW_TO_SETUP_AI_AGENT.md`, `OVERVIEW.md`,
  `templates/PROJECT_ROADMAP_TEMPLATE.md`.

---

## Table of Contents

- [1. Original Brief](#1-original-brief)
- [2. Clarifying Answers](#2-clarifying-answers)
- [3. Locked Objective](#3-locked-objective)
- [References](#references)

---

## 1. Original Brief

ViT face deepfake image classification

- **Goal**: Classify images using Vision Transformer.
- **Dataset**: Face deepfake related dataset, only 1 face per image.
- **Model**: ViT (patch embedding + Transformer encoder).
- **Task Type**: Classification.
- **Extension**: Compare ViT vs. CNN (same parameter count); visualize attention.

## 2. Clarifying Answers

Clarified via interview on 2026-08-18.

- **Problem/motivation**: This is a **coursework assignment** — the deliverable
  is a ViT-based face deepfake classifier satisfying a rubric. The driving
  question is *"can a Vision Transformer classify real vs. fake faces at a
  rubric-passing level?"*, and how it compares to a similarly sized CNN.
- **Success criteria**: **test accuracy > 95%** on the held-out test split
  (the explicit rubric target). The ViT-vs-CNN comparison and attention
  visualization are **required** deliverables, not optional extras.
- **Scope boundaries**:
  - **IN**: ViT classification; ViT vs. CNN comparison at matched parameter
    count; attention-map visualization.
  - **IN**: Pretrained ViT weights are allowed.
  - **OUT**: No web API, application, deployment, or serving layer — the
    deliverable is the offline classifier plus analysis.
- **Model**: **DINOv3 ViT-Small/16** — self-supervised ViT from Meta AI,
  `embed_dim=384`, `depth=12`, `num_heads=6`, 4 registers, SwiGLU gated MLP
  (see [src/models/dinov3_vit.py](../src/models/dinov3_vit.py)). Pretrained
  weights loaded from a local
  `experiments/checkpoints/weights/model.safetensors` (sourced from
  `facebook/dinov3-*` via Hugging Face Hub). Input **256×256** (patch 16),
  classification head `Linear(384, 2)`. LoRA and ConvNeXt-CNN variants also
  exist in the repo for the ViT-vs-CNN comparison.
- **Constraints**:
  - **Compute**: NVIDIA GeForce RTX 4060, **8 GB VRAM** — batch size, input
    resolution, and model size must fit this budget.
  - **Dataset**: **DF40 (Deepfake-40)** — a benchmark of **40 deepfake
    generation methods**; real faces come from **FF++ (FaceForensics++)** and
    **Celeb-DF**. One face per image. **Binary classification only** — every
    image is labeled **real** or **fake**, no other categories (2 classes).
    Training data from HF
    `ManhQuangAI/DF40_train` (~74.7 GB); test data from HF
    `ManhQuangAI/df40-test-data-v3` (imagefolder, **30,692** test images,
    ~4.53 GB, single `test` split).
  - **Time**: **Deadline ≥ 2026-09-01** — at least two weeks from the briefing
    date (2026-08-18); use the time budget to prioritize the required
    deliverables over any optional extras.
  - **Tooling**: Python 3.9; PyTorch ≥ 2.7.1 + torchvision (installed
    separately with CUDA index `cu124`); `timm` ≥ 1.0.20; `numpy`, `pillow`,
    `tqdm`, `scikit-learn`, `safetensors`, `matplotlib`, `huggingface_hub`
    (see [requirements.txt](../requirements.txt)). Dataset download via
    `gdown`/`rclone`.
- **Audience/context**: **Coursework** — primary audience is the course
  grader/teacher. Results must follow the
  [RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md) 5W1H rules so they are
  verifiable from the rubric's perspective.

## 3. Locked Objective

Build and train a **DINOv3 ViT-Small/16** model to classify **DF40 face
deepfake images** (binary: real vs. fake) using **pretrained weights**, with
**test accuracy > 95%** on the held-out DF40 test split, within an 8 GB VRAM
budget. The project must also **compare the ViT against a CNN of matched
parameter count** and **visualize the ViT's attention maps** — both required
deliverables — while producing a clean, reproducible, script-only training
pipeline and a 5W1H-documented analysis.

> **Status**: Locked 2026-08-18 — binary (real/fake) classification confirmed;
> deadline confirmed as ≥ 2026-09-01. No open items remain.

---

## References

- [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md) — the setup workflow
- [OVERVIEW.md](OVERVIEW.md) — project plan derived from this brief
- [rules/RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md) — 5W1H result rules
