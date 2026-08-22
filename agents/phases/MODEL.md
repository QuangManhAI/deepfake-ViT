# MODEL.md — Model Definitions

- **Title:** Model Definitions (DINOv3 ViT / ConvNeXt / LoRA)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** The DINOv3 ViT-S/16 backbone, the matched ConvNeXt CNN
  baseline, and LoRA adapters.
- **Status:** Done

## Background

The deliverable is a Vision Transformer classifier; a CNN of matched
parameter count is required for comparison. All models are implemented from
the checkpoint tensor layout.

## Goals / Purpose

- DINOv3 ViT-S/16 (embed 384, depth 12, 6 heads, 4 registers, SwiGLU MLP).
- ConvNeXt-Tiny CNN with comparable parameter count.
- LoRA adapters for parameter-efficient fine-tuning.
- Load from `.safetensors` with `strict=True` and `weights_only=True`.

## Input / Output

- **Input:** pretrained `model.safetensors` in `experiments/checkpoints/weights/`.
- **Output:** `nn.Module` graphs + loader functions.

## How to do it (general plan)

- [src/models/dinov3_vit.py](../src/models/dinov3_vit.py) — `load_dinov3()`, `DinoViT`.
- [src/models/dinov3_convnext.py](../src/models/dinov3_convnext.py) — `load_dinov3_convnext()`.
- [src/models/lora.py](../src/models/lora.py) — `apply_lora()`.

## Pipeline

```
safetensors → load_* (strict, weights_only) → nn.Module (backbone) → head
```

## Detailed plan / gotchas

- Classifier head is `Linear(384, 2)` on the CLS token (binary real/fake).
- Parameter-count parity with the CNN is verified by
  [src/experiments/model_param_count.py](../src/experiments/model_param_count.py).

## Links

- Progress: [../progress/MODEL_STATUS.md](../progress/MODEL_STATUS.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
