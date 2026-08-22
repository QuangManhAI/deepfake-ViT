# MODEL_STATUS.md — Model Definitions

- **Title:** Model Definitions (DINOv3 ViT / ConvNeXt / LoRA)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Status of the model definitions and loaders.
- **Status:** Done
- **Phase doc:** [../phases/MODEL.md](../phases/MODEL.md)

## Log

- 2026-08-18: All three models (DINOv3 ViT, ConvNeXt CNN, LoRA) implemented
  and loaders verified (`strict=True`).

## Blockers (if any)

- None.

## Decisions

- Load `.safetensors` with `weights_only=True` and `strict=True`.

## Next step

- None — models are ready for training/eval.

## Links

- Phase doc: [../phases/MODEL.md](../phases/MODEL.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
