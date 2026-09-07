"""Canonical image preprocessing.

Every notebook, training script and evaluation script should use these
transforms so image size, normalization and augmentation are identical
everywhere. Mirrors the constants used by ``src/training/train.py``.
"""

from __future__ import annotations

from torchvision import transforms

# ---------------------------------------------------------------------------
# Canonical constants. Do not redefine elsewhere.
# ---------------------------------------------------------------------------
IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]  # ImageNet
STD = [0.229, 0.224, 0.225]

# Training: resize + horizontal flip only. Deliberately conservative --
# aggressive colour/blur augmentation risks destroying the compression and
# edge artefacts that distinguish fakes.
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Evaluation: deterministic, no augmentation.
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_transforms(train: bool = False):
    """Return the canonical transform for the given phase."""
    return TRAIN_TF if train else EVAL_TF


def describe() -> str:
    return (
        "CANONICAL PREPROCESSING\n"
        f"  image size   : {IMG_SIZE}x{IMG_SIZE}  (MODEL INPUT)\n"
        f"  normalization: mean={MEAN} std={STD}\n"
        "  train aug    : Resize + RandomHorizontalFlip\n"
        "  eval aug     : Resize only (deterministic)\n"
        "  note         : original image dimensions vary by source; all are\n"
        "                 resized to the model input size before the network."
    )


def denormalize(tensor):
    """Undo normalization for visualization. Returns HWC array in [0, 1]."""
    import numpy as np
    import torch

    img = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    img = img.transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return img.clip(0, 1)


if __name__ == "__main__":  # pragma: no cover
    print(describe())
