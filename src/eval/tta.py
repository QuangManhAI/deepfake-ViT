"""Test-Time Augmentation (TTA) for Deepfake Detection Inference.

Averages model probability predictions across subtle geometric and photometric augmentations
to reduce predictive variance and improve accuracy on edge cases.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def predict_batch_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    use_flips: bool = True,
    use_multi_lighting: bool = True,
    device_type: str = "cuda",
) -> torch.Tensor:
    """Run TTA on a batch of normalized input tensors (B, 3, 256, 256).
    
    Returns:
        prob_fake: Tensor of shape (B,) with averaged probability of being Fake (label 1).
    """
    model.eval()
    all_probs = []

    # 1. Original Image
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits_orig = model(images)
        probs_orig = F.softmax(logits_orig, dim=-1)[:, 1]
    all_probs.append((probs_orig, 0.40))  # Weight 0.40

    # 2. Horizontal Flip
    if use_flips:
        images_flip = torch.flip(images, dims=[-1])
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_flip = model(images_flip)
            probs_flip = F.softmax(logits_flip, dim=-1)[:, 1]
        all_probs.append((probs_flip, 0.30))  # Weight 0.30

    # 3. Brightness variation (+5% and -5% in normalized space)
    if use_multi_lighting:
        # Since images are normalized with ImageNet mean/std, slight shift
        images_bright = images + 0.05
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_bright = model(images_bright)
            probs_bright = F.softmax(logits_bright, dim=-1)[:, 1]
        all_probs.append((probs_bright, 0.15))

        images_dark = images - 0.05
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_dark = model(images_dark)
            probs_dark = F.softmax(logits_dark, dim=-1)[:, 1]
        all_probs.append((probs_dark, 0.15))

    total_weight = sum(w for _, w in all_probs)
    final_probs = sum(p * w for p, w in all_probs) / total_weight
    return final_probs
