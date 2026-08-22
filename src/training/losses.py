"""Loss functions for deepfake detection.

Includes:
- LabelSmoothingCrossEntropy
- FocalLoss (Multi-class with label smoothing and gamma focusing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing."""

    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        
        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)
            
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets, weight=self.weight, reduction='mean')
        return (1.0 - self.smoothing) * nll + (self.smoothing / num_classes) * loss


class FocalLoss(nn.Module):
    """Focal Loss for focusing gradients on hard examples."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        
        # Soft targets for label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            if self.label_smoothing > 0:
                smooth_targets = (
                    smooth_targets * (1.0 - self.label_smoothing)
                    + self.label_smoothing / num_classes
                )

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Focal weights: (1 - p_t)^gamma
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1.0 - pt).clamp(min=1e-6) ** self.gamma

        # Cross entropy component
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Alpha weighting (optional class balance)
        if self.weight is not None:
            target_weights = self.weight[targets]
            loss = focal_weight * ce_loss * target_weights
        else:
            loss = focal_weight * ce_loss

        return loss.mean()
