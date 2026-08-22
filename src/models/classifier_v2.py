"""Enhanced Deepfake Classifier architectures.

Includes:
- DinoViTMLPClassifier: ViT backbone with LayerNorm, Dropout, and 2-layer GELU MLP head
- DualBackboneClassifier: Joint fusion of DINOv3 ViT-S/16 and ConvNeXt-Tiny
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoViTMLPClassifier(nn.Module):
    """DINOv3 ViT with LayerNorm, Dropout, and 2-layer GELU MLP head.
    
    Provides higher capacity than a simple linear probe while preventing overfitting.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # CLS token (B, embed_dim)
        return self.head(feat)


class DinoConvNextClassifier(nn.Module):
    """DINOv3 ConvNeXt-Tiny classifier with MLP head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        in_dim = 768  # ConvNeXt-Tiny output dimension

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B, 768)
        return self.head(feat)


class EnsembleClassifier(nn.Module):
    """Joint inference ensemble combining ViT and ConvNeXt."""

    def __init__(
        self,
        vit_model: nn.Module,
        cnn_model: nn.Module,
        vit_weight: float = 0.65,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.cnn_model = cnn_model
        self.vit_weight = vit_weight

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_logits = self.vit_model(x)
        cnn_logits = self.cnn_model(x)
        vit_probs = F.softmax(vit_logits, dim=-1)
        cnn_probs = F.softmax(cnn_logits, dim=-1)
        
        ensemble_probs = self.vit_weight * vit_probs + (1.0 - self.vit_weight) * cnn_probs
        return ensemble_probs
