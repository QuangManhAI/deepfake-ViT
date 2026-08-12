"""
DINOv3 ConvNeXt-Tiny backbone — tái dựng theo đúng tên tensor trong safetensors.

Kiến trúc ConvNeXt-Tiny (theo "A ConvNet for the 2020s", Liu et al. 2022):
  - Stem: Conv2d(3→96, 4×4, stride 4) + LayerNorm
  - 4 stages: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
  - Mỗi block: depthwise_conv 7×7 → LN → pw_conv1 (expand 4×) → GELU → pw_conv2 (project) → LayerScale
  - Downsample: LN → Conv2d(2×2, stride 2) giữa các stage
  - Output: GAP → 768-dim feature vector

Định dạng tên param (180 tensor):
  stages.{0..3}.downsample_layers.{0,1}.weight/bias
  stages.{0..3}.layers.{0..N}.depthwise_conv.weight/bias
  stages.{0..3}.layers.{0..N}.layer_norm.weight/bias
  stages.{0..3}.layers.{0..N}.pointwise_conv1.weight/bias
  stages.{0..3}.layers.{0..N}.pointwise_conv2.weight/bias
  stages.{0..3}.layers.{0..N}.gamma
  layer_norm.weight/bias
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """Một ConvNeXt block: depthwise 7×7 → LN → pw1(expand) → GELU → pw2(project) → LayerScale + residual."""

    def __init__(self, dim: int, expand_ratio: int = 4, eps: float = 1e-6):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.pointwise_conv1 = nn.Linear(dim, hidden_dim)
        self.pointwise_conv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))  # LayerScale, init=1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Depthwise spatial mixing (channels_first)
        x = self.depthwise_conv(x)
        # → channels_last cho LayerNorm + Linear
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = F.gelu(x)
        x = self.pointwise_conv2(x)
        # → channels_first trở lại
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x * self.gamma.view(1, -1, 1, 1)  # LayerScale
        return x + identity


class DownsampleBlock(nn.Module):
    """Downsample giữa các stage: LayerNorm → Conv2d(stride=2)."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=eps)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LN trong channels_last rồi permute về channels_first cho conv
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.conv(x)


class StemBlock(nn.Module):
    """Patchify stem cho stage 0: Conv2d(3→dim, 4×4, stride=4) + LayerNorm."""

    def __init__(self, out_dim: int = 96, eps: float = 1e-6):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=4, stride=4)
        self.layer_norm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # patchify
        x = x.permute(0, 2, 3, 1)  # channels_last cho LN
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # trở về channels_first
        return x


class DinoConvNext(nn.Module):
    """DINOv3 ConvNeXt-Tiny backbone — chỉ trích xuất đặc trưng (feature extractor)."""

    def __init__(
        self,
        depths: list = None,
        dims: list = None,
        expand_ratio: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if depths is None:
            depths = [3, 3, 9, 3]        # Tiny
        if dims is None:
            dims = [96, 192, 384, 768]   # Tiny
        self.dims = dims
        self.depths = depths
        self.embed_dim = dims[-1]         # 768 — để tương thích với pipeline so sánh

        # ---- Stem (stage 0 patchify) ----
        self.stages = nn.ModuleList()
        stage0 = nn.ModuleDict({
            "downsample_layers": nn.ModuleList([
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),   # patchify conv
                nn.LayerNorm(dims[0], eps=eps),                     # post-stem LN
            ]),
        })
        self.stages.append(stage0)

        # ---- Stage 0 blocks ----
        stage0["layers"] = nn.ModuleList([
            ConvNeXtBlock(dims[0], expand_ratio, eps) for _ in range(depths[0])
        ])

        # ---- Stages 1-3 (downsample + blocks) ----
        for i in range(1, 4):
            stage = nn.ModuleDict({
                "downsample_layers": nn.ModuleList([
                    nn.LayerNorm(dims[i - 1], eps=eps),              # pre-downsample LN
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),  # downsample conv
                ]),
                "layers": nn.ModuleList([
                    ConvNeXtBlock(dims[i], expand_ratio, eps) for _ in range(depths[i])
                ]),
            })
            self.stages.append(stage)

        # ---- Final LayerNorm ----
        self.layer_norm = nn.LayerNorm(dims[-1], eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về feature vector sau GAP (B, embed_dim)."""
        for stage in self.stages:
            # Downsample (stem cho stage 0, downsample cho stage 1-3)
            ds = stage["downsample_layers"]
            if len(ds) == 2 and isinstance(ds[0], nn.Conv2d):
                # Stage 0: Conv2d stem + LN
                x = ds[0](x)
                x = x.permute(0, 2, 3, 1)
                x = ds[1](x)
                x = x.permute(0, 3, 1, 2)
            else:
                # Stage 1-3: LN + downsample Conv2d
                x = x.permute(0, 2, 3, 1)
                x = ds[0](x)
                x = x.permute(0, 3, 1, 2)
                x = ds[1](x)

            # ConvNeXt blocks
            for blk in stage["layers"]:
                x = blk(x)

        # Global Average Pooling + final LN
        x = x.mean(dim=[-2, -1])  # GAP: (B, C, H, W) → (B, C)
        x = self.layer_norm(x)    # (B, embed_dim)
        return x


def load_dinov3_convnext(model_path: str, **kwargs) -> DinoConvNext:
    """Load DINOv3 ConvNeXt backbone từ file .safetensors."""
    from safetensors.torch import load_file

    sd = load_file(model_path)
    model = DinoConvNext(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Không khớp state_dict! missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model
