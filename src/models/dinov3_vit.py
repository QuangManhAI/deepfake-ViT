"""
DINOv3 ViT-Small/16 backbone — tái dựng theo đúng tên tensor trong `models/model.safetensors`.

Định dạng tên param (211 tensor):
  embeddings.cls_token            [1, 1, 384]
  embeddings.mask_token           [1, 1, 384]
  embeddings.register_tokens      [1, 4, 384]
  embeddings.patch_embeddings     Conv2d(3, 384, kernel=16, stride=16)
  layer.{0..11}.norm1 / norm2     LayerNorm(384)
  layer.{0..11}.attention.q_proj / k_proj / v_proj / o_proj   Linear(384, 384)
  layer.{0..11}.layer_scale1 / layer_scale2.lambda1           Parameter(384)
  layer.{0..11}.mlp.up_proj (384→1536) / down_proj (1536→384)
  norm                           LayerNorm(384)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    """LayerScale: nhân output của attention/MLP với một vector học được (init = 1.0)."""

    def __init__(self, dim: int, init_values: float = 1.0):
        super().__init__()
        self.lambda1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class Attention(nn.Module):
    """Multi-head self-attention với Q/K/V/O tách riêng (khớp tên q_proj/k_proj/v_proj/o_proj)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)  # DINOv3 bỏ bias ở K (file không có k_proj.bias)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, N, H, D).transpose(1, 2)  # (B,H,N,D)
        k = self.k_proj(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.o_proj(out)


class Mlp(nn.Module):
    """MLP: chuẩn (up→act→down) hoặc gated (act(gate) * up → down) như SwiGLU."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 act: str = "silu", gated: bool = False):
        super().__init__()
        self.gated = gated
        self.up_proj = nn.Linear(in_features, hidden_features)
        if gated:
            self.gate_proj = nn.Linear(in_features, hidden_features)
        self.down_proj = nn.Linear(hidden_features, out_features)
        self.act = {"silu": nn.SiLU(), "gelu": nn.GELU()}[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act(self.up_proj(x)))


class Block(nn.Module):
    """Transformer block pre-norm: x = x + ls1(attn(norm1(x))); x = x + ls2(mlp(norm2(x)))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, act: str = "silu",
                 gated_mlp: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, num_heads)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, act=act, gated=gated_mlp)
        self.layer_scale2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DinoViT(nn.Module):
    """DINOv3 ViT-Small/16 backbone — chỉ trích xuất đặc trưng (không có classification head)."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        act: str = "silu",
        gated_mlp: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.num_patches = (img_size // patch_size) ** 2
        self.gated_mlp = gated_mlp

        # ---- embeddings ----
        self.embeddings = nn.Module()
        self.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # chỉ dùng khi pretrain, giữ để khớp state_dict
        self.embeddings.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.embeddings.patch_embeddings = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # ---- transformer blocks ----
        self.layer = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, act=act, gated_mlp=gated_mlp) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về đặc trưng CLS token (B, embed_dim) sau layer cuối."""
        B = x.shape[0]

        # Patch embed → (B, N, C)
        x = self.embeddings.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)

        # Ghép [CLS, register_tokens, patches] — đúng thứ tự DINOv3 (CLS ở vị trí 0)
        cls = self.embeddings.cls_token.expand(B, -1, -1)
        regs = self.embeddings.register_tokens.expand(B, -1, -1)
        x = torch.cat([cls, regs, x], dim=1)

        for blk in self.layer:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


def load_dinov3(model_path: str, act: str = "silu", **kwargs) -> DinoViT:
    """Load DINOv3 backbone từ file .safetensors. Tự động phát hiện gated MLP."""
    from safetensors.torch import load_file

    sd = load_file(model_path)

    # Auto-detect gated_mlp từ state_dict
    if "gated_mlp" not in kwargs:
        kwargs["gated_mlp"] = any("gate_proj" in k for k in sd.keys())

    model = DinoViT(act=act, **kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Không khớp state_dict! missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model
