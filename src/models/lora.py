"""LoRA (Low-Rank Adaptation) cho DINOv3 ViT.

Chỉ wrap các Linear trong attention (mặc định q_proj, v_proj) bằng LoRALayer,
giữ nguyên trọng số gốc (frozen). Fine-tune nhẹ: backbone DINOv3 giữ đặc trưng
tổng quát, LoRA thích ứng theo domain deepfake — tránh sụp đặc trưng như full
finetune (lần đầu bị real-class collapse, AUC 0.45).

Cấu trúc param LoRA (mỗi layer được wrap):
  layer.{i}.attention.q_proj.lora_A  (r, in_features)
  layer.{i}.attention.q_proj.lora_B  (out_features, r)
  v_proj tương tự.

Dùng chung cho finetune_lora.py và eval (eval phải rebuild LoRA đúng config
trước khi load checkpoint).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Bọc quanh một nn.Linear có sẵn: y = base(x) + scaling * B(A(x))."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALayer chỉ bọc nn.Linear, được {type(base).__name__}")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # base frozen
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)


def apply_lora(model, r: int = 16, alpha: float = 32.0, targets=("q_proj", "v_proj")):
    """Wrap các projection của attention trong mọi block; freeze backbone còn lại.

    Trả về số layer đã wrap. LoRA params được bật requires_grad=True.
    """
    applied = 0
    for blk in model.layer:
        for name in targets:
            base = getattr(blk.attention, name)
            setattr(blk.attention, name, LoRALayer(base, r=r, alpha=alpha))
            applied += 1
    # freeze toàn bộ backbone (gồm cả base vừa wrap), sau đó mở lại LoRA
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALayer):
            for p in (m.lora_A, m.lora_B):
                p.requires_grad_(True)
    return applied
