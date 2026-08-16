"""Visualize attention maps từ DINOv3 ViT (CLS token → patches).

- Extract attention từ layer cuối, CLS token tới các patch token (bỏ qua register tokens)
- Average qua các head, reshape thành grid, upsample về kích thước ảnh
- Overlay heatmap lên ảnh gốc (dùng PIL, không cần matplotlib)

Output: outputs/attention/<name>_<layer>.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 256
NUM_REGISTERS = 4  # DINOv3 có 4 register tokens sau CLS


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def jet_colormap(x):
    """Colormap jet thủ công (0=blue → 0.5=cyan → 1=red). x: (H, W) trong [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)  # (H, W, 3)


@torch.no_grad()
def get_attention_maps(model, x, layers=(5, 8, 11)):
    """Trả về dict {layer_idx: attention_map (H, W)} cho CLS token.

    x: (1, 3, H, W). attention_map đã normalize về [0,1].
    """
    model.eval()
    x = x.to(next(model.parameters()).device)

    # Forward — attention được lưu trong mỗi layer.attention.attn
    model(x)

    num_patches = IMG_SIZE // 16
    maps = {}
    for L in layers:
        attn = model.layer[L].attention.attn  # (1, num_heads, N, N)
        attn = attn[0]  # (num_heads, N, N)
        # CLS (row 0) → patches (cols từ NUM_REGISTERS+1 trở đi)
        cls_attn = attn[:, 0, NUM_REGISTERS + 1:]  # (num_heads, num_patches)
        head_avg = cls_attn.mean(dim=0)  # (num_patches,)
        # Normalize về [0,1]
        amap = head_avg.cpu().numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        maps[L] = amap.reshape(num_patches, num_patches)  # (16, 16)

    return maps


def overlay(image, amap, alpha=0.6):
    """Overlay heatmap lên ảnh. image: PIL RGB, amap: (H, W) trong [0,1]."""
    # Upsample attention map về kích thước ảnh
    amap_img = Image.fromarray((amap * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    amap_np = np.array(amap_img) / 255.0

    # Colormap
    heat = jet_colormap(amap_np)  # (H, W, 3) trong [0,1]
    heat = (heat * 255).astype(np.uint8)

    # Blend
    base = np.array(image.convert("RGB")).astype(np.float32)
    blended = base * (1 - alpha) + heat * alpha
    return Image.fromarray(blended.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 attention")
    parser.add_argument("--model", default="models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    parser.add_argument("--real-dir", default="data/test/real")
    parser.add_argument("--fake-dir", default="data/test/fake")
    parser.add_argument("--n-samples", type=int, default=3, help="Số ảnh mỗi lớp")
    parser.add_argument("--output-dir", default="outputs/attention")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--layers", default="5,8,11", help="Các layer cần trích attention")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    layers = tuple(int(x) for x in args.layers.split(","))

    # ---------- Load model ----------
    print(f"Load model: {args.model}")
    model = load_dinov3(args.model, img_size=IMG_SIZE).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Layers: {len(model.layer)}")

    # ---------- Gather images ----------
    tf = build_transform()
    samples = []
    for label, d in [("real", args.real_dir), ("fake", args.fake_dir)]:
        files = sorted(glob.glob(os.path.join(d, "*.*")))[:args.n_samples]
        for f in files:
            samples.append((label, f))
    print(f"Tổng ảnh: {len(samples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for label, path in samples:
        name = os.path.splitext(os.path.basename(path))[0]
        orig = Image.open(path).convert("RGB")
        x = tf(orig).unsqueeze(0)  # (1, 3, H, W)

        maps = get_attention_maps(model, x, layers=layers)

        for L, amap in maps.items():
            viz = overlay(orig, amap)
            out_path = os.path.join(args.output_dir, f"{label}_{name}_layer{L}.png")
            viz.save(out_path)

        # Cũng lưu ảnh gốc để đối chiếu
        orig_out = os.path.join(args.output_dir, f"{label}_{name}_original.png")
        orig.save(orig_out)
        print(f"  ✓ {label}/{name}: {len(maps)} attention maps")

    print(f"\nĐã lưu attention maps vào: {args.output_dir}")


if __name__ == "__main__":
    main()
