"""Đánh giá detection rate (bắt ảnh fake) trên DF40 fake-only (blendface + ddim).

Dùng checkpoint fine-tuned ViT sẵn có (outputs/checkpoints/dinov3_finetuned.pt).
- blendface: face-swap fake, ddim: diffusion-synthesis fake.
- Metric: detection rate = % ảnh fake bị model gán p(fake) > 0.5.

Chạy:
  .venv/bin/python scripts/eval_df40_fake.py --n 500
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DinoViTClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def sample_images(root, n, seed=42):
    rng = random.Random(seed)
    imgs = [p for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)]
    rng.shuffle(imgs)
    return imgs[:n]


@torch.no_grad()
def detection_rate(model, paths, device, batch_size=32):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    model.eval()
    probs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  Infer", unit="batch"):
        batch = paths[i : i + batch_size]
        imgs = [tf(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(imgs).to(device)
        logits = model(x)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    probs = np.array(probs)
    return {
        "n": len(probs),
        "detection_rate": float((probs > 0.5).mean()),
        "mean_p_fake": float(probs.mean()),
        "median_p_fake": float(np.median(probs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ckpt", default="outputs/checkpoints/dinov3_finetuned.pt")
    ap.add_argument("--backbone", default="models/dinov3_small/model.safetensors")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    print(f"Device: {device}")

    backbone = load_dinov3(args.backbone, img_size=IMG_SIZE)
    model = DinoViTClassifier(backbone)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.ckpt} (epoch {ckpt.get('epoch')})")

    folders = {"blendface (face-swap)": "data/blendface/frames",
               "ddim (diffusion)": "data/ddim"}
    print(f"\n=== Detection rate on DF40 fake samples (n={args.n} each) ===")
    for name, root in folders.items():
        paths = sample_images(root, args.n)
        r = detection_rate(model, paths, device)
        print(f"{name:<22} : detected {r['detection_rate']*100:5.1f}%  "
              f"(mean p(fake)={r['mean_p_fake']:.3f}, median={r['median_p_fake']:.3f})")


if __name__ == "__main__":
    main()
