"""Trích xuất đặc trưng CLS token từ DINOv3 backbone cho toàn bộ ảnh trong data/test.

Output: outputs/results/features.npz gồm:
  features  (N, 384)  — đặc trưng CLS token
  labels    (N,)      — 0 = real, 1 = fake
  paths     (N,)      — đường dẫn ảnh
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 256

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

LABELS = {"real": 0, "fake": 1}


def collect_images(data_dir: str):
    """Gom tất cả ảnh trong data_dir/{real,fake}. Trả về list (path, label)."""
    items = []
    for cls, label in LABELS.items():
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            print(f"[Cảnh báo] Không tìm thấy thư mục: {folder}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            items.extend((p, label) for p in glob.glob(os.path.join(folder, ext)))
    return items


def collect_from_csv(csv_path: str):
    """Đọc manifest CSV (path,label). Trả về list (path, label)."""
    import csv

    items = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header != ["path", "label"]:
            print(f"[Cảnh báo] Header CSV không chuẩn: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            p, lbl = row[0], int(row[1])
            if not os.path.exists(p):
                print(f"[Cảnh báo] Thiếu file: {p}")
                continue
            items.append((p, lbl))
    return items


def extract_features(model, items, device, batch_size=32):
    model.to(device).eval()
    feats, labels, paths = [], [], []

    for i in tqdm(range(0, len(items), batch_size), desc="Extracting features", unit="batch"):
        batch = items[i : i + batch_size]
        imgs = [TRANSFORM(Image.open(p).convert("RGB")) for p, _ in batch]
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = model(x)
        feats.append(f.cpu().numpy())
        labels.extend(lbl for _, lbl in batch)
        paths.extend(p for p, _ in batch)

    return np.vstack(feats), np.array(labels), np.array(paths, dtype=object)


def main():
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng DINOv3")
    parser.add_argument("--data-dir", default="data/test", help="Thư mục chứa {real, fake}")
    parser.add_argument("--csv", default=None, help="Manifest CSV (path,label). Nếu có, bỏ qua --data-dir")
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--output", default="outputs/results/features.npz")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--act", default="silu", choices=["silu", "gelu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print(f"Load model: {args.model}")
    model = load_dinov3(args.model, act=args.act, img_size=IMG_SIZE)

    items = collect_from_csv(args.csv) if args.csv else collect_images(args.data_dir)
    print(f"Tổng ảnh: {len(items)} (real={sum(1 for _, l in items if l == 0)}, fake={sum(1 for _, l in items if l == 1)})")
    if not items:
        sys.exit("Không có ảnh nào — kiểm tra lại --data-dir / --csv")

    feats, labels, paths = extract_features(model, items, device, args.batch_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, features=feats, labels=labels, paths=paths)
    print(f"Đã lưu {args.output}: features={feats.shape}, labels={labels.shape}")


if __name__ == "__main__":
    main()
