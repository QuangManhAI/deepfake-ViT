"""Inference FaceForensics++ benchmark (1000 ảnh không nhãn) với 2 model.

- Dùng linear probe đã train trên DeepFakeFace để dự đoán
- 2 model: ViT-S/16 Plus + ConvNeXt-Tiny
- Output: predictions CSV + thống kê phân phối dự đoán
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 256


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_csv(csv_path: str):
    items = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                p, lbl = row[0], int(row[1])
                if os.path.exists(p):
                    items.append((p, lbl))
    return items


@torch.no_grad()
def extract_features(model, items, device, batch_size=32, desc="Extracting"):
    """Trích xuất feature từ list (path, label). Trả về (features, labels)."""
    model.to(device).eval()
    tf = build_transform()
    feats, labels = [], []
    for i in tqdm(range(0, len(items), batch_size), desc=desc, unit="batch"):
        batch = items[i : i + batch_size]
        imgs = [tf(Image.open(p).convert("RGB")) for p, _ in batch]
        x = torch.stack(imgs).to(device)
        f = model(x)
        feats.append(f.cpu().numpy())
        labels.extend(lbl for _, lbl in batch)
    return np.vstack(feats), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Inference FF++ benchmark")
    parser.add_argument("--benchmark-dir", default="data/faceforensics_benchmark_images")
    parser.add_argument("--train-csv", default="data/splits/train_insight.csv",
                        help="CSV train để train linear probe")
    parser.add_argument("--vit-model", default="experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    parser.add_argument("--cnn-model", default="experiments/checkpoints/weights/dinov3_next_cnn/model-2.safetensors")
    parser.add_argument("--output-dir", default="experiments/results/benchmark")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------- Gather benchmark images ----------
    img_dir = args.benchmark_dir
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    img_paths = [os.path.join(img_dir, f) for f in img_files]
    print(f"Benchmark images: {len(img_paths)}")

    # ---------- Load train data (DeepFakeFace) ----------
    print(f"\nLoad train data: {args.train_csv}")
    train_items = load_csv(args.train_csv)
    print(f"  Train samples: {len(train_items)}")

    # ---------- Models ----------
    models_config = [
        {"name": "ViT-S16+", "key": "vit",
         "loader": load_dinov3,
         "path": args.vit_model,
         "kwargs": {"img_size": IMG_SIZE}},
        {"name": "ConvNeXt-T", "key": "cnn",
         "loader": load_dinov3_convnext,
         "path": args.cnn_model,
         "kwargs": {}},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    all_predictions = {}

    for cfg in models_config:
        print(f"\n{'='*60}")
        print(f"→ {cfg['name']} ({cfg['path']})")
        t0 = time.time()

        # Load model
        model = cfg["loader"](cfg["path"], **cfg["kwargs"])
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.1f}M | Dim: {model.embed_dim}")

        # Extract train features
        X_train, y_train = extract_features(model, train_items, device,
                                            args.batch_size, f"  Train feat ({cfg['key']})")

        # Extract benchmark features (nhãn giả 0, chỉ cần feature)
        bench_items = [(p, 0) for p in img_paths]
        X_bench, _ = extract_features(model, bench_items, device,
                                      args.batch_size, f"  Bench feat ({cfg['key']})")

        # Train linear probe
        print(f"  Training linear probe...")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
        )
        clf.fit(X_train, y_train)

        # Predict
        y_prob = clf.predict_proba(X_bench)[:, 1]  # probability of fake
        y_pred = (y_prob > 0.5).astype(int)

        n_fake = int(y_pred.sum())
        n_real = len(y_pred) - n_fake
        print(f"  Predictions: REAL={n_real} ({n_real/len(y_pred)*100:.1f}%) | "
              f"FAKE={n_fake} ({n_fake/len(y_pred)*100:.1f}%)")
        print(f"  Time: {time.time() - t0:.0f}s")

        all_predictions[cfg["key"]] = {
            "name": cfg["name"],
            "prob_fake": y_prob,
            "pred_label": y_pred,
            "params_M": round(n_params, 1),
        }

        del model
        if device in ("cuda", "mps"):
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------- Save predictions ----------
    # CSV: filename, vit_prob, vit_label, cnn_prob, cnn_label
    csv_path = os.path.join(args.output_dir, "benchmark_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "vit_prob_fake", "vit_label", "cnn_prob_fake", "cnn_label"])
        for i, fname in enumerate(img_files):
            writer.writerow([
                fname,
                f"{all_predictions['vit']['prob_fake'][i]:.6f}",
                int(all_predictions['vit']['pred_label'][i]),
                f"{all_predictions['cnn']['prob_fake'][i]:.6f}",
                int(all_predictions['cnn']['pred_label'][i]),
            ])
    print(f"\nĐã lưu predictions: {csv_path}")

    # ---------- Agreement analysis ----------
    vit_pred = all_predictions["vit"]["pred_label"]
    cnn_pred = all_predictions["cnn"]["pred_label"]
    agree = (vit_pred == cnn_pred).sum()
    disagree = len(vit_pred) - agree
    print(f"\n{'='*60}")
    print(f"Agreement between 2 models: {agree}/{len(vit_pred)} ({agree/len(vit_pred)*100:.1f}%)")
    print(f"Disagree: {disagree} ({disagree/len(vit_pred)*100:.1f}%)")


if __name__ == "__main__":
    main()
