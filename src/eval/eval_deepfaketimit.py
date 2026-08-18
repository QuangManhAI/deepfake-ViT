"""Đánh giá detection rate của 2 model trên DeepfakeTIMIT (640 video faceswap, toàn bộ FAKE).

- Extract features từ 640 frames (đã trích sẵn bằng ffmpeg)
- Train linear probe trên subset DeepFakeFace (mẫu cân bằng để nhanh)
- Đo detection rate: % frame được flag là fake (prob > 0.5)
"""
import argparse
import csv
import os
import random
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


def load_csv(csv_path):
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


def balanced_subset(items, n_per_class):
    """Lấy n_per_class ảnh mỗi lớp (real/fake)."""
    real = [(p, l) for p, l in items if l == 0]
    fake = [(p, l) for p, l in items if l == 1]
    random.shuffle(real)
    random.shuffle(fake)
    return real[:n_per_class] + fake[:n_per_class]


@torch.no_grad()
def extract_features(model, items, device, batch_size=32, desc="Extracting"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/DeepfakeTIMIT_frames")
    parser.add_argument("--train-csv", default="data/splits/train_insight.csv")
    parser.add_argument("--n-per-class", type=int, default=5000,
                        help="Số ảnh mỗi lớp dùng train probe (để nhanh)")
    parser.add_argument("--vit-model", default="experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    parser.add_argument("--cnn-model", default="experiments/checkpoints/weights/dinov3_next_cnn/model-2.safetensors")
    parser.add_argument("--output", default="experiments/results/benchmark/deepfaketimit_report.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    random.seed(args.seed)

    # ---------- Frames ----------
    frame_files = sorted([f for f in os.listdir(args.frames_dir) if f.endswith('.png')])
    frame_items = [(os.path.join(args.frames_dir, f), 1) for f in frame_files]  # nhãn giả 1 (đều là fake)
    print(f"DeepfakeTIMIT frames: {len(frame_items)} (tất cả đều là FAKE/faceswap)")

    # ---------- Train subset ----------
    train_items = load_csv(args.train_csv)
    train_subset = balanced_subset(train_items, args.n_per_class)
    print(f"Train subset: {len(train_subset)} (mỗi lớp {args.n_per_class})")

    # ---------- Models ----------
    models_config = [
        {"name": "ViT-S/16+", "key": "vit", "loader": load_dinov3,
         "path": args.vit_model, "kwargs": {"img_size": IMG_SIZE}},
        {"name": "ConvNeXt-T", "key": "cnn", "loader": load_dinov3_convnext,
         "path": args.cnn_model, "kwargs": {}},
    ]

    report = {}
    for cfg in models_config:
        print(f"\n{'='*60}")
        print(f"→ {cfg['name']}")
        t0 = time.time()

        model = cfg["loader"](cfg["path"], **cfg["kwargs"])
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # Train features (subset)
        X_train, y_train = extract_features(model, train_subset, device,
                                            args.batch_size, f"  Train ({cfg['key']})")
        # Frame features
        X_frames, _ = extract_features(model, frame_items, device,
                                       args.batch_size, f"  Frames ({cfg['key']})")

        # Linear probe
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
        )
        clf.fit(X_train, y_train)

        # Predict
        prob_fake = clf.predict_proba(X_frames)[:, 1]
        n_detected = int((prob_fake > 0.5).sum())
        det_rate = n_detected / len(prob_fake)

        print(f"  Detection rate (fake flagged đúng): {n_detected}/{len(prob_fake)} = {det_rate*100:.1f}%")
        print(f"  Prob trung bình: {prob_fake.mean():.3f} | median: {np.median(prob_fake):.3f}")
        print(f"  Time: {time.time() - t0:.0f}s")

        report[cfg["key"]] = {
            "name": cfg["name"],
            "detection_rate": float(det_rate),
            "n_detected": n_detected,
            "n_total": len(prob_fake),
            "mean_prob": float(prob_fake.mean()),
            "median_prob": float(np.median(prob_fake)),
            "prob_fake": prob_fake.tolist(),
        }

        # Phân bố theo quality (HQ vs LQ)
        hq_idx = [i for i, f in enumerate(frame_files) if f.startswith('higher')]
        lq_idx = [i for i, f in enumerate(frame_files) if f.startswith('lower')]
        hq_rate = int((prob_fake[hq_idx] > 0.5).sum()) / len(hq_idx)
        lq_rate = int((prob_fake[lq_idx] > 0.5).sum()) / len(lq_idx)
        print(f"    HQ: {hq_rate*100:.1f}% | LQ: {lq_rate*100:.1f}%")
        report[cfg["key"]]["hq_detection_rate"] = float(hq_rate)
        report[cfg["key"]]["lq_detection_rate"] = float(lq_rate)

        del model
        if device in ("cuda", "mps"):
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------- Save ----------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {args.output}")


if __name__ == "__main__":
    main()
