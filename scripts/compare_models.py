"""So sánh DINOv3 ViT-Small vs DINOv3 ConvNeXt-Tiny trên cùng điều kiện.

Điều kiện công bằng:
  - Cùng dataset (DeepFakeFace splits via CSV)
  - Cùng preprocessing (resize 256×256, ImageNet normalization)
  - Cùng protocol đánh giá (linear probe: StandardScaler + LogisticRegression balanced)
  - Cùng seed, cùng batch size
  - Cả 2 đều là frozen feature extractor (không fine-tune)

Output:
  - Bảng so sánh metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Báo cáo JSON chi tiết
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
    """Đọc CSV (path,label) → list[(path, label)]."""
    import csv

    items = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                p, lbl = row[0], int(row[1])
                if os.path.exists(p):
                    items.append((p, lbl))
    return items


@torch.no_grad()
def extract_features(model, items, device, batch_size: int = 32, desc: str = "Extracting"):
    """Trích xuất feature vectors từ backbone (frozen)."""
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


def evaluate_linear_probe(X_train, y_train, X_test, y_test, seed: int = 42):
    """Train linear probe trên train, đánh giá trên test."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


def print_model_result(name: str, n_params: float, feat_dim: int, elapsed: float, m: dict):
    """In kết quả của 1 model."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  Params: {n_params:.1f}M | Feature dim: {feat_dim} | Time: {elapsed:.0f}s")
    print(f"{'='*65}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}   (fake = positive)")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"  CM       : TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  TP={m['tp']}")


def print_comparison_table(vit_result: dict, cnn_result: dict):
    """In bảng so sánh side-by-side."""
    print(f"\n{'='*70}")
    print(f"{'METRIC':<15} {'ViT-S/16':>12} {'ConvNeXt-T':>12} {'Δ (CNN-ViT)':>14}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*14}")

    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]:
        vit_val = vit_result[key]
        cnn_val = cnn_result[key]
        delta = cnn_val - vit_val
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "—"
        print(f"  {label:<13} {vit_val:>10.4f}  {cnn_val:>10.4f}  {delta:+10.4f} {direction}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh DINOv3 ViT vs ConvNeXt (cùng điều kiện)"
    )
    parser.add_argument("--train-csv", default="data/splits/train.csv",
                        help="CSV file cho tập train")
    parser.add_argument("--test-csv", default="data/splits/test.csv",
                        help="CSV file cho tập test")
    parser.add_argument("--vit-model", default="models/dinov3_small/model.safetensors",
                        help="Path đến DINOv3 ViT safetensors")
    parser.add_argument("--cnn-model", default="models/dinov3_next_cnn/model-2.safetensors",
                        help="Path đến DINOv3 ConvNeXt safetensors")
    parser.add_argument("--output", default="outputs/results/comparison_report.json",
                        help="Path output JSON report")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---------- Device ----------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ---------- Data ----------
    print(f"\nLoad data:")
    train_items = load_csv(args.train_csv)
    test_items = load_csv(args.test_csv)
    print(f"  Train: {len(train_items)} ảnh (real={sum(1 for _,l in train_items if l==0)}, "
          f"fake={sum(1 for _,l in train_items if l==1)})")
    print(f"  Test : {len(test_items)} ảnh (real={sum(1 for _,l in test_items if l==0)}, "
          f"fake={sum(1 for _,l in test_items if l==1)})")

    # ---------- Models ----------
    models_config = [
        {
            "name": "DINOv3 ViT-S/16",
            "key": "vit",
            "path": args.vit_model,
            "loader": load_dinov3,
            "loader_kwargs": {"img_size": IMG_SIZE},
        },
        {
            "name": "DINOv3 ConvNeXt-Tiny",
            "key": "cnn",
            "path": args.cnn_model,
            "loader": load_dinov3_convnext,
            "loader_kwargs": {},
        },
    ]

    report = {
        "config": {
            "img_size": IMG_SIZE,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": device,
        },
        "data": {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "train_samples": len(train_items),
            "test_samples": len(test_items),
        },
        "models": {},
    }

    metrics = {}

    for cfg in models_config:
        print(f"\n{'─'*65}")
        print(f"→ {cfg['name']}")

        # Load
        print(f"  Load: {cfg['path']}")
        model = cfg["loader"](cfg["path"], **cfg["loader_kwargs"])
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        feat_dim = model.embed_dim
        print(f"  Params: {n_params:.1f}M | Feature dim: {feat_dim}")

        # Extract features
        t0 = time.time()
        X_train, y_train = extract_features(model, train_items, device, args.batch_size,
                                            desc=f"  Train features ({cfg['key']})")
        X_test, y_test = extract_features(model, test_items, device, args.batch_size,
                                          desc=f"  Test features  ({cfg['key']})")
        elapsed = time.time() - t0

        # Evaluate
        m = evaluate_linear_probe(X_train, y_train, X_test, y_test, args.seed)
        metrics[cfg["key"]] = m

        print_model_result(cfg["name"], n_params, feat_dim, elapsed, m)

        report["models"][cfg["key"]] = {
            "name": cfg["name"],
            "params_M": round(n_params, 1),
            "feature_dim": feat_dim,
            "extract_time_s": round(elapsed, 1),
            "metrics": m,
        }

        # Giải phóng model khỏi GPU
        del model
        if device in ("cuda", "mps"):
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

    # ---------- Comparison Table ----------
    print_comparison_table(metrics["vit"], metrics["cnn"])

    # ---------- Save Report ----------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo: {args.output}")


if __name__ == "__main__":
    main()
