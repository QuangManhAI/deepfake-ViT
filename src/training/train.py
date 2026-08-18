"""Fine-tune DINOv3 backbone + classification head trên dữ liệu cân bằng.

Thiết kế:
  - Backbone: DINOv3 ViT-S/16 (load từ model.safetensors) — được fine-tune (không đông cứng)
  - Head: Linear(384, 2) trên CLS token
  - Optimizer: AdamW 2 nhóm LR (backbone thấp 1e-5, head cao 1e-3)
  - Scheduler: CosineAnnealingLR
  - Lưu checkpoint val acc tốt nhất; sau cùng đánh giá trên test

Cách chạy:
  .venv/bin/python src/training/train.py --train-csv data/splits/train_insight.csv \
      --val-csv data/splits/val_insight.csv --test-csv data/splits/test_insight.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self.rows.append((row[0], int(row[1])))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DinoViTClassifier(nn.Module):
    """Backbone DINOv3 + Linear head trên CLS token."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_y.extend(y.tolist())
        all_pred.extend(logits.argmax(1).tolist())
        all_prob.extend(probs[:, 1].tolist())
    y = np.array(all_y)
    pred = np.array(all_pred)
    prob = np.array(all_prob)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model", default="experiments/checkpoints/weights/model.safetensors")
    parser.add_argument("--output-dir", default="experiments/results/checkpoints")
    parser.add_argument("--report", default="experiments/results/finetune_report.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="mixed precision (bfloat16)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    device_type = "cuda" if device == "cuda" else "cpu"
    print(f"Device: {device}", flush=True)

    set_seed(args.seed)

    # ---------- Data ----------
    train_ds = ImageDataset(args.train_csv, TRAIN_TF)
    val_ds = ImageDataset(args.val_csv, EVAL_TF)
    test_ds = ImageDataset(args.test_csv, EVAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    # ---------- Model ----------
    backbone = load_dinov3(args.model, img_size=IMG_SIZE)
    model = DinoViTClassifier(backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M) — sẽ fine-tune CẢ backbone", flush=True)

    # ---------- Optimizer / Scheduler ----------
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---------- Train ----------
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "dinov3_finetuned.pt")
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    unit="batch", ncols=100, leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.amp):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            if n_batch % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = total_loss / n_batch
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        dt = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} | {dt:.0f}s",
              flush=True)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_metrics": val_metrics}, best_path)
            print(f"  → Lưu checkpoint tốt nhất (val_acc={best_val_acc:.4f})", flush=True)

    # ---------- Test ----------
    print(f"\nLoad checkpoint tốt nhất và đánh giá TEST...")
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    print("\n================= KẾT QUẢ TEST (sau fine-tune) =================")
    print(f"  accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall   : {test_metrics['recall']:.4f}")
    print(f"  f1       : {test_metrics['f1']:.4f}")
    print(f"  roc_auc  : {test_metrics['roc_auc']:.4f}")
    cm = test_metrics["confusion_matrix"]
    print(f"  CM [TN FP; FN TP]: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print("===================================================================")

    report = {"best_val_acc": best_val_acc, "test": test_metrics,
              "history": history, "args": vars(args)}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {best_path} + {args.report}")


if __name__ == "__main__":
    main()
