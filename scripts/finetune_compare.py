"""Fine-tune backbone + classification head cho cả ViT và ConvNeXt.

- Backbone: DINOv3 (ViT hoặc ConvNeXt) — fine-tune với LR thấp
- Head: Linear(feature_dim, 2) — LR cao
- Ghi lại loss curve (train loss theo batch) + val metrics theo epoch
- Output: JSON report chứa loss curve + test metrics

Chạy:
  .venv/bin/python scripts/finetune_compare.py --model-type vit
  .venv/bin/python scripts/finetune_compare.py --model-type cnn
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

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
    def __init__(self, csv_path, transform=None, max_samples=None):
        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self.rows.append((row[0], int(row[1])))
        # Cap số mẫu (giữ cân bằng real/fake nếu có thể)
        if max_samples and len(self.rows) > max_samples:
            real = [r for r in self.rows if r[1] == 0]
            fake = [r for r in self.rows if r[1] == 1]
            half = max_samples // 2
            self.rows = real[:half] + fake[:half]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BackboneClassifier(nn.Module):
    """Backbone (ViT hoặc ConvNeXt) + Linear head."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

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
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT hoặc ConvNeXt")
    parser.add_argument("--model-type", required=True, choices=["vit", "cnn"])
    parser.add_argument("--train-csv", default="data/splits/train_insight.csv")
    parser.add_argument("--val-csv", default="data/splits/val_insight.csv")
    parser.add_argument("--test-csv", default="data/splits/test_insight.csv")
    parser.add_argument("--max-train", type=int, default=8000, help="Giới hạn số ảnh train (để nhanh)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10, help="Ghi loss mỗi N batch")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- Data ----------
    train_ds = ImageDataset(args.train_csv, TRAIN_TF, max_samples=args.max_train)
    val_ds = ImageDataset(args.val_csv, EVAL_TF, max_samples=2000)
    test_ds = ImageDataset(args.test_csv, EVAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    # ---------- Model ----------
    if args.model_type == "vit":
        backbone = load_dinov3("models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors", img_size=IMG_SIZE)
        name = "ViT-S/16 Plus"
    else:
        backbone = load_dinov3_convnext("models/dinov3_next_cnn/model-2.safetensors")
        name = "ConvNeXt-Tiny"
    model = BackboneClassifier(backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {name} — {n_params/1e6:.1f}M params", flush=True)

    # ---------- Optimizer / Scheduler ----------
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Train ----------
    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, f"{args.model_type}_finetuned.pt")
    history = {"epochs": [], "train_loss_curve": []}  # train_loss_curve: list (global_batch, loss)

    global_batch = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{args.epochs}", ncols=100, leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            global_batch += 1
            if n_batch % args.log_every == 0:
                history["train_loss_curve"].append([global_batch, loss.item()])
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        scheduler.step()

        train_loss = total_loss / n_batch
        val_metrics = evaluate(model, val_loader, device)
        history["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        dt = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} | {dt:.0f}s", flush=True)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_metrics": val_metrics}, best_path)

    # ---------- Test ----------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n=== {name} TEST (fine-tuned) ===")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k}: {test_metrics[k]:.4f}")
    print("=================================")

    report = {
        "model": name,
        "model_type": args.model_type,
        "params_M": round(n_params / 1e6, 1),
        "best_val_acc": best_val_acc,
        "test": test_metrics,
        "history": history,
        "args": vars(args),
    }
    report_path = os.path.join(args.output_dir, f"{args.model_type}_finetune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {report_path}")


if __name__ == "__main__":
    main()
