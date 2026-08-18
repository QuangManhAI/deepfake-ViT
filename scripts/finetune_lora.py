"""Fine-tune DINOv3 ViT bằng LoRA (chỉ adapt attention q/v) + classification head.

So với full finetune (lần đầu sụp real-class, AUC 0.45):
  - Backbone DINOv3 giữ nguyên (frozen) → không phá đặc trưng tổng quát
  - Chỉ LoRA (rank thấp trên q/v) + head được train → học method yếu nhưng
    vẫn giữ được khả năng nhận real (đặc biệt real/ffc chưa từng thấy)
  - Class-weighted CE cho imbalance real/fake (dữ liệu mới: fake 6.8k / real 13.7k)

Checkpoint lưu kèm `lora_config` để eval rebuild LoRA đúng khi load.

Chạy:
  .venv/bin/python scripts/finetune_lora.py \
      --train-csv data_train/train.csv --val-csv data_train/val.csv
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
from src.models.lora import apply_lora

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BACKBONE_CKPT = "models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors"

# Aug nhẹ để giảm overfit (lần trước loss→0, val 1.0 = overfit nghiêm trọng)
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    """Backbone (có LoRA) + Linear head. Chỉ head + LoRA có gradient."""

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
    parser = argparse.ArgumentParser(description="LoRA finetune DINOv3 ViT")
    parser.add_argument("--train-csv", default="data_train/train.csv")
    parser.add_argument("--val-csv", default="data_train/val.csv")
    parser.add_argument("--max-train", type=int, default=0, help="0 = dùng hết")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
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
    train_ds = ImageDataset(args.train_csv, TRAIN_TF, max_samples=args.max_train or None)
    val_ds = ImageDataset(args.val_csv, EVAL_TF, max_samples=4000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    n0 = sum(1 for r in train_ds.rows if r[1] == 0)
    n1 = sum(1 for r in train_ds.rows if r[1] == 1)
    w0, w1 = n1 / (n0 + n1), n0 / (n0 + n1)
    print(f"Class weight: real={w0:.3f} fake={w1:.3f} (n_real={n0}, n_fake={n1})", flush=True)

    # ---------- Model ----------
    backbone = load_dinov3(BACKBONE_CKPT, img_size=IMG_SIZE)  # auto-detect gated_mlp
    n_lora = apply_lora(backbone, r=args.lora_rank, alpha=args.lora_alpha)
    model = BackboneClassifier(backbone).to(device)
    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    n_train = sum(p.numel() for p in lora_params) + sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {n_lora} layer wrapped (q,v) | trainable={n_train/1e6:.2f}M / {n_total/1e6:.1f}M", flush=True)

    # ---------- Optimizer ----------
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32).to(device))

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, "vit_lora_finetuned.pt")
    history = {"epochs": [], "train_loss_curve": []}
    lora_config = {"r": args.lora_rank, "alpha": args.lora_alpha, "targets": ["q_proj", "v_proj"]}

    global_batch = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"LoRA Epoch {epoch+1}/{args.epochs}", ncols=100, leave=True)
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
            if n_batch % 10 == 0:
                history["train_loss_curve"].append([global_batch, loss.item()])
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        scheduler.step()

        train_loss = total_loss / n_batch
        val_metrics = evaluate(model, val_loader, device)
        history["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        dt = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} | {dt:.0f}s", flush=True)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_metrics": val_metrics, "lora_config": lora_config}, best_path)
            print(f"  -> saved {best_path} (best val_acc={best_val_acc:.4f})", flush=True)

    report = {
        "model": "ViT-S/16 Plus + LoRA",
        "params_M": round(n_total / 1e6, 1),
        "trainable_M": round(n_train / 1e6, 2),
        "lora_config": lora_config,
        "best_val_acc": best_val_acc,
        "best_ckpt": best_path,
        "history": history,
        "args": vars(args),
    }
    report_path = os.path.join(args.output_dir, "vit_lora_finetune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {report_path}")


if __name__ == "__main__":
    main()
