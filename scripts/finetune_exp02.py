#!/usr/bin/env python3
"""Second-stage finetune of Tuấn's exp02 full-finetune on the weak-method dataset.

Loads dinov3_vit_exp02_best.pt (ViT-S/16 Plus + EnhancedDinoViTClassifier MLP head),
finetunes 3 epochs on data/finetune_exp02/train.csv (targeted weak methods, leak-free),
saves outputs/finetune/exp02_weak_methods.pt.

GPU note: shares the RTX 3060 with the user's own training — small model (29M), bf16,
batch 32 -> a few GB. Never killed; run is memory-conscious.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path("/workspace/quangmanh/deepfake")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_vit import DinoViT
from src.training.ema import ModelEMA
from src.training.losses import LabelSmoothingCrossEntropy

# ----------------------------------------------------------------------------
IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGENET_MEAN, IMAGENET_STD = MEAN, STD
EXP02_CKPT = Path("/workspace/hoangtuan/deepfake-ViT/experiments/checkpoints/dinov3_vit_exp02_best.pt")
TRAIN_CSV = PROJECT_ROOT / "data/finetune_exp02/train.csv"
OUT_CKPT = PROJECT_ROOT / "outputs/finetune/exp02_weak_methods.pt"


class EnhancedDinoViTClassifier(nn.Module):
    """Same head as hoangtuan train_exp02.py: LayerNorm, Dropout, 2-layer GELU MLP."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2, hidden_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, verify=True):
        self.rows = []
        skipped = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                path = r["path"]
                if verify:
                    try:
                        with Image.open(path) as im:
                            im.verify()
                    except Exception as e:
                        skipped.append((path, repr(e)))
                        continue
                self.rows.append((path, int(r["label"]), r.get("method", "")))
        if skipped:
            print(f"[ImageDataset] skipped {len(skipped)} unreadable images, e.g. {skipped[0]}")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, method = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_llrd_optimizer(model, base_lr=5e-6, head_lr=5e-4, decay_rate=0.85, weight_decay=0.05):
    """Layer-wise Learning Rate Decay — same as hoangtuan train_exp02.py."""
    groups = []
    num_layers = len(model.backbone.layer)
    groups.append({"params": [p for p in model.backbone.embeddings.parameters()], "lr": base_lr})
    for i, blk in enumerate(model.backbone.layer):
        lr = base_lr * (decay_rate ** (num_layers - 1 - i))
        groups.append({"params": blk.parameters(), "lr": lr})
    groups.append({"params": model.backbone.norm.parameters(), "lr": base_lr})
    groups.append({"params": model.head.parameters(), "lr": head_lr})
    return torch.optim.AdamW(groups, lr=head_lr, weight_decay=weight_decay)


def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses, preds, labels, probs = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            probs.extend(p[:, 1].cpu().tolist())
            if criterion is not None:
                losses.append(criterion(logits, y).item())
    preds, labels, probs = np.array(preds), np.array(labels), np.array(probs)
    acc = float((preds == labels).mean())
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else 0.0
    return {"accuracy": acc, "roc_auc": auc, "loss": float(np.mean(losses)) if losses else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--val-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-csv", type=str, default=str(TRAIN_CSV))
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze all backbone params; train only the MLP head (0 backbone LR)")
    ap.add_argument("--tag", type=str, default="exp02_weak_methods",
                    help="output checkpoint name (outputs/finetune/<tag>.pt)")
    ap.add_argument("--no-balanced", action="store_true",
                    help="disable WeightedRandomSampler (use raw class ratio)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model from exp02 checkpoint ----
    ck = torch.load(str(EXP02_CKPT), map_location="cpu", weights_only=False)
    backbone = DinoViT(img_size=IMG_SIZE, gated_mlp=True)  # ViT-S/16 Plus, auto gated MLP
    model = EnhancedDinoViTClassifier(backbone)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"])
    assert not missing and not unexpected, f"exp02 load: missing={missing} unexpected={unexpected}"
    model.to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print("backbone FROZEN (head-only finetune)")
    print(f"Loaded {EXP02_CKPT.name} (exp02 epoch {ck['epoch']}) | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # ---- data split (balanced val for EMA tracking) ----
    full = ImageDataset(args.train_csv)
    real_idx = [i for i, r in enumerate(full.rows) if r[1] == 0]
    fake_idx = [i for i, r in enumerate(full.rows) if r[1] == 1]
    rng = random.Random(args.seed)
    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)
    n_val_each = args.val_size // 2
    val_idx = set(real_idx[:n_val_each]) | set(fake_idx[:n_val_each])
    train_idx = [i for i in range(len(full)) if i not in val_idx]

    class Subset(Dataset):
        def __init__(self, base, idx, transform):
            self.base, self.idx, self.transform = base, idx, transform
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            x, y = self.base[self.idx[i]]
            return self.transform(x), y

    train_tf, eval_tf = get_train_transform(), get_eval_transform()
    train_ds = Subset(full, train_idx, train_tf)
    val_ds = Subset(full, sorted(val_idx), eval_tf)
    if args.no_balanced:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        # WeightedRandomSampler: real and fake each sum to equal probability per epoch
        train_labels = np.array([full.rows[i][1] for i in train_idx])
        n_real, n_fake = int((train_labels == 0).sum()), int((train_labels == 1).sum())
        weights = np.where(train_labels == 0, 1.0 / n_real, 1.0 / n_fake)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=2 * n_real if n_real <= n_fake else 2 * n_fake,
            replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"[balanced] real {n_real} / fake {n_fake} -> sampler {len(sampler)}/epoch")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=0.05)
    else:
        optimizer = build_llrd_optimizer(model, base_lr=args.lr_backbone, head_lr=args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    ema = ModelEMA(model, decay=0.999)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_auc, history = 0.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n, correct, seen = 0.0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item(); n += 1
            correct += (logits.argmax(1) == y).sum().item(); seen += y.size(0)
        scheduler.step()
        val_res = evaluate(ema.module, val_loader, device, criterion=criterion)
        hist = {"epoch": epoch + 1, "train_loss": total_loss / n, "train_acc": correct / seen,
                "val_acc": val_res["accuracy"], "val_auc": val_res["roc_auc"]}
        history.append(hist)
        print(f"Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.0f}s) | "
              f"loss {hist['train_loss']:.4f} acc {hist['train_acc']*100:.1f}% | "
              f"val acc {hist['val_acc']*100:.1f}% auc {hist['val_auc']:.4f}", flush=True)
        if val_res["roc_auc"] > best_val_auc:
            best_val_auc = val_res["roc_auc"]

    # ---- save EMA model (like exp02) ----
    out_ckpt = Path(PROJECT_ROOT / "outputs/finetune") / f"{args.tag}.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": ema.module.state_dict(),
        "val_metrics": {"accuracy": history[-1]["val_acc"], "roc_auc": history[-1]["val_auc"]},
        "history": history,
        "config": {"base": "dinov3_vit_exp02_best", "train_csv": args.train_csv, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
                   "freeze_backbone": args.freeze_backbone},
    }, str(out_ckpt))
    print(f"Saved -> {out_ckpt}  (best val auc {best_val_auc:.4f})")


if __name__ == "__main__":
    main()
