#!/usr/bin/env python3
"""Eval model Plus finetuned (backbone + head) trên val identity-disjoint 6,302 ảnh.

Đọc checkpoint outputs/finetune/plus_v3_best.pt (state_dict backbone.* + head.*),
chạy full inference trên CSV val → báo cáo tổng + real/fake + per-method (detection)
+ faceswap + per-domain. Không chạy khi đang train (MPS + RAM).

So sánh tham chiếu (baseline pretrained Plus, linear probe, official split):
  identity_disjoint_v3_vit.json — acc 0.9519, AUC 0.9746, real_acc 0.871, fake_det 0.955
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageListDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, method, domain = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, method, domain


class BackboneClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    out = []  # (prob1, pred, y, method, domain)
    for x, y, m, d in tqdm(loader, desc="Infer", ncols=90):
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        for p_, pr_, y_, m_, d_ in zip(logits.argmax(1).cpu().numpy(),
                                       prob.cpu().numpy(), y, m, d):
            out.append((pr_, p_, int(y_), m_, d_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/finetune/plus_v3_best.pt")
    ap.add_argument("--csv", default="data/splits/finetune_plus_val.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | ckpt: {args.ckpt}")

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], int(r["label"]), r["method"], r["domain"]))
    print(f"Eval rows: {len(rows)}")

    backbone = load_dinov3("models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors",
                           img_size=IMG_SIZE)
    model = BackboneClassifier(backbone).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded ckpt epoch {ckpt.get('epoch')} | val_acc(cuối) {ckpt.get('val_metrics', {}).get('accuracy')}")

    loader = DataLoader(ImageListDataset(rows, EVAL_TF), batch_size=args.batch_size,
                        shuffle=False, num_workers=2)
    res = infer(model, loader, device)
    prob = np.array([r[0] for r in res])
    pred = np.array([r[1] for r in res])
    y = np.array([r[2] for r in res])
    methods = [r[3] for r in res]
    domains = [r[4] for r in res]

    m = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    real_acc = float((pred[y == 0] == 0).mean())
    fake_det = float((pred[y == 1] == 1).mean())
    print(f"\n=== FINETUNED PLUS — VAL IDENTITY-DISJOINT ===")
    print(f"  Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
          f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
    print(f"  Real acc={real_acc:.4f} | Fake detection={fake_det:.4f}")
    print(f"  Confusion: {m['confusion_matrix']}")

    # per-method detection (fake) + real
    print(f"\n  {'method':22s}{'dom':8s}{'n':>6s}{'det':>8s}")
    per = {}
    import collections
    by_key = collections.defaultdict(list)
    for p_, y_, m_, d_ in zip(pred, y, methods, domains):
        by_key[(m_, d_)].append((p_, y_))
    for (mth, d), items in sorted(by_key.items()):
        n = len(items)
        if mth == "FaceForensics++ Real":
            continue
        det = float(sum(1 for p_, y_ in items if (p_ == y_)) / n)
        per[f"{mth}/{d}"] = {"n": n, "detection": round(det, 4)}
        print(f"  {mth:22s}{d:8s}{n:6d}{det:8.3f}")

    # faceswap riêng (mục tiêu recipe)
    fs_mask = np.array([m_ == "faceswap" for m_ in methods])
    if fs_mask.sum():
        print(f"\n  Faceswap detection: {(pred[fs_mask] == 1).mean():.4f} "
              f"({int(fs_mask.sum())} ảnh)")

    # per-domain
    print(f"\n  Per-domain:")
    dom = {}
    for d in sorted(set(domains)):
        mask = np.array([x == d for x in domains])
        if int(mask.sum()) == 0:
            continue
        acc = accuracy_score(y[mask], pred[mask])
        dom[d] = {"n": int(mask.sum()), "acc": round(acc, 4)}
        print(f"    {d:20s} {int(mask.sum()):6d}  acc={acc:.4f}")

    out = {"ckpt": args.ckpt, "n": len(rows), "metrics": m,
           "real_acc": real_acc, "fake_det": fake_det, "per_method": per,
           "per_domain": dom}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu: {args.output}")


if __name__ == "__main__":
    main()
