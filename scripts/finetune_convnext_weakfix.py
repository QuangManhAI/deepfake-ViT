#!/usr/bin/env python3
"""Finetune DINOv3 ConvNeXt-Tiny (dinov3_next_cnn) by replicating the v5_weakfix recipe.

Applies the exact method / technique / hyperparameters documented in doc/v5_weakfix
(hoangtuan's v5 DINOv3 ViT-S/16 weak-method repair) to this repo's ConvNeXt-Tiny
backbone. Same data, same samplers, same loss/optimizer/EMA/augmentation.

Stages (mirror doc/v5_weakfix):
  baseline : 3 epochs on train_v5_combined_universal_kaggle_boost.csv (54K balanced),
             plain shuffle -> the "v5-equivalent" for ConvNeXt-Tiny
  v2       : init from baseline, method-balanced sampler,
             2 epochs on train_v5_weakfix.csv (121,884)  -> doc reached 97.20%
  v3       : init from v2, faceswap-focused sampler,
             3 epochs on train_v5_weakfix_v3.csv (129,884, faceswap 12.6K) -> doc 97.88%

Recipe (identical to doc/v5_weakfix/03_finetune):
  LabelSmoothingCrossEntropy(0.05); AdamW {backbone base_lr, head head_lr, wd 0.05/0.005};
  EMA 0.999; CosineAnnealingLR(eta_min=1e-6); clip 1.0; bf16; batch 64; seed 42;
  transforms: BICUBIC 256, HFlip 0.5, ColorJitter(0.2,0.2,0.2,0.05)@0.5,
              GaussianBlur(3,5)@0.3, AdjustSharpness(2.0)@0.3, ToTensor, ImageNet norm.

Run with a pandas-capable venv (hoangtuan venv):
  /workspace/hoangtuan/deepfake-ViT/.venv/bin/python scripts/finetune_convnext_weakfix.py \
      --stage baseline
  ... --stage v2 --init-ckpt outputs/finetune/convnext_baseline.pt
  ... --stage v3 --init-ckpt outputs/finetune/convnext_weakfix_v2.pt
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path("/workspace/quangmanh/deepfake")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_convnext import load_dinov3_convnext
from src.models.classifier_v2 import DinoConvNextClassifier
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.ema import ModelEMA

PRETRAINED = PROJECT_ROOT / "models/dinov3_next_cnn/model-2.safetensors"
TEST_CSV = Path("/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv")
OUT_DIR = PROJECT_ROOT / "outputs" / "finetune"
RES_DIR = PROJECT_ROOT / "experiments" / "results" / "convnext_weakfix"

STAGE_CFG = {
    "baseline": dict(train_csv=PROJECT_ROOT / "data/splits/train_v5_combined_universal_kaggle_boost.csv",
                     val_csv=PROJECT_ROOT / "data/splits/val_v5_combined_universal_kaggle_boost.csv",
                     sampler="none", epochs=3, base_lr=2e-5, head_lr=5e-4, tag="convnext_baseline"),
    "v2": dict(train_csv=PROJECT_ROOT / "data/splits/train_v5_weakfix.csv",
               val_csv=PROJECT_ROOT / "data/splits/val_v5_combined_universal_kaggle_boost.csv",
               sampler="method_balanced", epochs=2, base_lr=2e-5, head_lr=5e-4, tag="convnext_weakfix_v2"),
    "v3": dict(train_csv=PROJECT_ROOT / "data/splits/train_v5_weakfix_v3.csv",
               val_csv=PROJECT_ROOT / "data/splits/val_v5_combined_universal_kaggle_boost.csv",
               sampler="faceswap_focused", epochs=3, base_lr=1.5e-5, head_lr=4e-4, tag="convnext_weakfix_v3"),
}


def get_train_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur((3, 5), (0.1, 2.0))], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(2.0)], p=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].values
        self.labels = self.df["label"].values.astype(np.int64)
        self.methods = (self.df["method"].values if "method" in self.df.columns
                        else np.array(["unknown"] * len(self.df)))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)
        return tensor, self.labels[idx], idx


def method_balanced_weights(df):
    """P(real)=0.5; P(fake method m)=0.5/num_methods. (doc 03 v2)"""
    labels = df["label"].values
    methods = df["method"].values
    num_real = int((labels == 0).sum())
    fake_methods = sorted(set(methods[labels == 1]))
    num_methods = len(fake_methods)
    w = np.ones(len(df), dtype=np.float64)
    w[labels == 0] = 1.0 / num_real
    for m in fake_methods:
        idx = (labels == 1) & (methods == m)
        w[idx] = 1.0 / (num_methods * int(idx.sum()))
    w /= w.sum()
    return w


def faceswap_focused_weights(df, p_real=0.35, p_faceswap=0.35):
    """P(real)=p_real; P(faceswap group)=p_faceswap; rest split uniformly. (doc 03 v3)"""
    labels = df["label"].values
    methods = df["method"].values
    num_real = int((labels == 0).sum())
    p_other = 1.0 - p_real - p_faceswap
    w = np.zeros(len(df), dtype=np.float64)
    w[labels == 0] = p_real / num_real
    is_fs = (labels == 1) & (methods == "faceswap")
    n_fs = int(is_fs.sum())
    w[is_fs] = p_faceswap / n_fs
    other = (labels == 1) & (~is_fs)
    other_methods = sorted(set(methods[other]))
    per_m = p_other / max(1, len(other_methods))
    for m in other_methods:
        idx = (labels == 1) & (methods == m) & (~is_fs)
        w[idx] = per_m / int(idx.sum())
    w /= w.sum()
    return w


def get_parameter_groups(model, base_lr, head_lr, weight_decay=0.05):
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if "head" in name else backbone).append(param)
    return [
        {"params": backbone, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head, "lr": head_lr, "weight_decay": weight_decay * 0.1},
    ]


def train_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema:
            ema.update(model)
        tl += loss.item() * targets.size(0)
        _, preds = torch.max(outputs.float(), 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return tl / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, probs, targets = [], [], []
    for images, tgt, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
        out32 = outputs.float()
        probs.extend(F.softmax(out32, 1)[:, 1].cpu().numpy())
        preds.extend(torch.max(out32, 1)[1].cpu().numpy())
        targets.extend(tgt.cpu().numpy())
    preds, probs, targets = map(np.array, (preds, probs, targets))
    probs = np.nan_to_num(probs, nan=0.5)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(targets, preds)),
        "prec": float(precision_score(targets, preds, zero_division=0)),
        "rec": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5,
        "cm": confusion_matrix(targets, preds, labels=[0, 1]).tolist(),
        "FP": int(fp), "FN": int(fn),
        "preds": preds, "probs": probs, "targets": targets,
    }


def build_model(init_ckpt, device):
    model = DinoConvNextClassifier(load_dinov3_convnext(str(PRETRAINED)))
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Init from {init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
        return model, ck
    print("Init from pretrained backbone (fresh MLP head)")
    return model, None


def _json_safe(v):
    """Recursively convert non-JSON types (Path, numpy scalars) to JSON-safe values."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGE_CFG), required=True)
    ap.add_argument("--init-ckpt", type=str, default="",
                    help="checkpoint to init from (v2/v3 only); baseline ignores")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    cfg = STAGE_CFG[args.stage]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Stage={args.stage} | sampler={cfg['sampler']} | epochs={cfg['epochs']} "
          f"base_lr={cfg['base_lr']} head_lr={cfg['head_lr']} | device={device}")

    df_train = pd.read_csv(cfg["train_csv"])
    df_val = pd.read_csv(cfg["val_csv"])
    df_test = pd.read_csv(TEST_CSV)
    print(f"Train: {len(df_train):,} ({int((df_train.label==0).sum()):,}R/{int((df_train.label==1).sum()):,}F)"
          f" | faceswap={int((df_train.method=='faceswap').sum()):,}"
          f" | Val: {len(df_val):,} | Test: {len(df_test):,}")

    train_ds = DeepfakeDataset(df_train, get_train_transforms(args.img_size))
    val_ds = DeepfakeDataset(df_val, get_eval_transforms(args.img_size))
    test_ds = DeepfakeDataset(df_test, get_eval_transforms(args.img_size))

    num_real = int((df_train.label == 0).sum())
    if cfg["sampler"] == "none":
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        print("[sampler] plain shuffle (baseline; v5 CSV is balanced)")
    else:
        w = (method_balanced_weights(df_train) if cfg["sampler"] == "method_balanced"
             else faceswap_focused_weights(df_train))
        sampler = WeightedRandomSampler(w, num_samples=2 * num_real, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True)
        print(f"[sampler] {cfg['sampler']} -> {len(sampler):,} draws/epoch")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model, ck = build_model(args.init_ckpt if args.stage != "baseline" else "", device)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.1f}M params | trainable {n_trainable/1e6:.1f}M")

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    groups = get_parameter_groups(model, cfg["base_lr"], cfg["head_lr"])
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    ema = ModelEMA(model, decay=0.999)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = OUT_DIR / f"{cfg['tag']}.pt"

    best_val_auc, best_epoch, history = 0.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, ema)
        val_res = evaluate(ema.module, val_loader, device)
        scheduler.step()
        hist = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_acc": val_res["acc"], "val_auc": val_res["auc"], "val_f1": val_res["f1"]}
        history.append(hist)
        print(f"[E{epoch}/{cfg['epochs']}] loss={tr_loss:.4f} acc={tr_acc*100:.2f}% "
              f"| val_acc={val_res['acc']*100:.2f}% auc={val_res['auc']*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_res["auc"] > best_val_auc:
            best_val_auc, best_epoch = val_res["auc"], epoch
            torch.save({"model_state_dict": ema.module.state_dict(), "epoch": epoch,
                        "best_val_auc": best_val_auc, "config": _json_safe({**cfg, **vars(args)}),
                        "history": history, "timestamp": datetime.now().isoformat()},
                       out_ckpt)
    print(f"Best val AUC {best_val_auc*100:.2f}% at epoch {best_epoch}")

    # ---- final test eval with the best EMA ckpt ----
    best = torch.load(out_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_res = evaluate(model, test_loader, device)
    print(f"\nTEST [{cfg['tag']}]: acc={test_res['acc']*100:.2f}% auc={test_res['auc']*100:.2f}% "
          f"prec={test_res['prec']*100:.2f}% rec={test_res['rec']*100:.2f}% "
          f"FP={test_res['FP']} FN={test_res['FN']} cm={test_res['cm']}")

    df_test["pred"] = test_res["preds"]
    df_test["prob"] = test_res["probs"]
    method_rows = []
    for m in df_test["method"].unique():
        sub = df_test[df_test["method"] == m]
        method_rows.append({"Method": m,
                            "Label": "REAL" if sub["label"].iloc[0] == 0 else "FAKE",
                            "Samples": len(sub),
                            "Accuracy": round(float((sub["pred"] == sub["label"]).mean()) * 100, 2),
                            "Mean Fake Probability": round(float(sub["prob"].mean()), 4)})
    df_method = pd.DataFrame(method_rows).sort_values(["Label", "Accuracy"], ascending=[True, True])
    print(df_method.to_string(index=False))

    report = {
        "timestamp": datetime.now().isoformat(),
        "stage": args.stage,
        "config": _json_safe({**cfg, **vars(args)}),
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "history": history,
        "test_metrics": {k: test_res[k] for k in ["acc", "auc", "prec", "rec", "f1", "cm", "FP", "FN"]},
        "method_breakdown": _json_safe(df_method.to_dict("records")),
    }
    rpath = RES_DIR / f"{cfg['tag']}_training_report.json"
    rpath.write_text(json.dumps(report, indent=2))
    df_method.to_csv(RES_DIR / f"{cfg['tag']}_per_method_accuracy.csv", index=False)
    np.savez_compressed(RES_DIR / f"{cfg['tag']}_test_preds.npz",
                        preds=test_res["preds"], probs=test_res["probs"],
                        labels=test_res["targets"], methods=df_test["method"].values)
    print(f"Saved ckpt -> {out_ckpt}\nSaved report -> {rpath}")


if __name__ == "__main__":
    main()
