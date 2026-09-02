#!/usr/bin/env python3
"""Finetune DINOv3 ViT-S/16 Plus trên bộ 129K (recipe v5_weakfix_v3).

Init      : models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors (Plus, 28.7M)
Train     : data/splits/finetune_plus_train.csv (123,582 = 29,557R/94,025F; faceswap 11,953)
Val       : data/splits/finetune_plus_val.csv (6,302, identity-disjoint)
Sampler   : faceswap-focused — P(faceswap)=0.35, P(real)=0.35, P(method khác)=0.30
            chia đều các fake-method còn lại → faceswap được nhìn ~1.7 vòng/epoch.
Số mẫu/epoch: 2 × num_real ≈ 59K
Loss      : CrossEntropy(label_smoothing=0.05)
Optimizer : AdamW (backbone 1.5e-5 / head 4e-4, weight_decay 0.05), EMA 0.999
Scheduler : CosineAnnealingLR (T_max=epochs)
Epochs    : 3, batch 32 (MPS; server dùng 64), fp32 trên MPS.

Output    : outputs/finetune/plus_v3_best.pt (EMA best theo val acc) + report JSON.

RAM-conscious: num_workers nhỏ, không pin_memory, không load toàn bộ ảnh vào RAM.
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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext
from src.models.classifier_v2 import DinoConvNextClassifier
from src.utils.seeding import set_seed

CONVNEXT_BACKBONE = "models/dinov3_next_cnn/model-2.safetensors"
CONVNEXT_CKPT = "models/convnext_weakfix_v3/convnext_weakfix_v3.pt"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUS_CKPT = os.path.join(ROOT, "models", "dinov3-vits16plus-pretrain-lvd1689m", "model-3.safetensors")

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(2.0, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_rows(csv_path, max_rows=None):
    """Đọc CSV → [(abs_path, label, method)]."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["path"], int(r["label"]), r["method"]))
    if max_rows and len(rows) > max_rows:
        # cap có pha trộn: chia đều real / faceswap / other (dùng cho smoke test)
        buckets = {"real": [], "faceswap": [], "other": []}
        for r_ in rows:
            b = "real" if r_[1] == 0 else ("faceswap" if r_[2] == "faceswap" else "other")
            buckets[b].append(r_)
        per = max(1, max_rows // 3)
        out = []
        for b in ("real", "faceswap", "other"):
            out.extend(buckets[b][:per])
        return out
    return rows


class ImageListDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform
        self.n_skip = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        n = len(self.rows)
        # robust: vài ảnh nguồn hỏng (PIL.UnidentifiedImageError) — nhảy tới ảnh kế hợp lệ
        for _ in range(n):
            path, label, _ = self.rows[i]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                self.n_skip += 1
                print(f"[skip] {path}", file=sys.stderr, flush=True)
                i = (i + 1) % n
                continue
            if self.transform:
                img = self.transform(img)
            return img, label
        raise RuntimeError("toàn bộ dataset không decode được — kiểm tra lại dữ liệu")


class WeakFamilyBoostedSampler(Sampler):
    """A1 — bucket boost mở rộng từ 'chỉ faceswap' sang cả họ swap/lip yếu.

    Diagnose (test balanced + val): Plus yếu nhất ở deepfake_faceswap (77%),
    facedancer, wav2lip, sadtalker, fsgan — các method pool lớn (2.9K–7.7K ảnh)
    đang bị bucket 'other' chia-đều-theo-method bỏ đói (~0.06–0.15×/ảnh/epoch).
    Faceswap (11.9K ảnh) lại được bơm 1.73×/ảnh/epoch → overfit cục bộ faceswap.

    Thiết kế (3 bucket, tổng/epoch = 2 × num_real = 59,114):
      P(real)  = 0.35 → 20,690 slots (real 0.70×/ảnh, giữ nguyên A0)
      P(boost) = 0.45 → 26,601 slots cho họ yếu
                   {faceswap, deepfake_faceswap, wav2lip, sadtalker,
                    fsgan, facedancer, inswap, mobileswap}
                   chia theo pool × weight (faceswap ×2.0):
                   faceswap ~1.0×/ảnh, các method yếu ~0.52×/ảnh (tăng ~9×)
      P(other) = 0.20 → 11,823 slots cho các fake-method còn lại, chia pool-prop
                 (mọi method ~0.22×/ảnh, không method nào bị bỏ đói)

    Deterministic (rng seed+epoch) để resume mid-epoch như sampler cũ.
    """

    FAMILY_WEIGHTS = {
        "faceswap": 2.0,
        "deepfake_faceswap": 1.0,
        "wav2lip": 1.0,
        "sadtalker": 1.0,
        "fsgan": 1.0,
        "facedancer": 1.0,
        "inswap": 1.0,
        "mobileswap": 1.0,
    }
    P_REAL, P_BOOST = 0.35, 0.45

    def __init__(self, rows, seed=42):
        self.real_idx = []
        self.family_by_method = {m: [] for m in self.FAMILY_WEIGHTS}
        self.other_by_method = {}          # fake, không thuộc family
        for i, (_, label, method) in enumerate(rows):
            if label == 0:
                self.real_idx.append(i)
            elif method in self.FAMILY_WEIGHTS:
                self.family_by_method[method].append(i)
            else:
                self.other_by_method.setdefault(method, []).append(i)
        self.n_real = len(self.real_idx)
        self.family_methods = sorted(self.family_by_method)
        self.other_methods = sorted(self.other_by_method.keys())
        self.seed = seed
        self.epoch = 0
        self._alloc = self._allocate()
        # lưu để in báo cáo exposure
        self.exposure = self._exposure_map()

    def __len__(self):
        return 2 * self.n_real

    @staticmethod
    def _largest_remainder(slots, weights):
        """Chia 'slots' nguyên cho các pool theo weight, tổng khớp chính xác."""
        if not weights:
            return []
        total = sum(weights)
        raw = [slots * w / total for w in weights]
        out = [int(x) for x in raw]
        rem = slots - sum(out)
        # phần dư trao theo fractional part lớn nhất (deterministic)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
        for i in order[:rem]:
            out[i] += 1
        return out

    def _allocate(self):
        total = 2 * self.n_real
        n_real = int(round(self.P_REAL * total))
        n_boost = int(round(self.P_BOOST * total))
        n_other = total - n_real - n_boost
        fam_pools = [len(self.family_by_method[m]) for m in self.family_methods]
        fam_weights = [p * self.FAMILY_WEIGHTS[m]
                       for p, m in zip(fam_pools, self.family_methods)]
        fam_counts = self._largest_remainder(n_boost, fam_weights)
        other_pools = [len(self.other_by_method[m]) for m in self.other_methods]
        other_counts = self._largest_remainder(n_other, other_pools)
        return {
            "n_real": n_real,
            "family": dict(zip(self.family_methods, fam_counts)),
            "other": dict(zip(self.other_methods, other_counts)),
        }

    def _exposure_map(self):
        """x/ảnh/epoch cho từng method (để in, debug overfit)."""
        em = {}
        for m, c in self._alloc["family"].items():
            p = len(self.family_by_method[m])
            em[m] = (c / p) if p else 0.0
        for m, c in self._alloc["other"].items():
            p = len(self.other_by_method[m])
            em[m] = (c / p) if p else 0.0
        return em

    def order(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        total = 2 * self.n_real
        n_real = self._alloc["n_real"]
        order = []
        # real: ưu tiên không lặp (giữ như A0)
        if self.real_idx:
            re = np.asarray(self.real_idx, dtype=np.int64)
            k = min(n_real, len(re))
            order.extend(rng.choice(re, k, replace=False).tolist())
            if n_real > len(re):
                order.extend(rng.choice(re, n_real - len(re), replace=True).tolist())
        # family: rút có lặp theo số slot đã allocate
        for m, cnt in self._alloc["family"].items():
            pool = np.asarray(self.family_by_method[m], dtype=np.int64)
            if cnt and len(pool):
                order.extend(rng.choice(pool, cnt, replace=True).tolist())
        # other: rút có lặp theo số slot đã allocate
        for m, cnt in self._alloc["other"].items():
            pool = np.asarray(self.other_by_method[m], dtype=np.int64)
            if cnt and len(pool):
                order.extend(rng.choice(pool, cnt, replace=True).tolist())
        # tổng sau cùng đúng bằng 2*num_real → shuffle
        assert len(order) == total, f"order len {len(order)} != {total}"
        rng.shuffle(order)
        return order

    def __iter__(self):
        return iter(self.order(self.epoch))


class FaceswapFocusedSampler(Sampler):
    """P(faceswap)=0.35, P(real)=0.35, P(method khác)=0.30 chia đều các fake-method còn lại.

    Mỗi epoch: tổng 2 × num_real; các nhóm "other" rút đều theo method (thay vì theo ảnh)
    để method yếu được nhìn ngang nhau — đúng tinh thần recipe v5_weakfix_v3.
    """

    def __init__(self, rows, p_faceswap=0.35, p_real=0.35, seed=42):
        self.faceswap_idx, self.real_idx = [], []
        self.other_by_method = {}          # method -> [idx...] (fake, không phải faceswap)
        for i, (_, label, method) in enumerate(rows):
            if label == 0:
                self.real_idx.append(i)
            elif method == "faceswap":
                self.faceswap_idx.append(i)
            else:
                self.other_by_method.setdefault(method, []).append(i)
        self.n_real = len(self.real_idx)
        self.seed = seed
        self.epoch = 0
        self.other_methods = sorted(self.other_by_method.keys())

    def __len__(self):
        return 2 * self.n_real

    def order(self, epoch):
        """Sinh deterministic thứ tự indices của 1 epoch (để resume mid-epoch)."""
        rng = np.random.default_rng(self.seed + epoch)
        total = 2 * self.n_real
        n_fs = int(round(0.35 * total))
        n_re = int(round(0.35 * total))
        n_ot = total - n_fs - n_re

        order = []
        # faceswap: rút có lặp (n_fs > pool)
        if self.faceswap_idx:
            fs = np.asarray(self.faceswap_idx, dtype=np.int64)
            order.extend(rng.choice(fs, n_fs, replace=True).tolist())
        # real: ưu tiên không lặp
        if self.real_idx:
            re = np.asarray(self.real_idx, dtype=np.int64)
            k = min(n_re, len(re))
            order.extend(rng.choice(re, k, replace=False).tolist())
            if n_re > len(re):
                order.extend(rng.choice(re, n_re - len(re), replace=True).tolist())
        # other: chia đều theo method (mỗi method rút n_ot/n_methods, có lặp)
        n_m = len(self.other_methods)
        if n_m > 0:
            base, rem = divmod(n_ot, n_m)
            for j, m in enumerate(self.other_methods):
                pool = np.asarray(self.other_by_method[m], dtype=np.int64)
                k = base + (1 if j < rem else 0)
                if k > 0:
                    order.extend(rng.choice(pool, k, replace=True).tolist())
        rng.shuffle(order)
        return order

    def __iter__(self):
        return iter(self.order(self.epoch))


class OrderSampler(Sampler):
    """Iter qua một danh sách indices có sẵn (dùng khi resume mid-epoch)."""

    def __init__(self, order):
        self.order = order

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order)


class BackboneClassifier(nn.Module):
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
    y = np.array(all_y); pred = np.array(all_pred); prob = np.array(all_prob)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def update_ema(ema, model, decay=0.999):
    for k, v in model.state_dict().items():
        ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/splits/finetune_plus_train.csv")
    ap.add_argument("--val-csv", default="data/splits/finetune_plus_val.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1.5e-5)
    ap.add_argument("--lr-head", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=0, help="giới hạn số dòng train (smoke)")
    ap.add_argument("--limit-iters", type=int, default=0, help="dừng sau N batch/epoch (smoke)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-val", action="store_true", help="bỏ val eval (smoke test)")
    ap.add_argument("--sampler", choices=["faceswap", "weak_family"], default="faceswap",
                    help="faceswap: A0 (P(faceswap)=0.35). weak_family: A1 boost họ swap/lip yếu")
    ap.add_argument("--kd-teacher", type=str, default="",
                    help="path ConvNeXt teacher cho KD (A2). rỗng = không KD")
    ap.add_argument("--kd-lambda", type=float, default=0.5,
                    help="hệ số pha loss KD (A2): loss = CE + λ·KD")
    ap.add_argument("--kd-temp", type=float, default=4.0,
                    help="temperature cho soft-target KD (A2)")
    ap.add_argument("--tag", type=str, default="plus_v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-every", type=int, default=800, help="lưu resume ckpt mỗi N global steps (~13 phút)")
    ap.add_argument("--resume", type=str, default="", help="path resume ckpt để tiếp tục run bị ngắt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | torch {torch.__version__}", flush=True)
    set_seed(args.seed)

    train_rows = load_rows(args.train_csv, max_rows=args.max_train or None)
    val_rows = load_rows(args.val_csv)
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}", flush=True)

    train_ds = ImageListDataset(train_rows, TRAIN_TF)
    val_ds = ImageListDataset(val_rows, EVAL_TF)
    if args.sampler == "weak_family":
        sampler = WeakFamilyBoostedSampler(train_rows, seed=args.seed)
        # in exposure theo method để debug overfit cục bộ
        top = sorted(sampler.exposure.items(), key=lambda kv: -kv[1])[:14]
        exp_str = "  ".join(f"{m}:{r:.2f}x" for m, r in top)
        print(f"Sampler A1 weak_family: total/epoch={len(sampler)} real={len(sampler.real_idx)} "
              f"family_methods={len(sampler.family_methods)} other_methods={len(sampler.other_methods)}", flush=True)
        print(f"  exposure (x/img/epoch, top): {exp_str}", flush=True)
    else:
        sampler = FaceswapFocusedSampler(train_rows, seed=args.seed)
        print(f"Sampler A0 faceswap: total/epoch={len(sampler)}  faceswap={len(sampler.faceswap_idx)} "
              f"real={len(sampler.real_idx)} other_methods={len(sampler.other_methods)}", flush=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---------- Model: Plus backbone + head ----------
    backbone = load_dinov3(PLUS_CKPT, img_size=IMG_SIZE)
    model = BackboneClassifier(backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DINOv3 ViT-S/16 Plus — {n_params/1e6:.1f}M params", flush=True)

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---------- Optimizer / Loss / Scheduler ----------
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---------- KD teacher (Stage A2): ConvNeXt frozen ----------
    kd_teacher = None
    if args.kd_teacher:
        t_ck = os.path.join(ROOT, args.kd_teacher) if not os.path.isabs(args.kd_teacher) else args.kd_teacher
        t_bk = os.path.join(ROOT, CONVNEXT_BACKBONE)
        kd_teacher = DinoConvNextClassifier(load_dinov3_convnext(t_bk)).to(device)
        ck = torch.load(t_ck, map_location=device, weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck["state_dict"]
        missing, unexpected = kd_teacher.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"teacher mismatch: {len(missing)}/{len(unexpected)}"
        kd_teacher.eval()
        for p in kd_teacher.parameters():
            p.requires_grad = False
        print(f"KD teacher: ConvNeXt ({sum(p.numel() for p in kd_teacher.parameters())/1e6:.1f}M) "
              f"frozen | λ={args.kd_lambda} T={args.kd_temp}", flush=True)

    out_dir = os.path.join(ROOT, "outputs", "finetune")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{args.tag}_best.pt")
    final_path = os.path.join(out_dir, f"{args.tag}_final.pt")
    history = {"train_loss_curve": [], "epochs": []}

    best_val_acc, global_step = 0.0, 0
    start_epoch, resume_skip = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        ema = ck["ema"]
        start_epoch = ck["epoch"]
        global_step = ck["global_step"]
        best_val_acc = ck.get("best_val_acc", 0.0)
        sampler.epoch = start_epoch
        # global_step đếm BATCH, len(sampler) đếm SAMPLE. Skip đúng = số sample
        # đã tiêu trong epoch hiện tại = (số batch vào epoch) × batch_size.
        batches_per_epoch = (len(sampler) + args.batch_size - 1) // args.batch_size
        resume_skip = (global_step % batches_per_epoch) * args.batch_size
        assert resume_skip < len(sampler), f"resume_skip {resume_skip} >= len(sampler) {len(sampler)}"
        print(f"✓ Resume: epoch {start_epoch+1}/{args.epochs}, "
              f"step {resume_skip}/{len(sampler)} (global {global_step})", flush=True)
    resume_path = os.path.join(out_dir, f"{args.tag}_resume.pt")

    for epoch in range(start_epoch, args.epochs):
        order = sampler.order(epoch)
        skip = resume_skip if epoch == start_epoch else 0
        if skip:
            order = order[skip:]
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=OrderSampler(order), num_workers=args.num_workers)
        model.train()
        t0 = time.time()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=110, leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if kd_teacher is not None:
                # KL(student_logits/T || teacher_logits/T) · T²
                with torch.no_grad():
                    t_out = kd_teacher(x).float()
                s_logp = F.log_softmax(out / args.kd_temp, dim=1)
                t_p = torch.softmax(t_out / args.kd_temp, dim=1)
                kd = F.kl_div(s_logp, t_p, reduction="batchmean") * (args.kd_temp ** 2)
                loss = loss + args.kd_lambda * kd
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            total_loss += loss.item(); n_batch += 1; global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % args.ckpt_every == 0:
                torch.save({"model_state": model.state_dict(), "ema": ema,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "epoch": epoch, "global_step": global_step, "best_val_acc": best_val_acc,
                            "args": vars(args)}, resume_path)
                history["train_loss_curve"].append([global_step, loss.item()])
            if args.limit_iters and n_batch >= args.limit_iters:
                break
        pbar.close()
        scheduler.step()

        # eval bằng EMA weights
        if args.skip_val:
            val_metrics = {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0}
        else:
            model.load_state_dict(ema)
            val_metrics = evaluate(model, val_loader, device)
            model.train()
        train_loss = total_loss / max(n_batch, 1)
        history["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        dt = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"val_auc={val_metrics['roc_auc']:.4f} | {dt:.0f}s", flush=True)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({"state_dict": ema, "epoch": epoch + 1, "val_metrics": val_metrics,
                        "args": vars(args)}, best_path)
            print(f"  → best ckpt lưu tại {best_path}", flush=True)
        # luôn lưu final EMA mỗi epoch
        torch.save({"state_dict": ema, "epoch": epoch + 1, "val_metrics": val_metrics,
                    "args": vars(args)}, final_path)
        if args.limit_iters:
            break

    report = {
        "model": "DINOv3 ViT-S/16 Plus (finetune 129K)",
        "params_M": round(n_params / 1e6, 1),
        "best_val_acc": best_val_acc,
        "best_ckpt": best_path,
        "final_ckpt": final_path,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "history": history,
        "args": vars(args),
    }
    report_path = os.path.join(out_dir, f"{args.tag}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {report_path}")


if __name__ == "__main__":
    main()
