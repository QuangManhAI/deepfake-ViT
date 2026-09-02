#!/usr/bin/env python3
"""Linear probe trên backbone pretrained (đông cứng) — eval test coursework balanced 21,446.

2 backbone pretrained (không head):
  - Pretr_Plus_v3    : models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors (ViT-S/16, embed 384)
  - Pretr_ConvNeXt_v3: models/dinov3_next_cnn/model-2.safetensors                (ConvNeXt, embed 768)

Protocol (sánh công bằng với finetune A0/A1/ConvNeXt):
  1) Extract feature đông cứng (no_grad) toàn bộ TRAIN 123,582 ảnh, L2-normalize từng ảnh.
  2) Fit LogisticRegression(class_weight='balanced') lên feature đông cứng.
  3) Eval trên TEST cân bằng 21,446 (cùng transform 256 + normalize như eval finetune)
     → overall + real_acc/fake_recall + per-method det + real theo source.
Lưu preds npz + eval json đặt cạnh các preds finetune trong experiments/results/coursework_vs/.

fp32 trên MPS. Chạy detached (bài học RAM: task bị kill → Popen start_new_session).
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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ảnh jpg bị cắt giữa chừng vẫn decode (PIL điền xám)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "data", "deepfake_test_suite_full_50k")
TRAIN_CSV = os.path.join(ROOT, "data", "splits", "finetune_plus_train.csv")
TEST_CSV = os.path.join(BUNDLE, "splits", "test_coursework_44methods_balanced_zero_leakage.csv")
OUT = os.path.join(ROOT, "experiments", "results", "coursework_vs")
CACHE_DIR = os.path.join(ROOT, "scratch", "probe_cache")  # train feats đã extract → không làm lại 123K ảnh

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# tag -> (backbone path, embed_dim)
BACKBONES = {
    "Pretr_Plus_v3": (os.path.join(ROOT, "models", "dinov3-vits16plus-pretrain-lvd1689m", "model-3.safetensors"), 384, "vit"),
    "Pretr_ConvNeXt_v3": (os.path.join(ROOT, "models", "dinov3_next_cnn", "model-2.safetensors"), 768, "convnext"),
}


class CsvImageDataset(Dataset):
    """CSV: path(absolute or under bundle/test_images), label, method, source."""
    def __init__(self, csv_path, bundle_img_root=None, has_source=True, verify=True):
        rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                p = r["path"]
                # CSV test lưu path dạng /workspace/... → nếu có bundle root thì LUÔN rebase
                # (file thật nằm ở <root>/test_images/workspace/data/... như eval_coursework_vs.py)
                if bundle_img_root:
                    p = os.path.join(bundle_img_root, p.lstrip("/"))
                src = r.get("source", r.get("method", "")) if has_source else r.get("method", "")
                rows.append((p, int(r["label"]), r["method"], src))
        if verify:  # bỏ ảnh hỏng (header không đọc được) để worker không crash
            bad = []
            for row in rows:
                try:
                    with Image.open(row[0]) as im:
                        im.verify()
                except Exception:
                    bad.append(row[0])
            if bad:
                print(f"  [verify] bỏ {len(bad)} ảnh hỏng; vd: {bad[:3]}", flush=True)
                badset = set(bad)
                rows = [r for r in rows if r[0] not in badset]
        self.rows = rows
        self.transform = EVAL_TF

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, method, source = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_backbone(tag):
    path, dim, kind = BACKBONES[tag]
    if kind == "convnext":
        model = load_dinov3_convnext(path)
    else:
        model = load_dinov3(path, img_size=IMG_SIZE)
    assert model.embed_dim == dim, (tag, model.embed_dim, dim)
    return model


@torch.no_grad()
def extract(model, loader, device, n, dim, desc):
    """Streaming extract → mảng float32 [n, dim]; giữ RAM thấp."""
    feats = np.zeros((n, dim), dtype=np.float32)
    model.eval()
    i0 = 0
    for x, *_ in tqdm(loader, desc=desc, ncols=100):
        x = x.to(device)
        f = model(x).float().cpu().numpy()
        b = f.shape[0]
        feats[i0:i0 + b] = f
        i0 += b
    assert i0 == n, (i0, n)
    return feats


def fit_probe(Xtr, ytr):
    # L2-normalize từng ảnh (chuẩn linear-probe DINO), fit Logistic cân bằng 2 lớp.
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    clf = LogisticRegression(solver="lbfgs", max_iter=3000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf


def run_tag(tag, args):
    t0 = time.time()
    device = args.device
    print(f"\n===== {tag} =====", flush=True)
    backbone = make_backbone(tag)
    backbone.to(device).eval()
    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"{tag}: {n_params:.1f}M backbone | device {device}", flush=True)

    xpath = os.path.join(CACHE_DIR, f"{tag}_Xtr.npy")
    ypath = os.path.join(CACHE_DIR, f"{tag}_ytr.npy")
    if os.path.exists(xpath) and os.path.exists(ypath):
        Xtr = np.load(xpath)
        ytr = np.load(ypath)
        print(f"  [cache] load train feats {Xtr.shape} (skip extract 123K)", flush=True)
    else:
        tr_ds = CsvImageDataset(TRAIN_CSV)
        ytr = np.array([r[1] for r in tr_ds.rows])
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
        print(f"Extract TRAIN features {len(tr_ds)} ảnh (real {int((ytr==0).sum())}/fake {int((ytr==1).sum())}) ...", flush=True)
        Xtr = extract(backbone, tr_loader, device, len(tr_ds), backbone.embed_dim, f"{tag} extract-train")
        del tr_ds, tr_loader
        torch.cuda.empty_cache() if device == "cuda" else None
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(xpath, Xtr); np.save(ypath, ytr)
        print(f"  [cache] đã lưu {xpath}", flush=True)
    print(f"  train features {Xtr.shape} | fit Logistic...", flush=True)
    clf = fit_probe(Xtr, ytr)
    tr_acc = accuracy_score(ytr, clf.predict(Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)))
    print(f"  probe train acc={tr_acc:.4f} | n_iter={clf.n_iter_} | C={clf.C:.3g}", flush=True)
    del Xtr

    te_ds = CsvImageDataset(TEST_CSV, bundle_img_root=os.path.join(BUNDLE, "test_images"))
    y = np.array([r[1] for r in te_ds.rows])
    methods = np.array([r[2] for r in te_ds.rows])
    sources = np.array([r[3] for r in te_ds.rows])
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    print(f"Extract TEST features {len(te_ds)} ảnh ...", flush=True)
    Xte = extract(backbone, te_loader, device, len(te_ds), backbone.embed_dim, f"{tag} extract-test")
    del te_ds, te_loader
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    prob1 = clf.predict_proba(Xn)[:, 1]
    pred = (prob1 >= 0.5).astype(int)
    del Xte, Xn
    np.savez_compressed(os.path.join(OUT, f"{tag}_preds.npz"),
                        preds=pred, probs=prob1, labels=y, methods=methods, sources=sources)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "ckpt": BACKBONES[tag][0], "n_params_backbone_M": round(n_params, 1),
        "probe": "LogisticReg balanced, feature L2-norm, train full 123,582",
        "train_acc": round(float(tr_acc), 4),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob1)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "real_acc": float((pred[y == 0] == 0).mean()),
        "fake_recall": float((pred[y == 1] == 1).mean()),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(OUT, f"eval_{tag}.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{tag}: acc={m['accuracy']:.4f} real_acc={m['real_acc']:.4f} "
          f"fake_rec={m['fake_recall']:.4f} AUC={m['roc_auc']:.4f} "
          f"FP={fp} FN={fn} | {m['elapsed_min']} min", flush=True)

    det = {}
    for mth in sorted(set(methods)):
        if mth == "real":
            continue
        sel = (methods == mth) & (y == 1)
        n = int(sel.sum())
        if n == 0:
            continue
        det[mth] = {"n": n, "det_rate": float((pred[sel] == 1).mean()),
                    "FN": int(((y[sel] == 1) & (pred[sel] == 0)).sum()),
                    "mean_prob": float(prob1[sel].mean())}
    with open(os.path.join(OUT, f"det_{tag}.json"), "w") as f:
        json.dump(det, f, indent=2)
    return m, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(BACKBONES.keys()))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=3)
    args = ap.parse_args()
    args.device = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {args.device} | torch {torch.__version__}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    per_ckpt, det_all = {}, {}
    for tag in args.tags:
        try:
            m, det = run_tag(tag, args)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[WARN] {tag} fail — bỏ qua, chạy tiếp tag khác. Train feats đã cache, rerun chạy nhanh.", flush=True)
            continue
        per_ckpt[tag] = m
        det_all[tag] = det
    summary = {"test": TEST_CSV, "n": None, "per_ckpt": per_ckpt, "note":
               "linear probe frozen backbone; train=finetune_plus_train full; balanced test 21,446"}
    with open(os.path.join(OUT, "pretrained_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDONE. Đã lưu eval_{tag}.json + {tag}_preds.npz + pretrained_summary.json", flush=True)


if __name__ == "__main__":
    main()
