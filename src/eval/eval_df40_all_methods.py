"""So sánh ViT-S/16 Plus (28.7M) vs ConvNeXt-Tiny (27.8M) trên subset DF40 cân bằng.

Protocol: frozen backbone + Linear Probe (LogisticRegression), split video-disjoint.
Đọc manifest.csv -> method=="real" là label 0, còn lại label 1.

TỐI ƯU RAM (máy người dùng dễ tràn RAM):
- Feature được STREAM ra đĩa (numpy memmap) thay vì giữ hết trong RAM.
- torch.mps.empty_cache() định kỳ để giữ pool MPS nhỏ.
- Del model trước khi fit LogisticRegression.
- Có cache -> bị kill giữa chừng thì chạy lại tiếp tục từ cache.

Chạy:
  .venv/bin/python src/eval/eval_df40_all_methods.py
"""
import argparse
import csv
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODELS = [
    {"name": "DINOv3 ViT-S/16 Plus", "key": "vit", "path": "experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors",
     "loader": load_dinov3, "kw": {"img_size": IMG_SIZE}},
    {"name": "DINOv3 ConvNeXt-Tiny", "key": "cnn", "path": "experiments/checkpoints/weights/dinov3_next_cnn/model-2.safetensors",
     "loader": load_dinov3_convnext, "kw": {}},
]


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def read_manifest(root_dir, manifest_path):
    items = []  # (abs_path, method, video, label)
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(root_dir, row["path"]),
                          row["method"], row["video"],
                          0 if row["method"] == "real" else 1))
    return items


def video_disjoint_split(items, train_ratio, seed):
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault((it[1], it[2]), []).append(it)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_tr = max(1, int(len(keys) * train_ratio))
    tr_keys, te_keys = set(keys[:n_tr]), set(keys[n_tr:])
    train, test = [], []
    for k, v in groups.items():
        (train if k in tr_keys else test).extend(v)
    rng.shuffle(train); rng.shuffle(test)
    return train, test


@torch.no_grad()
def extract_features_stream(model, items, device, batch_size, cache, feat_dim, empty_every=20):
    """Extract feature STREAM ra memmap. Trả (feats_memmap, labels, methods)."""
    n = len(items)
    if cache and os.path.exists(cache + ".meta.npz"):
        meta = np.load(cache + ".meta.npz")
        pos = meta["labels"].shape[0]
        feats = np.memmap(cache, dtype=np.float32, mode="r", shape=(pos, feat_dim))
        return feats, meta["labels"], meta["methods"]

    model.to(device).eval()
    tf = build_transform()
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        feats = np.memmap(cache, dtype=np.float32, mode="w+", shape=(n, feat_dim))
    else:
        feats = np.zeros((n, feat_dim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    methods = np.empty(n, dtype="U64")

    pos = 0
    for i in tqdm(range(0, n, batch_size), desc="  Extract", unit="batch"):
        batch = items[i:i + batch_size]
        imgs, ok_idx = [], []
        for j, (p, _, _, _) in enumerate(batch):
            try:
                imgs.append(tf(Image.open(p).convert("RGB")))
                ok_idx.append(j)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        f = model(x).cpu().numpy()
        for k, j in enumerate(ok_idx):
            feats[pos] = f[k]
            labels[pos] = batch[j][3]
            methods[pos] = batch[j][1]
            pos += 1
        if device == "mps" and (i // batch_size) % empty_every == 0:
            torch.mps.empty_cache()
    del f
    gc.collect()

    labels = labels[:pos]; methods = methods[:pos]
    if cache:
        feats.flush()
        feats = np.memmap(cache, dtype=np.float32, mode="r", shape=(pos, feat_dim))
        np.savez_compressed(cache + ".meta.npz", labels=labels, methods=methods)
    else:
        feats = feats[:pos]
    return feats, labels, methods


def subsample_balanced(X_tr, y_tr, max_n):
    """Giới hạn số mẫu fit LR (giữ cân bằng real/fake) để giảm RAM giai đoạn fit."""
    if len(y_tr) <= max_n:
        return X_tr, y_tr
    rng = random.Random(42)
    idx = []
    for cls in (0, 1):
        pos = np.where(y_tr == cls)[0]
        want = max(1, max_n // 2)
        rng.shuffle(list(pos))
        idx.extend(pos[:want])
    idx = np.array(sorted(idx))
    return X_tr[idx], y_tr[idx]


def evaluate(X_tr, y_tr, X_te, y_te, lr_max_train=None):
    if lr_max_train:
        X_tr, y_tr = subsample_balanced(X_tr, y_tr, lr_max_train)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, y_prob)),
    }, y_pred


def per_method(y_pred, y_te, methods_te):
    out = {}
    for m in sorted(set(methods_te)):
        mask = methods_te == m
        n = int(mask.sum())
        if m == "real":
            out[m] = {"n": n, "acc_real": float((y_pred[mask] == 0).mean())}
        else:
            out[m] = {"n": n, "detection_rate": float((y_pred[mask] == 1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/df40_subset")
    ap.add_argument("--manifest", default="data/df40_subset/manifest.csv")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="df40",
                    help="prefix tên cache feature (tách cache theo dataset)")
    ap.add_argument("--lr-max-train", type=int, default=20000,
                    help="số mẫu tối đa fit LR (giữ cân bằng real/fake), giảm RAM")
    ap.add_argument("--output", default="experiments/results/df40_all_methods_report.json")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}")

    items = read_manifest(args.root, args.manifest)
    train, test = video_disjoint_split(items, args.train_ratio, seed=42)
    n_tr_r = sum(1 for _, m, _, l in train if l == 0)
    n_te_r = sum(1 for _, m, _, l in test if l == 0)
    print(f"Manifest: {len(items):,} ảnh")
    print(f"Train: {len(train):,} (real={n_tr_r:,}, fake={len(train)-n_tr_r:,})")
    print(f"Test : {len(test):,} (real={n_te_r:,}, fake={len(test)-n_te_r:,})")

    report = {"config": {k: v for k, v in vars(args).items()},
              "split": {"train": len(train), "test": len(test)}, "models": {}}
    for cfg in MODELS:
        print(f"\n→ {cfg['name']}")
        cache_tr = None if args.no_cache else f"experiments/results/features/{args.tag}_{cfg['key']}.mmap"
        cache_te = None if args.no_cache else f"experiments/results/features/{args.tag}_{cfg['key']}_test.mmap"
        t0 = time.time()
        X_tr, y_tr, _ = extract_features_stream(None if (cache_tr and os.path.exists(cache_tr + ".meta.npz")) else cfg["loader"](cfg["path"], **cfg["kw"]),
                                                train, device, args.batch_size, cache_tr, 768 if cfg["key"] == "cnn" else 384)
        X_te, y_te, m_te = extract_features_stream(cfg["loader"](cfg["path"], **cfg["kw"]),
                                                   test, device, args.batch_size, cache_te, 768 if cfg["key"] == "cnn" else 384)
        dt = time.time() - t0

        # giải phóng model trước khi fit LR
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        metrics, y_pred = evaluate(X_tr, y_tr, X_te, y_te, args.lr_max_train)
        pm = per_method(y_pred, y_te, m_te)
        n_det = sum(1 for m, v in pm.items() if m != "real" and v["n"])
        mean_det = float(np.mean([v["detection_rate"] for m, v in pm.items() if m != "real" and v["n"]]))
        print(f"  Time={dt:.0f}s  Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
              f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")
        print(f"  Real acc={pm['real']['acc_real']:.4f} | Mean detection ({n_det} fake methods)={mean_det:.4f}")
        report["models"][cfg["key"]] = {"name": cfg["name"], "params_M": None,
                                        "extract_time_s": round(dt, 1), "metrics": metrics,
                                        "mean_detection_rate": round(float(mean_det), 4),
                                        "real_acc": round(float(pm['real']['acc_real']), 4),
                                        "per_method": {m: v for m, v in sorted(pm.items())}}
        del X_tr, X_te, y_tr, y_te, m_te, y_pred, metrics, pm
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
