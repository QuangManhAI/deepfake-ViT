"""Eval identity-disjoint trên test_data_v2 (frozen backbone + LR probe).

test_data_v2: mỗi khuôn mặt xuất hiện DUY NHẤT 1 lần. Phân chia theo IDENTITY key
(manifest cột `identity`): real+fake của cùng 1 người luôn cùng 1 nhánh split.
Model chỉ được test trên identity CHƯA từng thấy trong train.

Báo cáo:
  - tổng: Acc/Prec/Rec/F1/AUC, real acc, fake detection
  - theo DOMAIN: cdc (Celeb-DF) / ffc (FF++) / efs (tổng hợp) / oth (không ghép)
  - theo METHOD (fake): detection rate
  - PAIRED-ONLY: chỉ xét identity có cả real lẫn fake (test nghiêm ngặt nhất)

TỐI ƯU RAM: feature STREAM ra memmap, del model trước khi fit LR, empty_cache định kỳ.

Chạy:
  .venv/bin/python scripts/eval_identity_disjoint.py --model vit --tag test_data_v2
  .venv/bin/python scripts/eval_identity_disjoint.py --model cnn --tag test_data_v2
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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODELS = [
    {"name": "DINOv3 ViT-S/16 Plus", "key": "vit", "path": "models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors",
     "loader": load_dinov3, "kw": {"img_size": IMG_SIZE}, "dim": 384},
    {"name": "DINOv3 ConvNeXt-Tiny", "key": "cnn", "path": "models/dinov3_next_cnn/model-2.safetensors",
     "loader": load_dinov3_convnext, "kw": {}, "dim": 768},
]

FEAT_DIM = {m["key"]: m["dim"] for m in MODELS}


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def read_manifest(root):
    """items = (abs_path, method, video, label, identity, domain)."""
    items = []
    with open(os.path.join(root, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(root, row["path"]),
                          row["method"], row["video"],
                          0 if row["method"] == "real" else 1,
                          row["identity"], row["domain"]))
    return items


def identity_disjoint_split(items, train_ratio, seed):
    """Split theo identity key (manifest cột identity)."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault(it[4], []).append(it)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_tr = max(1, int(len(keys) * train_ratio))
    tr_keys, te_keys = set(keys[:n_tr]), set(keys[n_tr:])
    train, test = [], []
    for k, v in groups.items():
        (train if k in tr_keys else test).extend(v)
    rng.shuffle(train); rng.shuffle(test)
    return train, test, keys


@torch.no_grad()
def extract_features_stream(model, items, device, batch_size, cache, feat_dim):
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
    for i in tqdm(range(0, n, batch_size), desc="  Extract", unit="batch", ncols=90):
        batch = items[i:i + batch_size]
        imgs, ok_idx = [], []
        for j, (p, _, _, _, _, _) in enumerate(batch):
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
        if device == "mps" and (i // batch_size) % 20 == 0:
            torch.mps.empty_cache()
    labels = labels[:pos]; methods = methods[:pos]
    if cache:
        feats.flush()
        feats = np.memmap(cache, dtype=np.float32, mode="r", shape=(pos, feat_dim))
        np.savez_compressed(cache + ".meta.npz", labels=labels, methods=methods)
    else:
        feats = feats[:pos]
    return feats, labels, methods


def fit_eval(X_tr, y_tr, X_te, y_te):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=42))
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    pr = clf.predict_proba(X_te)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_te, yp)),
        "precision": float(precision_score(y_te, yp, zero_division=0)),
        "recall": float(recall_score(y_te, yp, zero_division=0)),
        "f1": float(f1_score(y_te, yp, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, pr)),
    }, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "cnn"])
    ap.add_argument("--root", default="test_data_v3")
    ap.add_argument("--tag", default="test_data_v3")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    cfg = next(m for m in MODELS if m["key"] == args.model)
    print(f"Device: {device} | model: {cfg['name']}")

    items = read_manifest(args.root)
    train, test, keys = identity_disjoint_split(items, args.train_ratio, seed=42)
    n_tr_r = sum(1 for it in train if it[3] == 0)
    n_te_r = sum(1 for it in test if it[3] == 0)
    print(f"Manifest: {len(items):,} ảnh | {len(keys):,} identity keys "
          f"({int(len(keys)*0.7):,} train / {len(keys)-int(len(keys)*0.7):,} test)")
    print(f"Train: {len(train):,} (real={n_tr_r:,}) | Test: {len(test):,} (real={n_te_r:,})")

    cache_tr = f"outputs/features/{args.tag}_{args.model}.mmap"
    cache_te = f"outputs/features/{args.tag}_{args.model}_test.mmap"
    need_tr = not os.path.exists(cache_tr + ".meta.npz")
    need_te = not os.path.exists(cache_te + ".meta.npz")
    model = None
    if need_tr or need_te:
        model = cfg["loader"](cfg["path"], **cfg["kw"])
    t0 = time.time()
    X_tr, y_tr, _ = extract_features_stream(
        model if need_tr else None, train, device, args.batch_size, cache_tr, cfg["dim"])
    X_te, y_te, m_te = extract_features_stream(
        model if need_te else None, test, device, args.batch_size, cache_te, cfg["dim"])
    dt = time.time() - t0

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # alignment: label cache == label items
    assert len(y_tr) == len(train), f"train cache lệch {len(y_tr)} vs {len(train)}"
    assert len(y_te) == len(test), f"test cache lệch {len(y_te)} vs {len(test)}"
    print(f"Alignment OK — extract {dt:.0f}s")

    metrics, yp = fit_eval(X_tr, y_tr, X_te, y_te)
    yte = y_te
    mr = yte == 0; mf = yte == 1
    out = {
        "model": args.model, "tag": args.tag, "protocol": "identity-disjoint",
        "n_identity_keys": len(keys), "n_train": int(len(train)), "n_test": int(len(test)),
        "metrics": metrics,
        "real_acc": float((yp[mr] == 0).mean()),
        "fake_detection": float((yp[mf] == 1).mean()),
    }
    print(f"\n=== {args.model.upper()} — IDENTITY-DISJOINT ===")
    print(f"  Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
          f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")
    print(f"  Real acc={out['real_acc']:.4f} | Fake detection={out['fake_detection']:.4f}")

    # ---- theo domain ----
    print(f"\n  Theo DOMAIN (test):")
    dom_te = np.array([it[5] for it in test])
    dom = {}
    for d in sorted(set(dom_te)):
        mask = dom_te == d
        n = int(mask.sum())
        if n == 0:
            continue
        if int((yte[mask] == 0).sum()) and int((yte[mask] == 1).sum()):
            acc_d = accuracy_score(yte[mask], yp[mask])
            real_d = float((yp[mask][yte[mask] == 0] == 0).mean()) if (yte[mask] == 0).any() else float("nan")
            fake_d = float((yp[mask][yte[mask] == 1] == 1).mean()) if (yte[mask] == 1).any() else float("nan")
            print(f"    {d:4s} {n:5d} ảnh  acc={acc_d:.4f} real={real_d:.3f} fake_det={fake_d:.3f}")
            dom[d] = {"n": n, "acc": round(acc_d, 4), "real_acc": round(real_d, 4),
                      "fake_det": round(fake_d, 4)}
        else:
            acc_d = accuracy_score(yte[mask], yp[mask])
            print(f"    {d:4s} {n:5d} ảnh  acc={acc_d:.4f}")
            dom[d] = {"n": n, "acc": round(acc_d, 4)}
    out["per_domain"] = dom

    # ---- theo method (fake) + real theo domain ----
    print(f"\n  {'method':12s}{'dom':5s}{'n':>6s}{'det':>8s}")
    pm = {}
    for mth in sorted(set(m_te)):
        mask = m_te == mth
        n = int(mask.sum())
        if n == 0:
            continue
        d = dom_te[mask][0]
        if mth == "real":
            det = float((yp[mask] == 0).mean())
            pm[f"real/{d}"] = {"n": n, "acc_real": round(det, 4)}
            print(f"  {'real':12s}{d:5s}{n:6d}{det:8.3f}")
        else:
            det = float((yp[mask] == 1).mean())
            pm[mth] = {"n": n, "detection_rate": round(det, 4)}
            print(f"  {mth:12s}{d:5s}{n:6d}{det:8.3f}")
    out["per_method"] = pm

    # ---- chi tiết theo (method, domain): 1 method có thể trải nhiều domain ----
    pmd = {}
    for mth in sorted(set(m_te)):
        mask_m = m_te == mth
        for d in sorted(set(dom_te[mask_m])):
            m = mask_m & (dom_te == d)
            n = int(m.sum())
            if n == 0:
                continue
            det = float((yp[m] == 1).mean()) if mth != "real" else float((yp[m] == 0).mean())
            pmd[f"{mth}/{d}"] = {"n": n, "rate": round(det, 4)}
    out["per_method_domain"] = pmd

    # ---- PAIRED-ONLY: chỉ identity có cả real+fake ----
    test_ids = {}
    for it in test:
        test_ids.setdefault(it[4], []).append(it)
    paired_mask = np.array([False] * len(yte))
    paired_info = {}
    idx = 0
    for it in test:
        if any(x[3] == 0 for x in test_ids[it[4]]) and any(x[3] == 1 for x in test_ids[it[4]]):
            paired_mask[idx] = True
            paired_info[idx] = it[4]
        idx += 1
    if paired_mask.any():
        ypp = yp[paired_mask]; yp2 = yte[paired_mask]
        acc_p = accuracy_score(yp2, ypp)
        n_p_r = int((yp2 == 0).sum()); n_p_f = int((yp2 == 1).sum())
        real_p = float((ypp[yp2 == 0] == 0).mean()) if n_p_r else float("nan")
        fake_p = float((ypp[yp2 == 1] == 1).mean()) if n_p_f else float("nan")
        print(f"\n  PAIRED-ONLY (identity có cả real+fake, {len(ypp):,} ảnh "
              f"real={n_p_r} fake={n_p_f}):")
        print(f"    Acc={acc_p:.4f} real_acc={real_p:.3f} fake_det={fake_p:.3f}")
        out["paired_only"] = {
            "n": int(len(ypp)), "n_real": n_p_r, "n_fake": n_p_f,
            "accuracy": round(acc_p, 4), "real_acc": round(float(real_p), 4),
            "fake_det": round(float(fake_p), 4),
        }
    else:
        print("\n  PAIRED-ONLY: không có identity paired trong test split")
        out["paired_only"] = None

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
