#!/usr/bin/env python3
"""So sánh 3 model trên test coursework balanced mới (zero-leakage, 21,446 ảnh).

Models:
  - plus_v3_best     : DINOv3 ViT-S/16 Plus finetune local (v5_weakfix_v3, 129K) — 28.7M
  - best_model_v3    : DINOv3 ViT-S/16 server (best_model_v3.pt trong bundle test) — 21.6M
  - convnext_weakfix_v3 : DINOv3 ConvNeXt best (tải từ HF ManhQuangAI/convnext-weakfix-datasets)

Báo cáo: overall acc/prec/rec/f1/auc + real_acc/fake_recall + per-method det_rate
(5 nhóm yếu được đánh dấu) + real theo source. fp32 trên MPS (không bf16).
"""
import argparse
import collections
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
from src.models.dinov3_convnext import load_dinov3_convnext
from src.models.classifier_v2 import DinoConvNextClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "data", "deepfake_test_suite_full_50k")
TEST_CSV = os.path.join(BUNDLE, "splits", "test_coursework_44methods_balanced_zero_leakage.csv")

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

WEAK = {"wav2lip", "deepfacelab", "fsgan", "faceswap", "sadtalker"}

MODELS = {
    "Plus_viT_v3": dict(
        ckpt=os.path.join(ROOT, "outputs", "finetune", "plus_v3_best.pt"),
        sd_key="state_dict",
        arch="vit",
        backbone=os.path.join(ROOT, "models", "dinov3-vits16plus-pretrain-lvd1689m", "model-3.safetensors"),
    ),
    "ViTsmall_server_v3": dict(
        ckpt=os.path.join(BUNDLE, "checkpoints", "best_model_v3.pt"),
        sd_key="model_state_dict",
        arch="vit",
        backbone=os.path.join(ROOT, "models", "dinov3_small", "model.safetensors"),
    ),
    "ConvNeXt_v3": dict(
        ckpt=os.path.join(ROOT, "models", "convnext_weakfix_v3", "convnext_weakfix_v3.pt"),
        sd_key="model_state_dict",
        arch="convnext",
        backbone=os.path.join(ROOT, "models", "dinov3_next_cnn", "model-2.safetensors"),
    ),
}


class BackboneClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class CsvImageDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []  # (local_path, label, method, source)
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                local = os.path.join(BUNDLE, "test_images", r["path"].lstrip("/"))
                self.rows.append((local, int(r["label"]), r["method"], r.get("source", "")))
        self.transform = EVAL_TF

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, method, source = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, method, source


def build_model(cfg):
    if cfg["arch"] == "convnext":
        model = DinoConvNextClassifier(load_dinov3_convnext(cfg["backbone"]))
    else:
        model = BackboneClassifier(load_dinov3(cfg["backbone"], img_size=IMG_SIZE))
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck[cfg["sd_key"]], strict=False)
    assert not missing and not unexpected, f"{cfg['ckpt']}: missing={len(missing)} unexpected={len(unexpected)}"
    return model, ck


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    preds, probs = [], []
    for x, *_ in tqdm(loader, desc="Infer", ncols=90, leave=False):
        x = x.to(device)
        logits = model(x).float()
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=None,
                    help="các tag model trong MODELS; mặc định chạy tất cả")
    ap.add_argument("--extra-model", action="append", default=None,
                    metavar="TAG=CKPT",
                    help="thêm model stage ablation (arch=vit Plus, sd_key=state_dict). "
                         "CKPT tương đối ROOT hoặc đường dẫn tuyệt đối")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    # Đăng ký model stage ablation (A1/A2...) không cần sửa MODELS
    if args.extra_model:
        plus_bk = os.path.join(ROOT, "models", "dinov3-vits16plus-pretrain-lvd1689m", "model-3.safetensors")
        for em in args.extra_model:
            tag, ckpt = em.split("=", 1)
            if not os.path.isabs(ckpt):
                ckpt = os.path.join(ROOT, ckpt)
            MODELS[tag] = dict(ckpt=ckpt, sd_key="state_dict", arch="vit", backbone=plus_bk)
    tags = args.tags or list(MODELS.keys())

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | torch {torch.__version__}")

    ds = CsvImageDataset(TEST_CSV)
    labels = np.array([r[1] for r in ds.rows])
    methods = np.array([r[2] for r in ds.rows])
    sources = np.array([r[3] for r in ds.rows])
    print(f"Test: {len(ds)} rows ({int((labels==0).sum())} real / {int((labels==1).sum())} fake)")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    out_dir = os.path.join(ROOT, "experiments", "results", "coursework_vs")
    os.makedirs(out_dir, exist_ok=True)
    per_ckpt, det_all = {}, {}
    for tag in tags:
        cfg = MODELS[tag]
        if not os.path.exists(cfg["ckpt"]):
            print(f"!! skip {tag}: {cfg['ckpt']} not found")
            continue
        model, ck = build_model(cfg)
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{tag}: {n_param:.1f}M params | best_val_auc(ck)={ck.get('best_val_auc')}")
        model.to(device)
        preds, probs = infer(model, loader, device)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m = {
            "ckpt": cfg["ckpt"],
            "n_params_M": round(n_param, 1),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "real_acc": float((preds[labels == 0] == 0).mean()),
            "fake_recall": float((preds[labels == 1] == 1).mean()),
        }
        per_ckpt[tag] = m
        np.savez_compressed(os.path.join(out_dir, f"{tag}_preds.npz"),
                            preds=preds, probs=probs, labels=labels, methods=methods, sources=sources)

        det = {}
        for mth in sorted(set(methods)):
            if mth == "real":
                continue
            sel = (methods == mth) & (labels == 1)
            n = int(sel.sum())
            if n == 0:
                continue
            det[mth] = {"n": n, "det_rate": float((preds[sel] == 1).mean()),
                        "FN": int(((labels[sel] == 1) & (preds[sel] == 0)).sum()),
                        "mean_prob": float(probs[sel].mean())}
        det_all[tag] = det
        with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
            json.dump(m, f, indent=2)

    # ---- Bảng in ----
    print(f"\n{'='*90}\nKẾT QUẢ — test coursework balanced zero-leakage ({len(ds)} ảnh)\n{'='*90}")
    hdr = f"{'model':<24}{'acc%':>7}{'auc':>8}{'real_acc':>10}{'fake_rec':>10}{'FP':>5}{'FN':>5}"
    print(hdr)
    for tag in tags:
        if tag not in per_ckpt:
            continue
        m = per_ckpt[tag]
        print(f"{tag:<24}{m['accuracy']*100:7.2f}{m['roc_auc']:8.4f}{m['real_acc']:10.4f}"
              f"{m['fake_recall']:10.4f}{m['FP']:5d}{m['FN']:5d}")

    # per-method det (chỉ in các nhóm fake; đánh dấu 5 nhóm yếu)
    first = [t for t in tags if t in det_all][0]
    print(f"\nPer-method detection-rate (fake):")
    print(f"  {'method':<16}{'n':>5}" + "".join(f"{t:>10}" for t in det_all))
    for mth in sorted(det_all[first], key=lambda k: -det_all[first][k]["n"]):
        mark = " ◄YẾU" if mth in WEAK else ""
        line = f"  {mth:<16}{det_all[first][mth]['n']:>5}"
        for tag in det_all:
            line += f"{det_all[tag].get(mth, {}).get('det_rate', float('nan'))*100:9.1f}%"
        print(line + mark)

    # Real theo source
    print(f"\nReal acc theo source:")
    for src in sorted(set(sources[labels == 0])):
        sel = (sources == src) & (labels == 0)
        n = int(sel.sum())
        row = f"  {src:<24}{n:>5}"
        for tag in per_ckpt:
            pr = np.load(os.path.join(out_dir, f"{tag}_preds.npz"))["preds"]
            row += f"{float((pr[sel] == 0).mean())*100:9.1f}%"
        print(row)

    # ---- Tổng hợp JSON ----
    weak_rows = {mth: {t: det_all[t].get(mth) for t in det_all if mth in det_all[t]} for mth in WEAK}
    summary = {"test": TEST_CSV, "n": len(ds), "per_ckpt": per_ckpt,
               "weak_groups": {mth: {t: (v["det_rate"] if v else None) for t, v in rows.items()}
                               for mth, rows in weak_rows.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu: {out_dir}/summary.json")


if __name__ == "__main__":
    main()
