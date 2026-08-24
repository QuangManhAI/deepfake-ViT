#!/usr/bin/env python3
"""Evaluate exp02_best vs exp02_weak_methods on the same balanced test set (BICUBIC protocol).

Writes per-ckpt metrics + a before/after per-method FN-rate comparison:
  experiments/results/exp02_weak_finetune/eval_<tag>.json
  experiments/results/exp02_weak_finetune/before_after_methods.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path("/workspace/quangmanh/deepfake")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/workspace/hoangtuan/deepfake-ViT")  # fallback DinoViT

from src.models.dinov3_vit import DinoViT

TEST_CSV = PROJECT_ROOT / "experiments/results/error_analysis_lora/test_balanced.csv"
OUT = PROJECT_ROOT / "experiments/results/exp02_weak_finetune"
IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    """EnhancedDinoViTClassifier — head Sequential directly so state_dict keys match head.0/2/5.*."""

    def __init__(self, backbone, hidden_dim=384, num_classes=2, dropout=0.2):
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


def build_model(ckpt_path):
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = Model(DinoViT(img_size=IMG_SIZE, gated_mlp=True))
    missing, unexpected = model.load_state_dict(ck["model_state_dict"])
    assert not missing and not unexpected, f"{ckpt_path.name}: missing={missing} unexpected={unexpected}"
    return model, ck


class CsvImageDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["path"], int(r["label"]), r.get("method", ""),
                                  r.get("domain", ""), r.get("identity", "")))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, *_ = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def evaluate(model, loader):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().tolist())
            probs.extend(p[:, 1].cpu().tolist())
    return np.array(preds), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "/workspace/hoangtuan/deepfake-ViT/experiments/checkpoints/dinov3_vit_exp02_best.pt",
        str(PROJECT_ROOT / "outputs/finetune/exp02_weak_methods.pt")])
    ap.add_argument("--tags", nargs="+", default=["exp02_best", "exp02_weak_finetune"])
    args = ap.parse_args()
    assert len(args.ckpts) == len(args.tags)
    OUT.mkdir(parents=True, exist_ok=True)

    ds = CsvImageDataset(str(TEST_CSV))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    methods = np.array([r[2] for r in ds.rows])
    labels = np.array([r[1] for r in ds.rows])
    print(f"Test: {len(labels)} rows ({int((labels == 0).sum())} real / {int((labels == 1).sum())} fake)")

    per_ckpt = {}
    all_det = {}  # method -> {tag: detection rate}
    for ckpt_path, tag in zip(args.ckpts, args.tags):
        if not Path(ckpt_path).exists():
            print(f"!! skip {tag}: {ckpt_path} not found")
            continue
        model, ck = build_model(ckpt_path)
        model.eval().to(device)
        print(f"{tag}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, epoch={ck.get('epoch')}")
        preds, probs = evaluate(model, loader)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m = {
            "ckpt": ckpt_path,
            "accuracy": float((preds == labels).mean()),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }
        per_ckpt[tag] = m
        print(f"  acc {m['accuracy']*100:.2f}% | FN {fn} | FP {fp} | auc {m['roc_auc']:.4f}")

        # per-method detection rate (fake recall per method)
        det = {}
        for mth in sorted(set(methods)):
            if mth == "real":
                continue
            sel = methods == mth
            n = int(sel.sum())
            if n == 0:
                continue
            det_rate = float((preds[sel] == 1).mean())
            det[mth] = {"n": n, "det_rate": det_rate, "FN": int(((labels[sel] == 1) & (preds[sel] == 0)).sum())}
        all_det[tag] = det

        np.savez_compressed(OUT / f"eval_{tag}.npz", preds=preds, probs=probs, labels=labels, methods=methods)
        with open(OUT / f"eval_{tag}.json", "w") as f:
            json.dump(m, f, indent=2)

    # ---- before/after comparison ----
    if len(per_ckpt) == 2:
        b, a = args.tags[0], args.tags[1]
        rows = []
        for mth in sorted(all_det[b], key=lambda k: all_det[b][k]["det_rate"]):
            rb, ra = all_det[b][mth], all_det[a].get(mth)
            if ra is None:
                continue
            rows.append({
                "method": mth,
                "n": rb["n"],
                "det_before": rb["det_rate"],
                "det_after": ra["det_rate"],
                "delta_det": ra["det_rate"] - rb["det_rate"],
                "FN_before": rb["FN"],
                "FN_after": ra["FN"],
            })
        rows.sort(key=lambda r: r["delta_det"])
        with open(OUT / "before_after_methods.json", "w") as f:
            json.dump({"before_tag": b, "after_tag": a, "rows": rows,
                       "before_metrics": per_ckpt[b], "after_metrics": per_ckpt[a]}, f, indent=2)
        print(f"\nPer-method detection-rate change ({b} -> {a}):")
        for r in rows:
            flag = "▼" if r["delta_det"] < -0.01 else ("▲" if r["delta_det"] > 0.01 else " ")
            print(f"  {flag} {r['method']:<16} {r['det_before']*100:5.1f}% -> {r['det_after']*100:5.1f}% "
                  f"({r['delta_det']*100:+.1f}pp)  FN {r['FN_before']:>4}->{r['FN_after']:>4}")
        n_better = sum(1 for r in rows if r["delta_det"] > 0.005)
        n_worse = sum(1 for r in rows if r["delta_det"] < -0.005)
        print(f"\nBetter: {n_better} | Worse: {n_worse} | real_acc before {per_ckpt[b]['accuracy']*100:.2f}% "
              f"-> after {per_ckpt[a]['accuracy']*100:.2f}%")
    print("\nDone. Artifacts in", OUT)


if __name__ == "__main__":
    main()
