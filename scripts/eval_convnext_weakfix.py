#!/usr/bin/env python3
"""Evaluate convnext baseline / v2 / v3 on the zero-leakage test benchmark.

Replicates doc/v5_weakfix/04_ket_qua: per-ckpt metrics, per-method detection rate,
faceswap per-identity detail, before/after comparison. Writes:
  experiments/results/convnext_weakfix/eval_<tag>.json
  experiments/results/convnext_weakfix/per_method_<tag>.csv
  experiments/results/convnext_weakfix/faceswap_detail.csv
  experiments/results/convnext_weakfix/comparison.json

Run: /workspace/hoangtuan/deepfake-ViT/.venv/bin/python scripts/eval_convnext_weakfix.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path("/workspace/quangmanh/deepfake")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_convnext import load_dinov3_convnext
from src.models.classifier_v2 import DinoConvNextClassifier

PRETRAINED = PROJECT_ROOT / "models/dinov3_next_cnn/model-2.safetensors"
TEST_CSV = Path("/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv")
OUT = PROJECT_ROOT / "experiments" / "results" / "convnext_weakfix"
IMG_SIZE = 256
BATCH = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CKPTS = {
    "baseline": str(PROJECT_ROOT / "outputs/finetune/convnext_baseline.pt"),
    "v2": str(PROJECT_ROOT / "outputs/finetune/convnext_weakfix_v2.pt"),
    "v3": str(PROJECT_ROOT / "outputs/finetune/convnext_weakfix_v3.pt"),
}


def build_model(ckpt_path):
    model = DinoConvNextClassifier(load_dinov3_convnext(str(PRETRAINED)))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    assert not missing and not unexpected, f"{Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)}"
    return model, ck


class CsvImageDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path, newline="") as f:
            for r in __import__("csv").DictReader(f):
                self.rows.append((r["path"], int(r["label"]), r.get("method", ""),
                                  r.get("domain", ""), r.get("identity", "")))
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, *_ = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def evaluate(model, loader):
    """bf16 autocast — identical protocol to the v5_weakfix finetune scripts' evaluate().
    (fp32 shifts borderline predictions vs the bf16 reference numbers.)"""
    model.eval()
    preds, probs = [], []
    for x, _ in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x)
        out32 = logits.float()
        p = torch.softmax(out32, 1)
        preds.extend(out32.argmax(1).cpu().tolist())
        probs.extend(p[:, 1].cpu().tolist())
    return np.array(preds), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=list(DEFAULT_CKPTS.values()))
    ap.add_argument("--tags", nargs="+", default=list(DEFAULT_CKPTS.keys()))
    args = ap.parse_args()
    assert len(args.ckpts) == len(args.tags)
    OUT.mkdir(parents=True, exist_ok=True)

    ds = CsvImageDataset(str(TEST_CSV))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    methods = np.array([r[2] for r in ds.rows])
    labels = np.array([r[1] for r in ds.rows])
    identities = np.array([r[4] for r in ds.rows])
    print(f"Test: {len(labels)} rows ({int((labels==0).sum())} real / {int((labels==1).sum())} fake) | device {device}")

    per_ckpt, all_det = {}, {}
    for ckpt_path, tag in zip(args.ckpts, args.tags):
        if not Path(ckpt_path).exists():
            print(f"!! skip {tag}: {ckpt_path} not found")
            continue
        model, ck = build_model(ckpt_path)
        model.eval().to(device)
        print(f"{tag}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | best_val_auc={ck.get('best_val_auc')}")
        preds, probs = evaluate(model, loader)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m = {
            "ckpt": ckpt_path,
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
        print(f"  acc {m['accuracy']*100:.2f}% | FN {fn} | FP {fp} | auc {m['roc_auc']:.4f}")

        det = {}
        for mth in sorted(set(methods)):
            if mth == "real":
                continue
            sel = methods == mth
            n = int(sel.sum())
            if n == 0:
                continue
            det[mth] = {"n": n,
                        "det_rate": float((preds[sel] == 1).mean()),
                        "FN": int(((labels[sel] == 1) & (preds[sel] == 0)).sum()),
                        "mean_prob": float(probs[sel].mean())}
        all_det[tag] = det

        with open(OUT / f"eval_{tag}.json", "w") as f:
            json.dump(m, f, indent=2)
        pd.DataFrame(det).T.reset_index().rename(columns={"index": "method"}).to_csv(
            OUT / f"per_method_{tag}.csv", index=False)
        np.savez_compressed(OUT / f"eval_{tag}.npz", preds=preds, probs=probs, labels=labels,
                            methods=methods, identities=identities)

    # ---- faceswap per-identity detail (only if v3 exists) ----
    if "v3" in per_ckpt:
        npz = np.load(OUT / "eval_v3.npz", allow_pickle=True)
        fs = npz["methods"] == "faceswap"
        detail = pd.DataFrame({
            "identity": npz["identities"][fs],
            "pred": npz["preds"][fs],
            "prob": npz["probs"][fs],
            "label": npz["labels"][fs],
        }).sort_values("identity")
        detail.to_csv(OUT / "faceswap_detail.csv", index=False)
        print("\nfaceswap per-identity (v3):")
        for _, row in detail.iterrows():
            ok = "OK" if row.pred == row.label else "MISS"
            print(f"  {row.identity:<18} label={int(row.label)} pred={int(row.pred)} prob={row.prob:.3f} {ok}")

    # ---- comparison table ----
    if len(per_ckpt) > 1:
        rows = []
        for mth in sorted(all_det[args.tags[0]], key=lambda k: all_det[args.tags[0]][k]["det_rate"]):
            r0 = all_det[args.tags[0]][mth]
            r1 = all_det.get(args.tags[-1], {}).get(mth)
            if r1 is None:
                continue
            rows.append({"method": mth, "n": r0["n"],
                         f"det_{args.tags[0]}": r0["det_rate"], f"det_{args.tags[-1]}": r1["det_rate"],
                         "delta_det": r1["det_rate"] - r0["det_rate"]})
        rows.sort(key=lambda r: r["delta_det"])
        pd.DataFrame(rows).to_csv(OUT / "comparison_methods.csv", index=False)
        with open(OUT / "comparison.json", "w") as f:
            json.dump({"per_ckpt": per_ckpt, "rows": rows}, f, indent=2)
        print(f"\nPer-method detection-rate ({args.tags[0]} -> {args.tags[-1]}):")
        for r in rows:
            flag = "▼" if r["delta_det"] < -0.01 else ("▲" if r["delta_det"] > 0.01 else " ")
            print(f"  {flag} {r['method']:<18} {r[f'det_{args.tags[0]}']*100:5.1f}% -> {r[f'det_{args.tags[-1]}']*100:5.1f}% ({r['delta_det']*100:+.1f}pp)")
    print("\nDone. Artifacts in", OUT)


if __name__ == "__main__":
    main()
