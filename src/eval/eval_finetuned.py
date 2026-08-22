"""Eval checkpoint finetuned (ViT-S/16 backbone + head) — so với frozen+probe.

- Test 1 (cùng-domain): test_data TEST split — baseline frozen+probe Acc=0.9575
- Test 2 (cross-domain): df40_subset TEST split — baseline frozen+probe Acc=0.766 (ViT)

Lưu ý: không biết chính xác finetune đã train trên ảnh nào (data/splits/*.csv đã xóa),
nên số test_data có thể bị lạc quan nếu trùng dữ liệu train của nó. Cross-domain là
thước đo khách quan hơn.

Chạy:
  .venv/bin/python src/eval/eval_finetuned.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import DinoViT

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
CKPT = "experiments/checkpoints/dinov3_finetuned.pt"


class BackboneClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def read_items(root, manifest):
    import csv
    items = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(root, row["path"]), row["method"],
                          row["video"], 0 if row["method"] == "real" else 1))
    return items


def video_disjoint_split(items, train_ratio, seed):
    import random
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


def decodable(p):
    try:
        Image.open(p).load()
        return True
    except Exception:
        return False


@torch.no_grad()
def evaluate(model, items, device, batch_size=64, label="eval"):
    model.eval()
    y_all, pred_all, prob_all, m_all = [], [], [], []
    bad = 0
    for i in tqdm(range(0, len(items), batch_size), desc=f"  {label}", ncols=90):
        batch = items[i:i + batch_size]
        xs, ys, ms = [], [], []
        for p, mth, v, lab in batch:
            try:
                xs.append(EVAL_TF(Image.open(p).convert("RGB")))
                ys.append(lab)
                ms.append(mth)
            except Exception:
                bad += 1
        if not xs:
            continue
        x = torch.stack(xs).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        y_all.extend(ys)
        pred_all.extend(logits.argmax(1).cpu().tolist())
        prob_all.extend(probs[:, 1].cpu().tolist())
        m_all.extend(ms)
    y = np.array(y_all); pred = np.array(pred_all); prob = np.array(prob_all)
    out = {
        "n": int(len(y)), "bad": bad,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "real_acc": float((pred[y == 0] == 0).mean()),
        "fake_det": float((pred[y == 1] == 1).mean()),
    }
    return out, (y, pred, prob, m_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"Device: {device}")

    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    # checkpoint là ViT-S/16 KHÔNG gated (gate_proj=0) — build thẳng kiến trúc
    backbone = DinoViT(img_size=IMG_SIZE)   # defaults = ViT-S/16, gated_mlp=False
    model = BackboneClassifier(backbone)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
    print(f"Loaded {CKPT} — epoch={ck.get('epoch')} strict OK")
    model.to(device).eval()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_p/1e6:.1f}M")

    # ---- Test 1: test_data (cùng split với baseline) ----
    items = read_items("test_data", "test_data/manifest.csv")
    _, te = video_disjoint_split(items, 0.7, seed=42)
    te = [it for it in te if decodable(it[0])]
    print(f"\n=== TEST 1: test_data TEST ({len(te):,} ảnh) — baseline Acc=0.9575 ===")
    r1, (y, pred, prob, ms) = evaluate(model, te, device, args.batch_size, "test_data")
    print(f"  Acc={r1['accuracy']:.4f} Prec={r1['precision']:.4f} Rec={r1['recall']:.4f} "
          f"F1={r1['f1']:.4f} AUC={r1['roc_auc']:.4f}")
    print(f"  Real acc={r1['real_acc']:.4f} | Fake det={r1['fake_det']:.4f}")
    per = {}
    for mth in sorted(set(ms)):
        mask = np.array(ms) == mth
        n = int(mask.sum())
        if mth == "real":
            per[mth] = {"n": n, "acc_real": float((pred[mask] == 0).mean())}
        else:
            per[mth] = {"n": n, "detection_rate": float((pred[mask] == 1).mean())}
    print(f"  {'method':12s}{'n':>6s}{'det':>8s}")
    for mth, v in sorted(per.items()):
        k = "acc_real" if mth == "real" else "detection_rate"
        print(f"  {mth:12s}{v['n']:6d}{v[k]:8.3f}")
    del y, pred, prob, ms
    torch.mps.empty_cache() if device == "mps" else None

    # ---- Test 2: df40_subset (cross-domain) ----
    items2 = read_items("data/df40_subset", "data/df40_subset/manifest.csv")
    _, te2 = video_disjoint_split(items2, 0.7, seed=42)
    te2 = [it for it in te2 if decodable(it[0])]
    print(f"\n=== TEST 2: df40_subset TEST ({len(te2):,} ảnh) — baseline Acc=0.766 ===")
    r2, (y2, pred2, prob2, ms2) = evaluate(model, te2, device, args.batch_size, "df40")
    print(f"  Acc={r2['accuracy']:.4f} Prec={r2['precision']:.4f} Rec={r2['recall']:.4f} "
          f"F1={r2['f1']:.4f} AUC={r2['roc_auc']:.4f}")
    print(f"  Real acc={r2['real_acc']:.4f} | Fake det={r2['fake_det']:.4f}")
    per2 = {}
    for mth in sorted(set(ms2)):
        mask = np.array(ms2) == mth
        n = int(mask.sum())
        if mth == "real":
            per2[mth] = {"n": n, "acc_real": float((pred2[mask] == 0).mean())}
        else:
            per2[mth] = {"n": n, "detection_rate": float((pred2[mask] == 1).mean())}
    print(f"  {'method':12s}{'n':>6s}{'det':>8s}")
    for mth, v in sorted(per2.items()):
        k = "acc_real" if mth == "real" else "detection_rate"
        print(f"  {mth:12s}{v['n']:6d}{v[k]:8.3f}")


if __name__ == "__main__":
    main()
