"""Eval model finetuned trên CÙNG protocol identity-disjoint (test_data_v3).

So trực tiếp với baseline frozen+probe:
  - Cùng split: identity_disjoint_split(test_data_v3 manifest, train_ratio=0.7, seed=42)
  - Chỉ test trên identity test (chưa từng thấy trong train của baseline)
LƯU Ý: model finetuned train trên data_train (Celeb-real + 6 method DF40) — về lý
thuyết là identity-disjoint với test_data_v3 vì real train là id*_* (Celeb-real)
còn real test là cdc:* (Celeb-DF YouTube-real) + ff:*. Vẫn check trùng identity để chắc.

Báo: tổng (Acc/AUC/real acc/fake det) + per-(method,domain) detection rate —
so được với experiments/results/eval/identity_disjoint_v3_vit.json (baseline).

Chạy (trên mac mini sau khi finetune xong):
  .venv/bin/python src/eval/eval_finetuned_identity_disjoint.py
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import DinoViT
from src.models.lora import apply_lora

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
CKPT = "experiments/checkpoints/finetune/vit_finetuned.pt"
ROOT = "test_data_v3"


class BackboneClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


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
def evaluate(model, items, device, batch_size=64):
    model.eval()
    y_all, pred_all, prob_all, m_all, d_all = [], [], [], [], []
    bad = 0
    for i in tqdm(range(0, len(items), batch_size), desc="  Eval", ncols=90):
        batch = items[i:i + batch_size]
        xs, ys, ms, ds = [], [], [], []
        for p, mth, v, lab, ident, dom in batch:
            try:
                xs.append(EVAL_TF(Image.open(p).convert("RGB")))
                ys.append(lab); ms.append(mth); ds.append(dom)
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
        m_all.extend(ms); d_all.extend(ds)
        if device == "mps" and (i // batch_size) % 20 == 0:
            torch.mps.empty_cache()
    y = np.array(y_all); pred = np.array(pred_all); prob = np.array(prob_all)
    return y, pred, prob, np.array(m_all), np.array(d_all), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="experiments/results/eval/finetuned_identity_disjoint_v3.json")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"Device: {device} | ckpt: {args.ckpt}")

    ck = torch.load(args.ckpt, map_location="cpu")
    backbone = DinoViT(img_size=IMG_SIZE, gated_mlp=True)  # model-3 pretrained có gated_mlp=True
    if "lora_config" in ck:
        lc = ck["lora_config"]
        n = apply_lora(backbone, r=lc.get("r", 16), alpha=lc.get("alpha", 32.0),
                       targets=lc.get("targets", ("q_proj", "v_proj")))
        print(f"LoRA: rebuild {n} layer từ lora_config (r={lc.get('r')})")
    model = BackboneClassifier(backbone)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.ckpt} — epoch={ck.get('epoch')} | Params: {n_p/1e6:.1f}M")

    items = read_manifest(args.root)
    train, test, keys = identity_disjoint_split(items, args.train_ratio, seed=42)
    n_te_r = sum(1 for it in test if it[3] == 0)
    print(f"Manifest: {len(items):,} ảnh | {len(keys):,} identity keys "
          f"({len(keys)-int(len(keys)*0.7):,} test)")
    print(f"Test: {len(test):,} (real={n_te_r:,})")

    y, pred, prob, ms, ds, bad = evaluate(model, test, device, args.batch_size)
    print(f"  (bad decode={bad})")
    print(f"Acc={accuracy_score(y, pred):.4f} AUC={roc_auc_score(y, prob):.4f}")
    real_acc = float((pred[y == 0] == 0).mean())
    fake_det = float((pred[y == 1] == 1).mean())
    print(f"Real acc={real_acc:.4f} | Fake det={fake_det:.4f}")

    # per-(method, domain)
    per = {}
    for mth in sorted(set(ms)):
        if mth == "real":
            continue
        for dom in sorted(set(ds[ms == mth])):
            mask = (ms == mth) & (ds == dom)
            n = int(mask.sum())
            det = float((pred[mask] == 1).mean())
            per[f"{mth}/{dom}"] = {"n": n, "detection_rate": det}
    # real per domain
    per_real = {}
    rm = ms == "real"
    for dom in sorted(set(ds[rm])):
        mask = rm & (ds == dom)
        per_real[dom] = {"n": int(mask.sum()),
                         "acc_real": float((pred[mask] == 0).mean())}
    per["real"] = per_real

    print(f"\n  {'method/domain':18s}{'n':>6s}{'det':>8s}")
    for k, v in sorted(per.items()):
        if k == "real":
            for dom, rv in v.items():
                print(f"  {'real/'+dom:18s}{rv['n']:6d}{rv['acc_real']:8.3f}")
            continue
        print(f"  {k:18s}{v['n']:6d}{v['detection_rate']:8.3f}")

    out = {
        "protocol": "identity_disjoint",
        "root": args.root, "seed": 42, "train_ratio": args.train_ratio,
        "ckpt": args.ckpt, "epoch": ck.get("epoch"),
        "n_test": int(len(test)), "n_real_test": n_te_r, "bad": bad,
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "real_acc": real_acc, "fake_det": fake_det,
        "per_method_domain": per,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
