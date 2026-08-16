"""Đánh giá 2 backbone DINOv3 (ViT-S/16 Plus vs ConvNeXt-Tiny, cùng ~28M params)
trên bộ Test-Challenge (DeepFakeFace challenge).

- Nhãn: đọc streaming từ Test-Challenge_poly.json (COCO-style). Ảnh "fake" nếu có
  ≥1 annotation category_id=1 (vùng bị chỉnh sửa), ngược lại là "real".
- Protocol: linear probe (frozen backbone + LogisticRegression balanced) huấn luyện
  trên train_insight, đánh giá trên Test-Challenge — giống report.
- RAM-safe: stream JSON theo chunk, 1 model tại 1 thời điểm, batch nhỏ, giải phóng
  model giữa 2 model.

Chạy:
  .venv/bin/python scripts/eval_challenge.py --labels-only
  .venv/bin/python scripts/eval_challenge.py --limit-train 2000 --limit-test 400   # smoke test
  .venv/bin/python scripts/eval_challenge.py                                       # full
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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
CHUNK = 16 * 1024 * 1024


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ---------------------------------------------------------------------------
# Labels from Test-Challenge_poly.json (streaming, RAM-safe)
# ---------------------------------------------------------------------------
def load_challenge_labels(json_path, img_dir):
    """Trả về list[(path, label)] cho mọi ảnh có trong img_dir.

    label = 1 (fake) nếu image_id có annotation category_id=1, ngược lại 0 (real).
    """
    with open(json_path, "rb") as f:
        head = f.read(8_000_000)
    images_section = head[: head.find(b'"annotations"')]
    objs = re.findall(
        rb'\{\s*"id":\s*(\d+),\s*"file_name":\s*"([^"]+)",\s*"width":\s*\d+,\s*"height":\s*\d+\s*\}',
        images_section,
    )
    basename_to_id = {os.path.basename(n.decode()): int(i) for i, n in objs}
    print(f"  JSON images: {len(basename_to_id)}")

    start = head.find(b'"annotations"')
    pat = re.compile(rb'"image_id":\s*(\d+).*?"category_id":\s*(\d+)', re.S)
    fake_ids = set()
    with open(json_path, "rb") as f:
        f.seek(start)
        buf = b""
        while True:
            c = f.read(CHUNK)
            if not c:
                break
            buf += c
            proc, buf = buf[:-1024], buf[-1024:]
            for m in pat.finditer(proc):
                if int(m.group(2)) == 1:
                    fake_ids.add(int(m.group(1)))
        for m in pat.finditer(buf):
            if int(m.group(2)) == 1:
                fake_ids.add(int(m.group(1)))
    print(f"  Fake image ids: {len(fake_ids)}")

    items = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        iid = basename_to_id.get(fn)
        if iid is None:
            continue
        items.append((os.path.join(img_dir, fn), 1 if iid in fake_ids else 0))
    return items


def load_csv(csv_path):
    items = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and os.path.exists(row[0]):
                items.append((row[0], int(row[1])))
    return items


@torch.no_grad()
def extract_features(model, items, device, batch_size, desc):
    model.to(device).eval()
    tf = build_transform()
    feats, labels = [], []
    for i in tqdm(range(0, len(items), batch_size), desc=desc, unit="batch"):
        batch = items[i : i + batch_size]
        imgs = [tf(Image.open(p).convert("RGB")) for p, _ in batch]
        x = torch.stack(imgs).to(device)
        f = model(x)
        feats.append(f.cpu().numpy())
        labels.extend(l for _, l in batch)
    return np.vstack(feats), np.array(labels)


def evaluate(X_train, y_train, X_test, y_test, seed=42):
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "detection_rate_fake": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_real": int((y_test == 0).sum()),
        "n_fake": int((y_test == 1).sum()),
        "real_correct": int(((y_pred == 0) & (y_test == 0)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/Test-Challenge_poly.json")
    ap.add_argument("--img-dir", default="data/Test-Challenge")
    ap.add_argument("--train-csv", default="data/splits/train_insight.csv")
    ap.add_argument("--vit-model", default="models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    ap.add_argument("--cnn-model", default="models/dinov3_next_cnn/model-2.safetensors")
    ap.add_argument("--output", default="outputs/results/challenge_report.json")
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    print(f"Device: {device}")

    test_items = load_challenge_labels(args.json, args.img_dir)
    real = sum(1 for _, l in test_items if l == 0)
    fake = sum(1 for _, l in test_items if l == 1)
    print(f"Test-Challenge available: {len(test_items)} (real={real}, fake={fake})")

    if args.labels_only:
        return

    train_items = load_csv(args.train_csv)
    if args.limit_train:
        r = [t for t in train_items if t[1] == 0][: args.limit_train // 2]
        fk = [t for t in train_items if t[1] == 1][: args.limit_train // 2]
        train_items = r + fk
    if args.limit_test:
        test_items = test_items[: args.limit_test]
    print(f"Train: {len(train_items)} | Test: {len(test_items)}")

    models = [
        {"name": "DINOv3 ViT-S/16 Plus", "key": "vit", "path": args.vit_model,
         "loader": load_dinov3, "kw": {"img_size": IMG_SIZE}},
        {"name": "DINOv3 ConvNeXt-Tiny", "key": "cnn", "path": args.cnn_model,
         "loader": load_dinov3_convnext, "kw": {}},
    ]

    report = {"config": {"device": device, "batch_size": args.batch_size,
                         "train_samples": len(train_items), "test_samples": len(test_items)},
              "test_class_balance": {"real": real, "fake": fake}, "models": {}}

    for cfg in models:
        print(f"\n→ {cfg['name']}")
        model = cfg["loader"](cfg["path"], **cfg["kw"])
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        t0 = time.time()
        Xtr, ytr = extract_features(model, train_items, device, args.batch_size, f"  Train features ({cfg['key']})")
        Xte, yte = extract_features(model, test_items, device, args.batch_size, f"  Test features  ({cfg['key']})")
        dt = time.time() - t0
        m = evaluate(Xtr, ytr, Xte, yte)
        print(f"  Params={n_params:.1f}M | Time={dt:.0f}s | Acc={m['accuracy']:.4f} "
              f"| DetRate(fake)={m['detection_rate_fake']:.4f} | AUC={m['roc_auc']:.4f} "
              f"| real correct={m['real_correct']}/{m['n_real']}")
        report["models"][cfg["key"]] = {"name": cfg["name"], "params_M": round(n_params, 1),
                                        "extract_time_s": round(dt, 1), "metrics": m}
        del model
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
