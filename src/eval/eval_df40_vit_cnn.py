"""So sánh ViT-S/16 Plus (28.7M) vs ConvNeXt-Tiny (27.8M) trên DF40 real/fake.

- Real  = FaceForensics++ (crop 256x256)
- Fake  = blendface (face-swap) + ddim (diffusion synthesis)
- Protocol: frozen backbone + LogisticRegression (linear probe).

CHIA THEO VIDEO (identity-disjoint): train/test tách theo video ID, KHÔNG video
nào xuất hiện cả 2 phía -> tránh data leakage.

Chạy:
  .venv/bin/python src/eval/eval_df40_vit_cnn.py
"""
import argparse
import glob
import json
import os
import random
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3
from src.models.dinov3_convnext import load_dinov3_convnext

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SOURCES = [
    ("data/FaceForensics++/original_sequences/youtube/c23/frames", 0, "real"),
    ("data/blendface-2/frames", 1, "blendface"),
    ("data/ddim", 1, "ddim"),
]


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def list_by_video(root):
    """Trả về dict {video_id: [image_paths]}. video_id = thư mục cha của ảnh."""
    d = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            vid = os.path.basename(os.path.dirname(p))
            d.setdefault(vid, []).append(p)
    return d


def split_videos(by_vid, train_ratio, seed):
    rng = random.Random(seed)
    vids = sorted(by_vid.keys())
    rng.shuffle(vids)
    n_train = max(1, int(len(vids) * train_ratio))
    return set(vids[:n_train]), set(vids[n_train:])


@torch.no_grad()
def extract_features(model, items, device, batch_size, desc):
    model.to(device).eval()
    tf = build_transform()
    feats, labels, sources = [], [], []
    for i in tqdm(range(0, len(items), batch_size), desc=desc, unit="batch"):
        batch = items[i : i + batch_size]
        imgs = [tf(Image.open(p).convert("RGB")) for p, _, _ in batch]
        x = torch.stack(imgs).to(device)
        f = model(x)
        feats.append(f.cpu().numpy())
        labels.extend(l for _, l, _ in batch)
        sources.extend(s for _, _, s in batch)
    return np.vstack(feats), np.array(labels), np.array(sources)


def evaluate(X_train, y_train, X_test, y_test, seed=42):
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "detection_rate_fake": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "cm": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }, y_pred


def per_source(y_pred, y_true, sources):
    out = {}
    for s in sorted(set(sources)):
        m = sources == s
        if s == "real":
            out[s] = {"n": int(m.sum()), "acc": float((y_pred[m] == y_true[m]).mean())}
        else:
            out[s] = {"n": int(m.sum()), "detection_rate": float((y_pred[m] == 1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--cap-train-real", type=int, default=3000)
    ap.add_argument("--cap-train-fake", type=int, default=1500)
    ap.add_argument("--cap-test-real", type=int, default=1500)
    ap.add_argument("--cap-test-fake", type=int, default=750)
    ap.add_argument("--vit-model", default="experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    ap.add_argument("--cnn-model", default="experiments/checkpoints/weights/dinov3_next_cnn/model-2.safetensors")
    ap.add_argument("--output", default="experiments/results/df40_vit_cnn_report.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    print(f"Device: {device}")

    # cap_train / cap_test theo từng source: real dùng cap_real, fake dùng cap_fake
    caps_train = {"real": args.cap_train_real, "blendface": args.cap_train_fake, "ddim": args.cap_train_fake}
    caps_test = {"real": args.cap_test_real, "blendface": args.cap_test_fake, "ddim": args.cap_test_fake}

    print("Build video-disjoint split:")
    train_items, test_items = [], []
    for root, label, name in SOURCES:
        by_vid = list_by_video(root)
        tr_vids, te_vids = split_videos(by_vid, args.train_ratio, seed=42)
        tr_paths = [p for v in tr_vids for p in by_vid[v]]
        te_paths = [p for v in te_vids for p in by_vid[v]]
        rng = random.Random(42)
        rng.shuffle(tr_paths); rng.shuffle(te_paths)
        tr = tr_paths[:caps_train[name]]; te = te_paths[:caps_test[name]]
        train_items += [(p, label, name) for p in tr]
        test_items += [(p, label, name) for p in te]
        print(f"  {name:<10} videos train={len(tr_vids)} test={len(te_vids)} | images train={len(tr)} test={len(te)}")
    rng = random.Random(42)
    rng.shuffle(train_items); rng.shuffle(test_items)

    n_tr_real = sum(1 for _, l, _ in train_items if l == 0)
    n_tr_fake = sum(1 for _, l, _ in train_items if l == 1)
    n_te_real = sum(1 for _, l, _ in test_items if l == 0)
    n_te_fake = sum(1 for _, l, _ in test_items if l == 1)
    print(f"Train: {len(train_items)} (real={n_tr_real}, fake={n_tr_fake})")
    print(f"Test : {len(test_items)} (real={n_te_real}, fake={n_te_fake})")

    if not train_items or not test_items:
        print("ERROR: no images found — this eval needs the RAW DF40 frame sources below,")
        for root, _label, name in SOURCES:
            print(f"  - {name:<10} {root}  (exists={os.path.isdir(root)})")
        raise SystemExit(
            "Raw DF40 sources not present. Use eval_identity_disjoint.py --root test_data_v3 "
            "to evaluate on the downloaded test set instead."
        )

    models = [
        {"name": "DINOv3 ViT-S/16 Plus", "key": "vit", "path": args.vit_model, "loader": load_dinov3, "kw": {"img_size": IMG_SIZE}},
        {"name": "DINOv3 ConvNeXt-Tiny", "key": "cnn", "path": args.cnn_model, "loader": load_dinov3_convnext, "kw": {}},
    ]

    report = {"config": {k: v for k, v in vars(args).items() if k in
                         ["train_ratio", "cap_train_real", "cap_train_fake", "cap_test_real", "cap_test_fake", "device"]},
              "split": {"train": {"real": n_tr_real, "fake": n_tr_fake}, "test": {"real": n_te_real, "fake": n_te_fake}},
              "models": {}}

    for cfg in models:
        print(f"\n→ {cfg['name']}")
        model = cfg["loader"](cfg["path"], **cfg["kw"])
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        t0 = time.time()
        Xtr, ytr, _ = extract_features(model, train_items, device, args.batch_size, f"  Train feat ({cfg['key']})")
        Xte, yte, src = extract_features(model, test_items, device, args.batch_size, f"  Test feat  ({cfg['key']})")
        dt = time.time() - t0
        m, y_pred = evaluate(Xtr, ytr, Xte, yte)
        ps = per_source(y_pred, yte, src)
        print(f"  Params={n_params:.1f}M | Time={dt:.0f}s")
        print(f"  Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
              f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} DetRate(fake)={m['detection_rate_fake']:.4f}")
        print(f"  Per-source: {ps}")
        report["models"][cfg["key"]] = {"name": cfg["name"], "params_M": round(n_params, 1),
                                        "extract_time_s": round(dt, 1), "metrics": m, "per_source": ps}
        del model
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
