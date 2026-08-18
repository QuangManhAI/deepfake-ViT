"""Protocol "người lạ" (unseen identity): gom REAL + FAKE cùng source video.

Thay vì split theo (method, video) như eval_df40_all_methods (cùng 1 video nguồn có thể
nằm cả train lẫn test), protocol này gom theo SOURCE VIDEO:
  - mọi real frame của source S (dù nằm trong folder method nào)
  - mọi fake frame sinh từ source S (dù method nào)
phải cùng vào 1 nhánh. Model chỉ được test trên source CHƯA từng thấy trong train.

Source key rút từ tên video:
  - real: idN_M (Celeb-real) / 5-số (YouTube-real) giữ nguyên
  - real FE (stargan/starganv2/styleclip): ảnh CelebA đơn lẻ, mỗi ảnh là 1 identity riêng
  - fake có idN_M giấu trong tên (simswap `id0_id16_0003` -> `id16_0003`, wav2lip `id7_0004_...`) -> lấy idN_M cuối
  - fake toàn số (StyleGAN2/VQGAN/ddim `00000`) -> quy về id0_<số> (best-effort)
  - fake khác (sadtalker `id27_<youtubeid>_...`) -> identity riêng (không gom được)

Tái dùng cache feature sẵn có; tái lập split CŨ để map feature về path.

Chạy:
  .venv/bin/python src/eval/eval_unseen_identity.py --model vit --tag test_data
  .venv/bin/python src/eval/eval_unseen_identity.py --model cnn --tag test_data
"""
import argparse
import csv
import json
import os
import random
import re

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT_DIM = {"vit": 384, "cnn": 768}
CACHE_DIR = "experiments/results/features"
ROOT = "test_data"
MANIFEST = "test_data/manifest.csv"
FE = {"stargan", "starganv2", "styleclip"}
IDN_M = re.compile(r"id\d+_\d+")


def read_manifest():
    items = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(ROOT, row["path"]),
                          row["method"], row["video"],
                          0 if row["method"] == "real" else 1))
    return items


def video_disjoint_split(items, train_ratio, seed):
    """Split CŨ (theo method,video) — chỉ để map feature về path."""
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


def load(tag, key, suffix):
    mm = os.path.join(CACHE_DIR, f"{tag}_{key}{suffix}.mmap")
    meta = np.load(mm + ".meta.npz")
    X = np.memmap(mm, dtype=np.float32, mode="r",
                  shape=(meta["labels"].shape[0], FEAT_DIM[key]))
    return X, meta["labels"], meta["methods"]


def folder_of(path):
    return path[len(ROOT) + 1:].split("/")[0]


def decodable(p):
    """Ảnh nào extract feature lúc trước decode được (cache loại ảnh hỏng)."""
    try:
        Image.open(p).load()
        return True
    except Exception:
        return False


def source_key(method, video, folder):
    """Source identity của 1 item. Hai item CÙNG key => phải cùng 1 nhánh split."""
    if method == "real":
        if folder in FE:
            return "FE-r:" + video          # ảnh CelebA đơn lẻ
        return video                         # idN_M (Celeb-real) / 5-số (YouTube-real)
    if folder in FE:
        return "FE-f:" + video              # fake CelebA: identity riêng
    ms = IDN_M.findall(video)
    if ms:
        return ms[-1]                        # simswap/wav2lip... -> source idN_M
    if re.fullmatch(r"\d+", video):
        return "id0_" + video                # EFS all-digit -> id0_<số> (best-effort)
    return "oth:" + video                    # sadtalker youtube-id: identity riêng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "cnn"])
    ap.add_argument("--tag", default="test_data")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    items = read_manifest()
    old_train, old_test = video_disjoint_split(items, 0.7, seed=42)
    # cache extract đã loại ảnh hỏng — loại lại cho khớp
    old_train = [it for it in old_train if decodable(it[0])]
    old_test = [it for it in old_test if decodable(it[0])]
    X_tr, y_tr, _ = load(args.tag, args.model, "")
    X_te, y_te, m_te = load(args.tag, args.model, "_test")

    # tái lập split cũ -> map feature về item
    lbl_tr = np.array([it[3] for it in old_train])
    lbl_te = np.array([it[3] for it in old_test])
    assert len(lbl_tr) == len(y_tr) and (lbl_tr == y_tr).all(), "train cache lệch"
    assert len(lbl_te) == len(y_te) and (lbl_te == y_te).all(), "test cache lệch"
    print(f"Alignment OK — train={len(old_train):,} test={len(old_test):,}")

    items_all = old_train + old_test
    X = np.concatenate([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    # method thật lấy từ items
    methods = np.array([it[1] for it in items_all], dtype="U64")
    del X_tr, X_te, y_tr, y_te, m_te

    # gom theo source key
    keys = [source_key(it[1], it[2], folder_of(it[0])) for it in items_all]
    grp = {}
    for i, k in enumerate(keys):
        grp.setdefault(k, []).append(i)
    uniq = sorted(grp)
    rng = random.Random(42)
    rng.shuffle(uniq)
    n_tr_keys = max(1, int(len(uniq) * 0.7))
    tr_key = set(uniq[:n_tr_keys])
    te_key = set(uniq[n_tr_keys:])
    tr_idx = np.array(sorted(i for k in tr_key for i in grp[k]))
    te_idx = np.array(sorted(i for k in te_key for i in grp[k]))

    n_r_tr = int((y[tr_idx] == 0).sum()); n_f_tr = int((y[tr_idx] == 1).sum())
    n_r_te = int((y[te_idx] == 0).sum()); n_f_te = int((y[te_idx] == 1).sum())
    print(f"\nUnseen-identity split: {len(uniq):,} source keys "
          f"({len(tr_key):,} train / {len(te_key):,} test)")
    print(f"  Train: {len(tr_idx):,} (real={n_r_tr:,} fake={n_f_tr:,})")
    print(f"  Test : {len(te_idx):,} (real={n_r_te:,} fake={n_f_te:,})")

    # coverage: bao nhiêu key có cả real lẫn fake (ghép được cặp)
    paired = sum(1 for k, idx in grp.items()
                 if any(items_all[i][1] == "real" for i in idx)
                 and any(items_all[i][1] != "real" for i in idx))
    n_paired = sum(len(idx) for k, idx in grp.items()
                   if any(items_all[i][1] == "real" for i in idx)
                   and any(items_all[i][1] != "real" for i in idx))
    print(f"  Key ghép được real+fake: {paired}/{len(uniq)} ({n_paired:,} ảnh)")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=42))
    clf.fit(X[tr_idx], y[tr_idx])
    yp = clf.predict(X[te_idx])
    pr = clf.predict_proba(X[te_idx])[:, 1]

    yte = y[te_idx]; mte = methods[te_idx]
    acc = accuracy_score(yte, yp)
    out = {
        "model": args.model, "tag": args.tag, "protocol": "unseen-identity",
        "n_keys": len(uniq), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "paired_keys": paired, "paired_images": n_paired,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision_score(yte, yp, zero_division=0)),
            "recall": float(recall_score(yte, yp, zero_division=0)),
            "f1": float(f1_score(yte, yp, zero_division=0)),
            "roc_auc": float(roc_auc_score(yte, pr)),
        },
    }
    print(f"\n=== {args.model.upper()} — UNSEEN IDENTITY ===")
    print(f"  Acc={out['metrics']['accuracy']:.4f} Prec={out['metrics']['precision']:.4f} "
          f"Rec={out['metrics']['recall']:.4f} F1={out['metrics']['f1']:.4f} "
          f"AUC={out['metrics']['roc_auc']:.4f}")

    mr = yte == 0; mf = yte == 1
    real_acc = float((yp[mr] == 0).mean()); fpr = float((yp[mr] == 1).mean())
    fake_det = float((yp[mf] == 1).mean())
    out["real_acc"] = real_acc; out["fake_detection"] = fake_det
    print(f"  Real acc={real_acc:.4f} (FPR={fpr:.4f}) | Fake detection={fake_det:.4f}")

    pm = {}
    print(f"  {'method':12s}{'n':>6s}{'det':>8s}")
    for mth in sorted(set(mte)):
        mask = mte == mth
        n = int(mask.sum())
        if n == 0:
            continue
        if mth == "real":
            det = float((yp[mask] == 0).mean())
            pm[mth] = {"n": n, "acc_real": det}
            print(f"  {'real':12s}{n:6d}{det:8.3f}")
        else:
            det = float((yp[mask] == 1).mean())
            pm[mth] = {"n": n, "detection_rate": det}
            print(f"  {mth:12s}{n:6d}{det:8.3f}")
    out["per_method"] = pm

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
