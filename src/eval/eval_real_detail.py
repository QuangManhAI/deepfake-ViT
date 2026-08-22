"""Test CHI TIẾT trên REAL: model nhầm real nào, thuộc nguồn/method nào.

Tái lập split video-disjoint (seed 42, train_ratio 0.7) để map feature test
về path gốc, rồi phân tích phần REAL (label 0):
  - real acc theo NGUỒN (YouTube-real / Celeb-real / CelebA-FE)
  - real acc theo FOLDER method
  - danh sách ảnh real bị nhầm (false positive) nhiều nhất

Chạy:
  .venv/bin/python src/eval/eval_real_detail.py --model vit --tag test_data
  .venv/bin/python src/eval/eval_real_detail.py --model cnn --tag test_data
"""
import argparse
import csv
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT_DIM = {"vit": 384, "cnn": 768}
CACHE_DIR = "experiments/results/features"
ROOT = "test_data"
MANIFEST = "test_data/manifest.csv"


def read_manifest():
    items = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(ROOT, row["path"]),
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


def load(tag, key, suffix):
    mm = os.path.join(CACHE_DIR, f"{tag}_{key}{suffix}.mmap")
    meta = np.load(mm + ".meta.npz")
    X = np.memmap(mm, dtype=np.float32, mode="r", shape=(meta["labels"].shape[0],
                                                          FEAT_DIM[key]))
    return X, meta["labels"], meta["methods"]


FE_FOLDERS = {"stargan", "starganv2", "styleclip"}


def src_of(video, folder):
    """Nguồn real: FE-CelebA (3 method unknown) / Celeb-real / YouTube-real."""
    import re
    if folder in FE_FOLDERS:
        return "FE-CelebA"
    if re.match(r"^id\d+_\d+$", video):     # id6_0002 -> Celeb-real
        return "Celeb-real"
    return "YouTube-real"                     # 5-digit -> YouTube-real


def folder_of(path):
    """Method folder mà ảnh real này thuộc (test_data/<folder>/real/...)."""
    rel = path[len(ROOT) + 1:]
    return rel.split("/")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "cnn"])
    ap.add_argument("--tag", default="test_data")
    args = ap.parse_args()

    X_tr, y_tr, _ = load(args.tag, args.model, "")
    X_te, y_te, m_te = load(args.tag, args.model, "_test")

    # tái lập split -> test items (để map feature về path)
    items = read_manifest()
    train, test = video_disjoint_split(items, 0.7, seed=42)
    # kiểm tra alignment: label từ cache == label từ items (cùng thứ tự)
    labels_te = np.array([it[3] for it in test])
    assert len(labels_te) == len(y_te), f"len lệch {len(labels_te)} vs {len(y_te)}"
    assert (labels_te == y_te).all(), "thứ tự test không khớp — kiểm tra split"
    print(f"Alignment OK — {len(test):,} test items")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=42))
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # chỉ xét REAL
    real_idx = np.where(y_te == 0)[0]
    n_real = len(real_idx)
    real_acc = float((y_pred[real_idx] == 0).mean())
    fp_idx = [i for i in real_idx if y_pred[i] == 1]
    print(f"\n=== {args.model.upper()} — REAL ({n_real:,} ảnh) ===")
    print(f"Real acc={real_acc:.4f} | false positive={len(fp_idx)} ({len(fp_idx)/n_real:.1%})")

    # theo NGUỒN
    from collections import defaultdict
    by_src = defaultdict(lambda: [0, 0])     # src -> [đúng, tổng]
    for i in real_idx:
        s = src_of(test[i][2], folder_of(test[i][0]))
        by_src[s][1] += 1
        by_src[s][0] += int(y_pred[i] == 0)
    print(f"\nTheo nguồn real:")
    for s, (ok, n) in sorted(by_src.items(), key=lambda x: -x[1][1]):
        print(f"  {s:14s} {n:5d} ảnh  acc={ok/n:.4f}  FP={n-ok}")

    # theo METHOD FOLDER
    by_fold = defaultdict(lambda: [0, 0])
    for i in real_idx:
        f = folder_of(test[i][0])
        by_fold[f][1] += 1
        by_fold[f][0] += int(y_pred[i] == 0)
    print(f"\nTheo folder method:")
    for f, (ok, n) in sorted(by_fold.items(), key=lambda x: x[0]):
        print(f"  {f:12s} {n:5d} ảnh  acc={ok/n:.4f}  FP={n-ok}")

    # ví dụ nhầm
    print(f"\nVí dụ 10 ảnh real bị nhầm thành fake:")
    for i in fp_idx[:10]:
        print(f"  {test[i][0]}")


if __name__ == "__main__":
    main()
