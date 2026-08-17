"""Đánh giá REAL và FAKE tách biệt trên test_data_v2 (identity-disjoint).

Cùng 1 probe (LR fit trên train cả 2 lớp), rồi đo TÁCH 2 phần test:
  1) REAL-ONLY  — model nhận real thật tới đâu: real acc, FPR, theo domain (cdc/ffc),
                  theo loại id (Celeb-real idN_M / YouTube-real 5-số)
  2) FAKE-ONLY  — phát hiện từng loại fake: detection theo method, theo domain
                  (cdc paired / ffc paired / efs tổng hợp / oth không ghép)

Tái dùng cache feature {tag}_{model}.mmap (đã extract) + tái lập split identity giống
eval_identity_disjoint (seed 42, 70/30 theo cột identity) để map feature về path.

Chạy:
  .venv/bin/python scripts/eval_real_fake_separate.py --model vit --tag test_data_v2
  .venv/bin/python scripts/eval_real_fake_separate.py --model cnn --tag test_data_v2
  (thêm --part real|fake để chỉ chạy 1 phần)
"""
import argparse
import csv
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT_DIM = {"vit": 384, "cnn": 768}
CACHE_DIR = "outputs/features"
ROOT = "test_data_v3"


def read_manifest():
    items = []  # (abs_path, method, video, label, identity, domain)
    with open(os.path.join(ROOT, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(ROOT, row["path"]),
                          row["method"], row["video"],
                          0 if row["method"] == "real" else 1,
                          row["identity"], row["domain"]))
    return items


def identity_disjoint_split(items, train_ratio, seed):
    """Split theo cột identity (giống eval_identity_disjoint)."""
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
    return train, test


def load(tag, key, suffix):
    mm = os.path.join(CACHE_DIR, f"{tag}_{key}{suffix}.mmap")
    meta = np.load(mm + ".meta.npz")
    X = np.memmap(mm, dtype=np.float32, mode="r",
                  shape=(meta["labels"].shape[0], FEAT_DIM[key]))
    return X, meta["labels"], meta["methods"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "cnn"])
    ap.add_argument("--tag", default="test_data_v2")
    ap.add_argument("--part", default="both", choices=["both", "real", "fake"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    items = read_manifest()
    train, test = identity_disjoint_split(items, 0.7, seed=42)
    X_tr, y_tr, _ = load(args.tag, args.model, "")
    X_te, y_te, _ = load(args.tag, args.model, "_test")

    assert len(y_tr) == len(train) and (y_tr == np.array([it[3] for it in train])).all(), \
        "train cache lệch"
    assert len(y_te) == len(test) and (y_te == np.array([it[3] for it in test])).all(), \
        "test cache lệch"
    print(f"Alignment OK — train={len(train):,} test={len(test):,}")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=42))
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    del X_tr, X_te

    methods = np.array([it[1] for it in test], dtype="U64")
    domains = np.array([it[5] for it in test], dtype="U8")
    videos = np.array([it[2] for it in test], dtype="U64")
    yte = y_te

    out = {"model": args.model, "tag": args.tag, "protocol": "identity-disjoint"}

    # ================= REAL-ONLY =================
    if args.part in ("both", "real"):
        mr = yte == 0
        n_r = int(mr.sum())
        ok_r = int((yp[mr] == 0).sum())
        real_acc = ok_r / n_r
        fpr = 1.0 - real_acc
        print(f"\n{'='*50}\n=== {args.model.upper()} — REAL-ONLY ({n_r:,} ảnh) ===")
        print(f"  Real acc={real_acc:.4f} (nhận đúng real) | FPR={fpr:.4f} (real bị lầm thành fake)")

        # theo domain
        print(f"\n  Theo domain:")
        dom_r = {}
        for d in sorted(set(domains[mr])):
            m = mr & (domains == d)
            n = int(m.sum())
            if n == 0:
                continue
            a = float((yp[m] == 0).mean())
            print(f"    {d:4s} {n:5d} ảnh  real_acc={a:.4f}  FP={n - int(yp[m].sum())}")
            dom_r[d] = {"n": n, "real_acc": round(a, 4)}
        out["real"] = {"n": n_r, "real_acc": round(real_acc, 4), "fpr": round(fpr, 4),
                       "per_domain": dom_r}

        # theo loại id (Celeb-real idN_M vs YouTube-real 5-số) trong cdc
        import re
        re5 = re.compile(r"\d{5}")
        reid = re.compile(r"id\d+_\d+")
        m_cdc = mr & (domains == "cdc")
        if m_cdc.any():
            print(f"\n  Celeb-DF (cdc) theo loại id:")
            for name, pat in [("Celeb-real (idN_M)", reid), ("YouTube-real (5-số)", re5)]:
                m2 = m_cdc & np.array([bool(pat.fullmatch(v)) if m_cdc[i] else False
                                       for i, v in enumerate(videos)])
                n = int(m2.sum())
                if n == 0:
                    continue
                a = float((yp[m2] == 0).mean())
                print(f"    {name:24s} {n:4d} ảnh  real_acc={a:.4f}")
                out["real"].setdefault("by_id_type", {})[name] = {"n": n, "real_acc": round(a, 4)}

    # ================= FAKE-ONLY =================
    if args.part in ("both", "fake"):
        mf = yte == 1
        n_f = int(mf.sum())
        det_all = float((yp[mf] == 1).mean())
        print(f"\n{'='*50}\n=== {args.model.upper()} — FAKE-ONLY ({n_f:,} ảnh) ===")
        print(f"  Detection overall={det_all:.4f}")

        # theo domain
        print(f"\n  Theo domain:")
        dom_f = {}
        for d in sorted(set(domains[mf])):
            m = mf & (domains == d)
            n = int(m.sum())
            if n == 0:
                continue
            a = float((yp[m] == 1).mean())
            print(f"    {d:4s} {n:5d} ảnh  det={a:.4f}")
            dom_f[d] = {"n": n, "detection_rate": round(a, 4)}
        out["fake"] = {"n": n_f, "detection_rate": round(det_all, 4),
                       "per_domain": dom_f}

        # theo method
        print(f"\n  Theo method (fake):")
        per_m = {}
        for m in sorted(set(methods[mf])):
            mask = mf & (methods == m)
            n = int(mask.sum())
            if n == 0:
                continue
            a = float((yp[mask] == 1).mean())
            d = domains[mf & (methods == m)][0]
            print(f"    {m:12s} {d:5s} {n:5d} ảnh  det={a:.4f}")
            per_m[m] = {"n": n, "domain": d, "detection_rate": round(a, 4)}
        out["fake"]["per_method"] = per_m

        # theo method + domain chi tiết (1 method có thể có cả cdf lẫn ff)
        print(f"\n  Theo (method, domain):")
        md = {}
        for m in sorted(set(methods[mf])):
            for d in sorted(set(domains[mf])):
                mask = mf & (methods == m) & (domains == d)
                n = int(mask.sum())
                if n == 0:
                    continue
                a = float((yp[mask] == 1).mean())
                print(f"    {m:12s} {d:4s} {n:5d} ảnh  det={a:.4f}")
                md[f"{m}/{d}"] = {"n": n, "detection_rate": round(a, 4)}
        out["fake"]["per_method_domain"] = md

        # PAIRED fake: chỉ fake thuộc identity có cả real+fake (matched frame)
        paired_ident = {}
        for it in test:
            paired_ident.setdefault(it[4], {"r": 0, "f": 0})
            paired_ident[it[4]]["r" if it[3] == 0 else "f"] += 1
        m_pair = mf & np.array([paired_ident[it[4]]["r"] > 0 for it in test])
        if m_pair.any():
            a = float((yp[m_pair] == 1).mean())
            print(f"\n  PAIRED fake (cùng người với real, matched frame) "
                  f"{int(m_pair.sum()):,} ảnh: det={a:.4f}")
            out["fake"]["paired_detection"] = {"n": int(m_pair.sum()),
                                               "detection_rate": round(a, 4)}

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
