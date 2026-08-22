#!/usr/bin/env python3
"""Gom subset train finetune: fake (từ extract_df40_train_subset) + real (Celeb-DF).

Tạo thư mục `data_train/`:
  data_train/<method>/fake/<pair>/<frame>.png   (hard-link từ extract dir)
  data_train/real/<identity>/<frame>.png
  data_train/train.csv, data_train/val.csv      (path tương đối từ project root)

Split train/val theo IDENTITY (cặp fake / video real) — không trộn lẫn identity.
Real lấy từ Celeb-real (identity id*_* — KHÔNG trùng test set vốn dùng cdc:/ff:).

Cách dùng:
  .venv/bin/python src/data/build_data_train_finetune.py \
      --extract data_train_local --out data_train \
      --real data/raw/real-root/Celeb-DF-v2/Celeb-real/frames \
      --n-real 1000 --seed 42
"""
import argparse
import csv
import json
import os
import random
import re
import shutil


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="data_train_local")
    ap.add_argument("--out", default="data_train")
    ap.add_argument("--real", default="data/raw/real-root/Celeb-DF-v2/Celeb-real/frames")
    ap.add_argument("--n-real", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-manifest", default="test_data_v3/manifest.csv")
    ap.add_argument("--no-leak-manifest", default="", help="manifest để loại fake pair chạm eval-test identity (protocol identity-disjoint seed 42)")
    ap.add_argument("--no-leak-train-ratio", type=float, default=0.7)
    ap.add_argument("--no-leak-seed", type=int, default=42)
    ap.add_argument("--ffc-real-manifest", default="", help="manifest test_data_v3 để thêm FF++-real thuộc EVAL-TRAIN split (không leak vào eval-test)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- tập identity real trong TEST (loại trừ khỏi train real) ----
    test_real_ids = set()
    if args.test_manifest and os.path.exists(args.test_manifest):
        with open(args.test_manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] == "0":
                    test_real_ids.add(row["identity"])
    print(f"real identity trong test: {len(test_real_ids)} (sẽ loại nếu trùng)")

    # ---- no-leak fake: loại pair có component chạm eval-test identity ----
    # Tái lập đúng protocol identity_disjoint_split(test_data_v3, train_ratio=0.7, seed=42)
    # → eval-test là 30% identity cuối. Pair fake chạm video FF++ trong eval-test bị loại.
    te_ffc = set()
    if args.no_leak_manifest and os.path.exists(args.no_leak_manifest):
        with open(args.no_leak_manifest, newline="") as f:
            mrows = list(csv.DictReader(f))
        m_items = [(r["path"], r["method"], r["video"], 0 if r["label"] == "0" else 1,
                    r["identity"], r["domain"]) for r in mrows]
        rng2 = random.Random(args.no_leak_seed)
        groups = {}
        for it in m_items:
            groups.setdefault(it[4], []).append(it)
        keys = sorted(groups.keys())
        rng2.shuffle(keys)
        n_tr = max(1, int(len(keys) * args.no_leak_train_ratio))
        te_keys = set(keys[n_tr:])
        # chỉ video FF++ (key dạng 'ffc:NNN') — số NNN là video thật
        te_ffc = {k.split(":")[-1] for k in te_keys if k.startswith("ffc:")}
        print(f"no-leak: eval-test có {len(te_ffc)} video FF++ -> loại pair chạm")
    elif args.no_leak_manifest:
        print(f"!! --no-leak-manifest không tồn tại: {args.no_leak_manifest}")

    # ---- sample real từ Celeb-real, theo identity ----
    real_dir = os.path.join(args.out, "real")
    ident_dirs = sorted(d for d in os.listdir(args.real)
                        if os.path.isdir(os.path.join(args.real, d)))
    # map identity dạng idX_YYYY → dùng nguyên bản làm identity local
    chosen_real = []
    rng.shuffle(ident_dirs)
    total = 0
    for ident in ident_dirs:
        if total >= args.n_real:
            break
        if f"cdc:{ident}" in test_real_ids:
            continue
        imgs = sorted(f for f in os.listdir(os.path.join(args.real, ident))
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not imgs:
            continue
        # lấy tối đa 60 ảnh/video để phủ nhiều identity
        take = imgs[:60]
        for fn in take:
            link(os.path.join(args.real, ident, fn),
                 os.path.join(real_dir, ident, fn))
        chosen_real.append(ident)
        total += len(take)
    print(f"real: {len(chosen_real)} video, {total} ảnh -> {real_dir}")

    # ---- fake từ extract dir ----
    with open(os.path.join(args.extract, "_extract_manifest.json")) as f:
        mani = json.load(f)
    fake_rows = []  # (method, pair)
    n_skip_leak = 0
    for mth, info in mani.items():
        pairs = info["pairs"]
        rng.shuffle(pairs)
        kept = []
        for pair in pairs:
            if te_ffc and any(c in te_ffc for c in re.findall(r"\d+", pair)):
                n_skip_leak += 1
                continue  # pair chạm eval-test identity -> loại
            src = os.path.join(args.extract, mth, "fake", pair)
            dst = os.path.join(args.out, mth, "fake", pair)
            for fn in os.listdir(src):
                link(os.path.join(src, fn), os.path.join(dst, fn))
            kept.append(pair)
        fake_rows.append((mth, kept))
    if n_skip_leak:
        print(f"no-leak: đã loại {n_skip_leak} fake pair chạm eval-test identity")

    # ---- real FF++ (eval-train) từ test_data_v3 ----
    # FF++-real nằm trong eval-TRAIN split (cùng seed/ratio với eval) — baseline CŨNG
    # dùng các identity này để train, nên thêm vào finetune là công bằng, không leak
    # vào eval-test (302 video còn lại). Bắt buộc để hết domain shortcut
    # "trông giống FF++ = fake" (fake train toàn FF++-style, real train toàn YouTube).
    n_ffc = 0
    ffc_dir = os.path.join(args.out, "real_ffc")
    if args.ffc_real_manifest and os.path.exists(args.ffc_real_manifest):
        with open(args.ffc_real_manifest, newline="") as f:
            mrows = list(csv.DictReader(f))
        m_items = [(r["path"], r["method"], r["video"], 0 if r["label"] == "0" else 1,
                    r["identity"], r["domain"]) for r in mrows]
        rng3 = random.Random(42)
        groups3 = {}
        for it in m_items:
            groups3.setdefault(it[4], []).append(it)
        keys3 = sorted(groups3.keys())
        rng3.shuffle(keys3)
        n_tr3 = max(1, int(len(keys3) * 0.7))
        tr_keys3 = set(keys3[:n_tr3])
        for r in mrows:
            if r["label"] == "0" and r["domain"] == "ffc" and r["identity"] in tr_keys3:
                src = os.path.join(os.path.dirname(args.ffc_real_manifest), r["path"])
                vid = r["identity"].split(":")[-1]
                link(src, os.path.join(ffc_dir, vid, os.path.basename(r["path"])))
                n_ffc += 1
        print(f"real_ffc (eval-train, disjoint eval-test): {n_ffc} ảnh -> {ffc_dir}")
    elif args.ffc_real_manifest:
        print(f"!! --ffc-real-manifest không tồn tại: {args.ffc_real_manifest}")
    n_fake_imgs = 0
    for mth, pairs in fake_rows:
        for pair in pairs:
            d = os.path.join(args.out, mth, "fake", pair)
            n_fake_imgs += len(os.listdir(d))
    print(f"fake: {n_fake_imgs} ảnh ({len(fake_rows)} method) -> {args.out}")

    # ---- train/val CSV (split theo identity) ----
    train_rows, val_rows = [], []
    # fake: 8 cặp/method đã shuffle → 6 train + 2 val
    for mth, pairs in fake_rows:
        n_tr = max(1, int(len(pairs) * 0.75))
        for i, pair in enumerate(pairs):
            d = os.path.join(mth, "fake", pair)
            imgs = sorted(os.listdir(os.path.join(args.out, d)))
            rows = [(os.path.join("data_train", d, fn), 1) for fn in imgs]
            (train_rows if i < n_tr else val_rows).extend(rows)
    # real: gom cả YouTube-real (real/) và FF++-real (real_ffc/), split theo video
    real_groups = []
    for sub in ("real", "real_ffc"):
        sub_full = os.path.join(args.out, sub)
        if not os.path.isdir(sub_full):
            continue
        for vid in sorted(os.listdir(sub_full)):
            imgs = sorted(os.listdir(os.path.join(sub_full, vid)))
            if imgs:
                real_groups.append((sub, vid, imgs))
    rng.shuffle(real_groups)
    n_tr = max(1, int(len(real_groups) * 0.8))
    for i, (sub, vid, imgs) in enumerate(real_groups):
        rows = [(os.path.join("data_train", sub, vid, fn), 0) for fn in imgs]
        (train_rows if i < n_tr else val_rows).extend(rows)

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for p, l in rows:
                w.writerow([p, l])

    write_csv(os.path.join(args.out, "train.csv"), train_rows)
    write_csv(os.path.join(args.out, "val.csv"), val_rows)
    print(f"\ntrain.csv: {len(train_rows)} | val.csv: {len(val_rows)}")
    print("done:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
