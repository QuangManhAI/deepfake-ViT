"""Xây real pool cho finetune từ eval-TRAIN của test_data_v3 (leak-free).

Tái lập đúng identity_disjoint_split(manifest, train_ratio=0.7, seed=42) mà
eval dùng → eval-train = 70% identity đầu. Chỉ lấy ảnh REAL thuộc eval-train
(cdc + ffc), hard-link vào out/<identity>/<frame>.png. Identity eval-train
KHÔNG xuất hiện trong eval-test nên không leak.

Real pool nhỏ (~824 ảnh) nhưng đủ để LoRA học "FF++-real là real" — khắc phục
shortcut "FF++-look = fake" (nguyên nhân sụp real/ffc).

Chạy:
  .venv/bin/python scripts/make_train_real.py \
      --root test_data_v3 --out data_train_real_evaltrain
"""
import argparse
import csv
import os
import random
import shutil


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def identity_disjoint_split(items, train_ratio, seed):
    """(item, identity) list → chia train/test theo identity, trả về eval-train set."""
    rng = random.Random(seed)
    groups = {}
    for it, ident in items:
        groups.setdefault(ident, []).append(it)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_tr = max(1, int(len(keys) * train_ratio))
    return set(keys[:n_tr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="test_data_v3")
    ap.add_argument("--out", default="data_train_real_evaltrain")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = os.path.join(args.root, "manifest.csv")
    items = []  # (abs_path, identity, domain)
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == "real":
                items.append((os.path.join(args.root, r["path"]),
                              r["identity"], r["domain"]))

    tr_ident = identity_disjoint_split(
        [(p, i) for p, i, _d in items], args.train_ratio, args.seed)
    print(f"real tổng: {len(items)} | eval-train identity: {len(tr_ident)}")

    n = 0
    by_dom = {"cdc": 0, "ffc": 0}
    for p, ident, dom in sorted(items, key=lambda t: t[1]):
        if ident not in tr_ident:
            continue
        dst = os.path.join(args.out, ident, os.path.basename(p))
        link(p, dst)
        by_dom[dom] = by_dom.get(dom, 0) + 1
        n += 1
    print(f"Đã hard-link {n} ảnh real eval-train vào {args.out}/  "
          f"(cdc={by_dom.get('cdc',0)}, ffc={by_dom.get('ffc',0)})")
    print(f"   identity mẫu: {sorted(tr_ident)[:5]} ...")
    print("done")


if __name__ == "__main__":
    main()
