#!/usr/bin/env python3
"""Prep dữ liệu cho finetune Plus trên 129K (v5_weakfix_v3).

- Đọc data/train_v5_weakfix_v3.csv (path /workspace/data/...) → rewrite sang path local
  (data/train_images/...), drop dòng thiếu file.
- Tính identity key từ cấu trúc path:
    * FaceForensics++ Real  → ff/<vid>
    * Celeb-DF Real         → celebdf/<vid>           (từ filename "<vid>_<f>.png")
    * celebvhq_real         → celebvhq/<vid>          (từ filename "<vid>_...")
    * DF40_train_extracted  → <method>/<id-dir>       (frames/<id> hoặc <id>)
    * còn lại               → unique theo filename
- Chia identity-disjoint: giữ lại ~5% nhóm identity (theo từng nhóm label+method) làm val.
- Ghi:
    data/splits/finetune_plus_full.csv   (đủ 129K, local paths)
    data/splits/finetune_plus_train.csv
    data/splits/finetune_plus_val.csv
"""
import csv, os, random, collections, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "train_v5_weakfix_v3.csv")
OUT = os.path.join(ROOT, "data", "splits")
IMG_DIR = os.path.join(ROOT, "data", "train_images")
VAL_FRAC = 0.05
VAL_MIN = 4000   # muốn val >= ~4K ảnh
rng = random.Random(42)


def identity_key(method, path):
    """Trả về key identity (group) cho một dòng."""
    if "/workspace/data/" in path:
        rel = path.split("/workspace/data/", 1)[1]
    elif "/workspace/hoangtuan/" in path:
        rel = path.split("/workspace/hoangtuan/", 1)[1]
    else:
        rel = path.lstrip("/")
    parts = rel.split("/")
    base = os.path.basename(rel)

    if method == "FaceForensics++ Real":
        # .../FaceForensics++/original_sequences/youtube/c23/frames/<vid>/<f>.png
        return "ff/" + parts[-2]
    if method == "Celeb-DF Real":
        # .../celeb_df_extracted/<vid>_<f>.png  (vid=00157)
        return "celebdf/" + base.split("_")[0]
    if method == "celebvhq_real":
        # .../celebvhq_frames/real/<vid>_<f>.jpg
        return "celebvhq/" + base.split("_")[0]
    if rel.startswith("DF40_train_extracted/"):
        # DF40_train_extracted/<method>/frames/<id>/<f>.png  hoặc /<method>/<id>/<f>.png
        # → identity = folder chứa frame (drop chỉ filename)
        return rel.rsplit("/", 1)[0]
    # deep-fake-face-swap, kaggle, df-40-test-full, test_data_v3 → mỗi ảnh 1 identity
    return method + "/" + base


def main():
    rows = []          # (local_path, label, method, domain, identity)
    missing = 0
    with open(SRC, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row["path"]
            # Ảnh lưu dưới data/train_images/<toàn bộ path từ /workspace...>
            local = os.path.join(IMG_DIR, p.lstrip("/"))
            if not os.path.exists(local):
                missing += 1
                continue
            label = int(row["label"])
            ident = identity_key(row["method"], p)
            rows.append((local, label, row["method"], row["domain"], ident))

    print(f"Tổng dòng CSV: 129884 | còn lại: {len(rows)} | thiếu file: {missing}")

    # dedup
    seen = set(); uniq = []
    for r_ in rows:
        if r_[0] in seen:
            continue
        seen.add(r_[0]); uniq.append(r_)
    print(f"Sau dedup path: {len(uniq)}")
    rows = uniq

    # ---- nhóm identity theo (label, method) để giữ tỷ lệ val ngang ngửa train ----
    groups_by_cat = collections.defaultdict(list)   # (label, method) -> [(ident, count, rows)]
    ident_rows = collections.defaultdict(list)       # ident -> [row...]
    for r_ in rows:
        ident_rows[r_[4]].append(r_)
    for ident, rws in ident_rows.items():
        lbl, meth = rws[0][1], rws[0][2]
        groups_by_cat[(lbl, meth)].append((ident, len(rws)))

    train_rows, val_rows = [], []
    for cat, groups in groups_by_cat.items():
        n_grp = len(groups)
        n_val_grp = max(1, round(VAL_FRAC * n_grp)) if n_grp > 3 else 0
        chosen = set(rng.sample(groups, n_val_grp)) if n_val_grp else set()
        chosen_idents = {g[0] for g in chosen}
        for ident, cnt in groups:
            dest = val_rows if ident in chosen_idents else train_rows
            dest.extend(ident_rows[ident])

    # nếu val quá nhỏ (do nhóm to), bổ sung thêm identity từ các category dày
    if len(val_rows) < VAL_MIN:
        need = VAL_MIN - len(val_rows)
        candidates = [r_ for r_ in train_rows if (r_[1], r_[2]) in {c for c, gs in groups_by_cat.items() if len(gs) > 3}]
        # shuffle deterministic, lấy đủ nhóm
        rng.shuffle(candidates)
        taken_ids = set()
        add = []
        for r_ in candidates:
            if r_[4] in taken_ids:
                continue
            taken_ids.add(r_[4]); add.append(r_)
            if len(val_rows) + len(add) >= VAL_MIN:
                break
        move_ids = {r_[4] for r_ in add}
        val_rows.extend(add)
        train_rows = [r_ for r_ in train_rows if r_[4] not in move_ids]

    def dump(name, data):
        p = os.path.join(OUT, name)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "method", "domain", "identity"])
            for r_ in data:
                w.writerow(r_)
        return p

    full_p = os.path.join(OUT, "finetune_plus_full.csv")
    with open(full_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "method", "domain", "identity"])
        for r_ in rows:
            w.writerow(r_)

    train_p = dump("finetune_plus_train.csv", train_rows)
    val_p = dump("finetune_plus_val.csv", val_rows)

    def stats(data):
        n = len(data)
        real = sum(1 for r_ in data if r_[1] == 0)
        fs = sum(1 for r_ in data if r_[1] == 1 and r_[2] == "faceswap")
        n_id = len({r_[4] for r_ in data})
        return n, real, n - real, fs, n_id

    for name, data in [("train", train_rows), ("val", val_rows)]:
        n, re, fa, fs, n_id = stats(data)
        print(f"{name:6s}: total={n:6d} real={re:6d} fake={fa:6d} faceswap={fs:6d} identity={n_id}")

    # kiểm tra identity không trùng giữa train/val
    train_ids = {r_[4] for r_ in train_rows}
    val_ids = {r_[4] for r_ in val_rows}
    overlap = train_ids & val_ids
    print(f"identity overlap train∩val: {len(overlap)}")
    assert not overlap, "LEAK identity giữa train/val!"

    print(f"\nĐã ghi:\n  {full_p}\n  {train_p}\n  {val_p}")


if __name__ == "__main__":
    main()
