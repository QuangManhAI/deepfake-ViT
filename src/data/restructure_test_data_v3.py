"""Tái cấu trúc test_data_v2 -> test_data_v3: chia theo METHOD rõ ràng.

  test_data_v3/
    manifest.csv
    real/                      # 1,177 ảnh real (dùng chung cho mọi method)
    <method>/fake/             # 40 method, mỗi method 1 folder chứa fake của nó

Vì ảnh real DÙNG CHUNG giữa các method (cùng identity có fake từ nhiều method),
ta KHÔNG nhân bản real vào từng method (sẽ bị tính 40 lần khi eval) mà để real
ở top-level 1 lần duy nhất — vẫn giữ identity-unique như test_data_v2.

Dùng HARD LINK từ test_data_v2/ (cùng volume APFS -> 0 byte thêm, chạy tức thì).
Không xoá gì ở v2; có thể xoá v2 sau khi đã verify v3.

Chạy:
  .venv/bin/python src/data/restructure_test_data_v3.py
"""
import csv
import os
import sys

SRC = "test_data_v2"
OUT = "test_data_v3"


def main():
    src_manifest = os.path.join(SRC, "manifest.csv")
    assert os.path.exists(src_manifest), f"thiếu {src_manifest}"
    os.makedirs(OUT, exist_ok=True)

    rows = []
    with open(src_manifest) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            rows.append(dict(row))
    print(f"Đọc {len(rows):,} rows từ {src_manifest}")

    hardlink_fail = {"n": 0, "e": None}
    n_real = n_fake = 0
    for row in rows:
        v2_rel = row["path"]
        src_abs = os.path.join(SRC, v2_rel)
        method = row["method"]
        if method == "real":
            new_rel = os.path.join("real", os.path.basename(v2_rel))
        else:
            new_rel = os.path.join(method, "fake", os.path.basename(v2_rel))
        dst_abs = os.path.join(OUT, new_rel)
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        if not os.path.exists(dst_abs):
            try:
                os.link(src_abs, dst_abs)          # hard link — 0 byte thêm
            except OSError as e:
                hardlink_fail["n"] += 1
                hardlink_fail["e"] = e
                import shutil
                shutil.copy2(src_abs, dst_abs)     # fallback copy
        row["path"] = new_rel
        if method == "real":
            n_real += 1
        else:
            n_fake += 1

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---------- verify ----------
    import collections
    paths = [r["path"] for r in rows]
    dups = [p for p, c in collections.Counter(paths).items() if c > 1]
    methods = sorted(set(r["method"] for r in rows if r["method"] != "real"))
    missing = [p for p in paths if not os.path.exists(os.path.join(OUT, p))]
    print(f"\nOUT: {OUT}/")
    print(f"  real={n_real} | fake={n_fake} | tổng={len(rows):,}")
    print(f"  method (fake): {len(methods)} — path trùng={len(dups)} — thiếu file={len(missing)}")
    if dups[:5]:
        print("  dups:", dups[:5])
    if hardlink_fail["n"]:
        print(f"  [warn] {hardlink_fail['n']} file copy thay vì hard link ({hardlink_fail['e']})")
    if missing:
        print("  [LỖI] thiếu file:", missing[:5])
        sys.exit(1)
    print("OK — verify 1 ảnh mỗi method (decode)...")
    bad = 0
    from PIL import Image
    for m in methods:
        sample = next(r for r in rows if r["method"] == m)
        p = os.path.join(OUT, sample["path"])
        try:
            Image.open(p).convert("RGB")
        except Exception as ex:
            print(f"  [LỖI] {m}: {ex}")
            bad += 1
    s_real = next(r for r in rows if r["method"] == "real")
    try:
        Image.open(os.path.join(OUT, s_real["path"])).convert("RGB")
    except Exception as ex:
        print(f"  [LỖI] real: {ex}")
        bad += 1
    print(f"  decode lỗi: {bad}")


if __name__ == "__main__":
    main()
