"""Build DF40 subset CÂN BẰNG real/fake (per-method) từ ổ mount Air.

Chỉ lấy các method có BOTH real/ và fake/. Mỗi method: n real == n fake
(giới hạn bởi số ảnh có sẵn). Tổng ~ 12k ảnh cho test nhanh.

Nguồn  : $DF40_ROOT (mặc định data/raw/DF40)
Đích   : data/df40_subset
Manifest: method, video, path — real ảnh method="real", fake method=<tên method>.
          (eval_df40_all_methods.py: method=="real" -> label 0, else 1)

Chạy:
  .venv/bin/python src/data/build_df40_balanced.py --n 750 --dry-run
  .venv/bin/python src/data/build_df40_balanced.py --n 750
"""
import argparse
import csv
import os
import random
import shutil
import sys

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SKIP_DIRS = {"landmarks", "__MACOSX", ".DS_Store"}

# Method mà ảnh là ảnh độc lập (không nhóm theo video) — mỗi ảnh là 1 "video"
# để split video-disjoint không gộp cả method vào 1 nhóm (lỗi: toàn bộ vào train).
FLAT_METHODS = {"MidJourney", "stargan", "starganv2", "styleclip", "whichfaceisreal"}

SRC = os.environ.get("DF40_ROOT", "data/raw/DF40")
DST = "data/df40_subset"

# method -> (real_root, fake_root) relative to SRC
METHODS = {
    "CollabDiff":     ("CollabDiff/real",              "CollabDiff/fake"),
    "MidJourney":     ("MidJourney/real",              "MidJourney/fake"),
    "deepfacelab":    ("deepfacelab/real/frames",      "deepfacelab/fake/frames"),
    "stargan":        ("stargan/real",                 "stargan/fake"),
    "starganv2":      ("starganv2/real",               "starganv2/fake"),
    "styleclip":      ("styleclip/real",               "styleclip/fake"),
    "whichfaceisreal":("whichfaceisreal/real",         "whichfaceisreal/fake"),
    "heygen":         ("heygen/heygen_new/real",       "heygen/heygen_new/fake/frames"),
}


def list_images(root, flat=False):
    """Trả về list (video_id, abs_path). video_id = thư mục con trực tiếp;
    ảnh phẳng (không có subdir) -> video_id = img_<i>.
    flat=True: bỏ qua cấu trúc thư mục, mỗi ảnh là 1 video (dùng cho method
    mà subdir trùng tên như stargan/fake/fake/...)."""
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if flat:
            # mỗi ảnh là 1 video -> không gộp cả thư mục thành 1 nhóm
            for f in filenames:
                if f.lower().endswith(IMG_EXT):
                    out.append((f"img_{len(out)}", os.path.join(dirpath, f)))
            continue
        rel = os.path.relpath(dirpath, root)
        for f in filenames:
            if not f.lower().endswith(IMG_EXT):
                continue
            vid = rel.split(os.sep)[0] if rel != "." else f"img_{len(out)}"
            out.append((vid, os.path.join(dirpath, f)))
    return out


def sample(items, n, seed):
    """Chọn n ảnh, shuffle video để giữ đa dạng. Trả list (vid, path)."""
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    # shuffle theo video id để không lấy chồng 1 video
    by_vid = {}
    for vid, p in items:
        by_vid.setdefault(vid, []).append(p)
    vids = sorted(by_vid)
    rng.shuffle(vids)
    picked, count = [], 0
    for v in vids:
        if count >= n:
            break
        take = by_vid[v][: max(1, n - count)]
        picked.extend((v, p) for p in take)
        count += len(take)
    return picked


def copy_batch(items, dst_dir, writer, method_label):
    """Copy ảnh, ghi manifest. items = list (vid, abs_path)."""
    for vid, src_path in items:
        rel = os.path.relpath(src_path, SRC)
        dst = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)
        writer.writerow({"method": method_label, "video": vid,
                         "path": os.path.relpath(dst, DST)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=750, help="real/fake mỗi method")
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dst_root = args.dst

    print(f"Nguồn : {SRC}")
    print(f"Đích  : {dst_root}  (n={args.n}, dry_run={args.dry_run})")
    print(f"Methods ({len(METHODS)}): {', '.join(METHODS)}")

    if not args.dry_run:
        os.makedirs(dst_root, exist_ok=True)
        manifest = os.path.join(dst_root, "manifest.csv")
        mode = "w" if not os.path.exists(manifest) else "a"
        f = open(manifest, mode, newline="")
        writer = csv.DictWriter(f, fieldnames=["method", "video", "path"])
        if mode == "w":
            writer.writeheader()

    grand_real = grand_fake = 0
    for method, (rr, fr) in METHODS.items():
        flat = method in FLAT_METHODS
        real = sample(list_images(os.path.join(SRC, rr), flat=flat), args.n, 42)
        fake = sample(list_images(os.path.join(SRC, fr), flat=flat), args.n, 42)
        # mỗi method cân bằng: lấy cùng số lượng (min)
        m = min(len(real), len(fake))
        real, fake = real[:m], fake[:m]
        print(f"  {method:<16} real={len(real):>4}  fake={len(fake):>4}")
        grand_real += len(real)
        grand_fake += len(fake)
        if not args.dry_run and real:
            # fake -> <method>/fake/... ; real -> <method>/real/...
            copy_batch(fake, os.path.join(dst_root, method), writer, method)
            # real path giữ nguyên cấu trúc thư mục gốc (vd real/frames/...) => copy vào
            # <method>/real/... bằng cách remap relpath từ real_root
            rroot = os.path.join(SRC, rr)
            for vid, src_path in real:
                rel = os.path.relpath(src_path, rroot)
                dst = os.path.join(dst_root, method, "real", rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)
                writer.writerow({"method": "real", "video": vid,
                                 "path": os.path.relpath(dst, dst_root)})

    if not args.dry_run:
        f.close()
    print(f"\nTổng real={grand_real:,}  fake={grand_fake:,}  (ratio {grand_real/max(1,grand_fake):.2f}:1)")
    print(f"Tổng ảnh={grand_real + grand_fake:,}")
    if args.dry_run:
        print("Dry-run — chưa copy gì. Bỏ --dry-run để chạy thật.")


if __name__ == "__main__":
    main()
