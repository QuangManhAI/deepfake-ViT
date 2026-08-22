"""Xây dựng subset DF40 CÂN BẰNG real/fake để test ViT vs CNN.

- Nguồn : $DF40_ROOT (mặc định data/raw/DF40)
- Đích  : data/df40_subset
- Fake  : mỗi method ~N ảnh (mặc định 1000), nhóm theo video để split video-disjoint.
- Real  : lấy CÂN BẰNG với fake (test không lệch về fake):
    * real riêng của 8 method có folder real/ (CollabDiff, MidJourney, deepfacelab,
      stargan, starganv2, styleclip, whichfaceisreal, heygen/heygen_new)
    * FF++ real (data/FaceForensics++/.../c23/frames) bù phần còn lại cho đủ tổng fake.
- Ghi manifest CSV (method, video, path). Real: method="real".

Chạy:
  .venv/bin/python src/data/build_df40_subset.py --dry-run        # xem kế hoạch
  .venv/bin/python src/data/build_df40_subset.py                  # copy thật
"""
import argparse
import csv
import os
import random
import shutil

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

SKIP_METHODS = {"__MACOSX", "mobileswap"}  # mobileswap còn dạng frames.zip

# Method mà fake/ (và real/) là ảnh phẳng, không nhóm theo video
FLAT_FAKE_METHODS = {"MidJourney", "stargan", "starganv2", "styleclip", "whichfaceisreal"}

# Method có fake/real nằm dưới đường dẫn riêng
FAKE_SUBDIR = {"CollabDiff": "fake", "deepfacelab": "fake", "heygen": "heygen_new/fake"}
REAL_SUBDIR = {"CollabDiff": "real", "deepfacelab": "real", "heygen": "heygen_new/real"}

REAL_METHODS = ["CollabDiff", "MidJourney", "deepfacelab", "stargan",
                "starganv2", "styleclip", "whichfaceisreal", "heygen"]


def list_images_video(root, rng, target, cap_video):
    """Đệ quy (thứ tự ngẫu nhiên, dừng sớm) -> dict {video_id: [abs_paths]}."""
    videos = {}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if count >= target:
            break
        dirnames[:] = [d for d in dirnames
                       if d not in ("__MACOSX", "landmarks", ".DS_Store")]
        rng.shuffle(dirnames)
        imgs = [os.path.join(dirpath, f) for f in filenames
                if f.lower().endswith(IMG_EXT)]
        if not imgs:
            continue
        vid = os.path.basename(os.path.normpath(dirpath))
        add = imgs if len(imgs) <= cap_video else rng.sample(imgs, cap_video)
        videos.setdefault(vid, []).extend(add)
        count += len(add)
    return videos


def pick_n(videos, n, seed):
    """Chốt ~n ảnh từ dict video->paths (shuffle video)."""
    rng = random.Random(seed)
    vids = sorted(videos.keys())
    rng.shuffle(vids)
    picked, count = {}, 0
    for v in vids:
        if count >= n:
            break
        pick = videos[v][: max(1, n - count)]
        picked[v] = pick
        count += len(pick)
    return picked


def sample_dir(base, n, cap_video, seed, force_flat=False):
    """Lấy ~n ảnh từ base. force_flat = ảnh phẳng, không nhóm video (không cap)."""
    rng = random.Random(seed)
    if force_flat:
        imgs = []
        for dp, _, fns in os.walk(base):
            imgs += [os.path.join(dp, f) for f in fns if f.lower().endswith(IMG_EXT)]
        rng.shuffle(imgs)
        imgs = imgs[:n]
        return {f"img_{i}": [p] for i, p in enumerate(imgs)}
    videos = list_images_video(base, rng, n, cap_video)
    if len(videos) < 8:
        imgs = [p for v in videos.values() for p in v]
        rng.shuffle(imgs)
        imgs = imgs[:n]
        return {f"img_{i}": [p] for i, p in enumerate(imgs)}
    return pick_n(videos, n, seed)


def sample_fake(root, method, n, cap_video, seed):
    if method in FLAT_FAKE_METHODS:
        return sample_dir(os.path.join(root, method, "fake"), n, cap_video, seed, force_flat=True)
    if method in FAKE_SUBDIR:
        return sample_dir(os.path.join(root, method, FAKE_SUBDIR[method]), n, cap_video, seed)
    return sample_dir(os.path.join(root, method), n, cap_video, seed)


def sample_own_real(root, method, n, cap_video, seed):
    """Real riêng của method (chỉ các method có folder real/)."""
    if method not in REAL_METHODS:
        return {}
    sub = REAL_SUBDIR.get(method, "real")
    base = os.path.join(root, method, sub)
    if not os.path.isdir(base):
        return {}
    flat = method in FLAT_FAKE_METHODS or method == "CollabDiff"
    return sample_dir(base, n, cap_video, seed, force_flat=flat)


def copy_items(src_items, method, src_base, dst_root, writer, dst_subdir=None):
    """Copy ảnh. dst_subdir (mặc định = method) để tránh đụng tên file giữa các nguồn real."""
    if dst_subdir is None:
        dst_subdir = method
    for _vid, paths in src_items.items():
        for p in paths:
            rel = os.path.relpath(p, src_base)
            dst = os.path.join(dst_root, dst_subdir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
            writer.writerow({"method": method, "video": _vid,
                             "path": os.path.relpath(dst, dst_root)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("DF40_ROOT", "data/raw/DF40"))
    ap.add_argument("--dst", default="data/df40_subset")
    ap.add_argument("--ffpp-real", default="data/FaceForensics++/original_sequences/youtube/c23/frames")
    ap.add_argument("--n", type=int, default=1000, help="số fake mỗi method")
    ap.add_argument("--cap-video", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()

    methods = sorted(m for m in os.listdir(args.src)
                     if os.path.isdir(os.path.join(args.src, m))
                     and m not in SKIP_METHODS and not m.startswith("."))
    if args.methods:
        want = {x.strip() for x in args.methods.split(",")}
        methods = [m for m in methods if m in want]
    print(f"Fake methods: {len(methods)}  n={args.n}  cap_video={args.cap_video}")
    print(f"Source: {args.src}\nTarget: {args.dst}  (dry_run={args.dry_run})")
    print("[skip] mobileswap (frames.zip chưa giải nén)")

    f, writer = None, None
    if not args.dry_run:
        os.makedirs(args.dst, exist_ok=True)
        manifest = os.path.join(args.dst, "manifest.csv")
        mode = "w" if not os.path.exists(manifest) else "a"
        f = open(manifest, mode, newline="")
        writer = csv.DictWriter(f, fieldnames=["method", "video", "path"])
        if mode == "w":
            writer.writeheader()

    # ---- Fake ----
    total_fake = 0
    for method in methods:
        picked = sample_fake(args.src, method, args.n, args.cap_video, args.seed)
        n_img = sum(len(v) for v in picked.values())
        total_fake += n_img
        print(f"  fake  {method:<20} videos={len(picked):>4}  images={n_img:>4}")
        if not args.dry_run and picked:
            copy_items(picked, method, os.path.join(args.src, method), args.dst, writer)

    # ---- Real: cân bằng với fake ----
    # 1) real riêng của các method có real/ (mỗi method tối đa ~n)
    total_real = 0
    real_sources = []
    for method in methods:
        if method not in REAL_METHODS:
            continue
        picked = sample_own_real(args.src, method, args.n, args.cap_video, args.seed)
        n_img = sum(len(v) for v in picked.values())
        if n_img:
            total_real += n_img
            real_sources.append((method, n_img))
            print(f"  real  {method:<20} videos={len(picked):>4}  images={n_img:>4}")
            if not args.dry_run:
                copy_items(picked, "real",
                           os.path.join(args.src, method, REAL_SUBDIR.get(method, "real")),
                           args.dst, writer, dst_subdir=f"real/{method}")

    # 2) FF++ real bù cho đủ tổng fake
    need = max(0, total_fake - total_real)
    ffpp_root = os.path.abspath(args.ffpp_real)
    if need > 0 and os.path.isdir(ffpp_root):
        rng = random.Random(args.seed)
        picked = pick_n(list_images_video(ffpp_root, rng, need + 500, args.cap_video), need, args.seed)
        n_img = sum(len(v) for v in picked.values())
        total_real += n_img
        print(f"  real  {'FF++ (bù)':<20} videos={len(picked):>4}  images={n_img:>4}")
        if not args.dry_run:
            copy_items(picked, "real", ffpp_root, args.dst, writer, dst_subdir="real/ffpp")
    else:
        print(f"  real  FF++ (bù)          need={need} — không cần hoặc thiếu nguồn")

    if f is not None:
        f.close()
    print(f"\nFake={total_fake:,}  Real={total_real:,}  (ratio {total_fake/max(1,total_real):.2f}:1)")
    if args.dry_run:
        print("Dry-run — chưa copy gì. Bỏ --dry-run để chạy thật.")
    else:
        print(f"Manifest: {os.path.join(args.dst, 'manifest.csv')}")


if __name__ == "__main__":
    main()
