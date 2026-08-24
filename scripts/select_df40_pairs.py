"""Select pairs fake từ DF40_train ĐÃ GIẢI NÉN (shared /workspace/data) → data_train_local.

Sau khi unzip toàn bộ zips vào shared root, script này:
  - với mỗi method, gom ảnh theo thư mục cha (pair) — chịu mọi layout
    (faceswap/frames/<pair>/..., DiT/<id>/..., StyleGAN2/<id>/seed*, RDDM/cdf/.../<id>/)
  - chọn min(pairs, có) cặp (seed 42), cap <= max_frames_per_pair frame/pair
    (uniform sampling) để train set không bị phình bởi method sinh hàng nghìn frame/cặp
  - hard-link frame vào out/<method>/fake/<pair>/ + ghi _extract_manifest.json
    đúng format build_data_train_finetune.py cần.

Chạy:
  .venv/bin/python scripts/select_df40_pairs.py \
      --src /workspace/data/DF40_train_extracted --out data_train_local \
      --pairs 24 --max-frames-per-pair 32 \
      --methods faceswap fsgan inswap simswap blendface uniface mobileswap \
                MRAA e4s lia facedancer DiT SiT RDDM StyleGAN2 StyleGAN3 \
                StyleGANXL VQGAN ddim sd2.1 pixart wav2lip sadtalker mcnet
"""
import argparse
import json
import os
import random
import re
import shutil

FRAME_RE = re.compile(r"\.(png|jpg|jpeg)$", re.IGNORECASE)


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
    ap.add_argument("--src", default="/workspace/data/DF40_train_extracted")
    ap.add_argument("--out", default="data_train_local")
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--max-frames-per-pair", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    for mth in args.methods:
        mdir = os.path.join(args.src, mth)
        if not os.path.isdir(mdir):
            print(f"[{mth}] !! thiếu thư mục: {mdir}", flush=True)
            continue
        # gom ảnh theo dir cha
        by_dir = {}
        for root, _dirs, files in os.walk(mdir):
            imgs = sorted(f for f in files if FRAME_RE.search(f))
            if imgs:
                by_dir.setdefault(root, imgs)
        if not by_dir:
            print(f"[{mth}] !! không có ảnh", flush=True)
            continue
        dirs = sorted(by_dir)
        chosen_dirs = rng.sample(dirs, min(args.pairs, len(dirs)))
        chosen = [os.path.basename(d) for d in chosen_dirs]
        n_imgs = n_done = 0
        for d in chosen_dirs:
            pair = os.path.basename(d)
            out_pair = os.path.join(args.out, mth, "fake", pair)
            if os.path.isdir(out_pair) and os.listdir(out_pair):
                n_done += len(os.listdir(out_pair))
                continue  # resume
            imgs = sorted(by_dir[d])
            if len(imgs) > args.max_frames_per_pair:
                # uniform sample để giữ tần suất đều, seed cố định
                idx = [int(round(i * (len(imgs) - 1) / (args.max_frames_per_pair - 1)))
                       for i in range(args.max_frames_per_pair)]
                imgs = [imgs[i] for i in sorted(set(idx))]
            os.makedirs(out_pair, exist_ok=True)
            for fn in imgs:
                link(os.path.join(d, fn), os.path.join(out_pair, fn))
            n_imgs += len(imgs)
        manifest[mth] = {
            "pairs": chosen,
            "n_images": n_imgs + n_done,
            "n_total_pairs": len(by_dir),
        }
        print(f"[{mth}] xong: {len(chosen)} cặp, {n_imgs + n_done} ảnh "
              f"(tổng {len(by_dir)} cặp)", flush=True)

    with open(os.path.join(args.out, "_extract_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(v["n_images"] for v in manifest.values())
    print(f"\nXONG: {total} ảnh fake trong {args.out}")


if __name__ == "__main__":
    main()
