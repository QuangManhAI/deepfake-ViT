#!/usr/bin/env python3
"""Trích subset từ zip DF40_train LOCAL (không cần tải lại từ HF Hub).

Tương đương extract_df40_train_subset.py nhưng đọc thẳng từ file zip trong
data/raw/DF40_train/ (đã tải về). Với mỗi method:
  - liệt kê member trong zip, lọc frame ảnh, nhóm theo cặp (pair)
  - chọn ngẫu nhiên `--pairs` cặp (seed cố định, hỗ trợ resume)
  - giải nén frame của từng cặp vào out/<method>/fake/<pair>/

zipfile.read() tự kiểm tra CRC → file lỗi sẽ báo BadZipFile.

Output `_extract_manifest.json` cùng format để build_data_train_finetune.py dùng.

Cách dùng:
  .venv/bin/python src/data/extract_df40_local.py \
      --zip-dir data/raw/DF40_train --out data_train_local \
      --pairs 30 --seed 42 \
      --methods faceswap facedancer inswap fsgan simswap blendface \
               pixart DiT uniface SiT lia mobileswap MRAA e4s
"""
import argparse
import json
import os
import random
import re
import zipfile

FRAME_RE = re.compile(r"\.(png|jpg|jpeg)$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", default="data/raw/DF40_train")
    ap.add_argument("--out", default="data_train_local")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    for mth in args.methods:
        zpath = os.path.join(args.zip_dir, f"{mth}.zip")
        if not os.path.exists(zpath):
            print(f"[{mth}] !! thiếu zip: {zpath}", flush=True)
            continue
        zf = zipfile.ZipFile(zpath)
        frames = [i for i in zf.infolist() if FRAME_RE.search(i.filename)]
        # Nhóm frame theo full dir path — chịu được nhiều layout zip khác nhau:
        #   <mth>/frames/<pair>/<frame>  (faceswap, fsgan, ...)
        #   <mth>/<pair>/<frame>         (DiT, SiT, pixart)
        #   frames/<pair>/<frame>        (uniface)
        by_dir = {}
        for i in frames:
            d = "/".join(i.filename.split("/")[:-1])
            by_dir.setdefault(d, []).append(i)
        if not by_dir:
            print(f"[{mth}] !! không có cặp nào", flush=True)
            zf.close()
            continue
        dirs = sorted(by_dir)
        chosen_dirs = rng.sample(dirs, min(args.pairs, len(dirs)))
        chosen = [d.split("/")[-1] for d in chosen_dirs]
        n_imgs = n_done = 0
        for d in chosen_dirs:
            out_pair = os.path.join(args.out, mth, "fake", d.split("/")[-1])
            if os.path.isdir(out_pair) and os.listdir(out_pair):
                # đã extract (resume) — đếm lại, không giải nén lại
                n_done += len(os.listdir(out_pair))
                continue
            os.makedirs(out_pair, exist_ok=True)
            for zi in by_dir[d]:
                data = zf.read(zi)
                fn = os.path.join(out_pair, os.path.basename(zi.filename))
                with open(fn, "wb") as f:
                    f.write(data)
                n_imgs += 1
        zf.close()
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
