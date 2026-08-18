#!/usr/bin/env python
"""Build test_data: 12 method (4 nhóm x 3) cân bằng real=fake theo protocol DF40.

Nhóm / method:
  FS  (face-swap):      simswap, faceswap, blendface
  FR  (face-reenact):   wav2lip, sadtalker, fomm
  EFS (entire synth):   StyleGAN2, VQGAN, ddim
  FE  (face-edit):      stargan, starganv2, styleclip   [unknown — real/fake đi kèm]

Nguồn:
  real known (cdf): data/raw/real-root/Celeb-DF-v2/...          (đã trích theo JSON)
  fake known (cdf): /Volumes/quangmanh/Downloads/DF40/<method>/cdf/frames/
  unknown (FE)    : /Volumes/quangmanh/Downloads/DF40/<method>/{real,fake}/*.jpg

Cân bằng: mỗi method lấy N real + N fake (mặc định 1000), seed cố định theo method.
Manifest test_data/manifest.csv: method,video,path — real rows method="real"
(khớp eval_df40_all_methods.py: method=="real" -> label 0, còn lại -> label 1).

Chạy:
  .venv/bin/python src/data/build_test_data.py --dry-run
  .venv/bin/python src/data/build_test_data.py --n 1000
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

JSON_PREFIX = "deepfakes_detection_datasets/"
JSON_DIR = "data/dataset_json"
AIR = "/Volumes/quangmanh/Downloads/DF40"
CELEB_REAL = "data/raw/real-root"
DST = "test_data"
WORKERS = 8

GROUPS = {
    "FS":  ["simswap", "faceswap", "blendface"],
    "FR":  ["wav2lip", "sadtalker", "fomm"],
    "EFS": ["StyleGAN2", "VQGAN", "ddim"],
    "FE":  ["stargan", "starganv2", "styleclip"],
}
KNOWN = {m: g for g, ms in GROUPS.items() for m in ms if g != "FE"}
UNKNOWN = {m: g for g, ms in GROUPS.items() for m in ms if g == "FE"}


def air_path(json_path, method):
    """JSON path -> đường dẫn vật lý trên Air (cấu trúc lệch nhau theo method).

    - cdf thường:  .../<method>/cdf/frames/<vid>/<f>.png      (simswap, faceswap...)
    - EFS: JSON .../<method>/cdf/{Celeb-real,YouTube-real}/   -> Air .../cdf/Fake_from_{...}/
    - stargan nested: .../<method>/{real,fake}/<id>.jpg       -> Air .../{real,fake}/{real,fake}/<id>.jpg
    """
    rel = json_path[len(JSON_PREFIX):]
    assert rel.startswith("DF40/"), json_path
    rel = rel[len("DF40/"):]
    rel = rel.replace("cdf/Celeb-real/", "cdf/Fake_from_Celeb-real/")
    rel = rel.replace("cdf/YouTube-real/", "cdf/Fake_from_Youtube-real/")
    if method == "stargan":
        rel = rel.replace("/real/", "/real/real/")
        rel = rel.replace("/fake/", "/fake/fake/")
    return os.path.join(AIR, rel)


def celeb_real_path(json_path):
    """deepfakes_detection_datasets/Celeb-DF-v2/... -> data/raw/real-root/..."""
    return os.path.join(CELEB_REAL, json_path[len(JSON_PREFIX):])


def collect_side(method, json_key, kind):
    """(video, json_path) list cho 1 bên Real/Fake từ split 'test'."""
    d = json.load(open(os.path.join(JSON_DIR, json_key + ".json")))[json_key]
    sd = d[f"{method}_{kind}"]
    out = []
    for vid, info in sd["test"].items():
        for p in info["frames"]:
            out.append((vid, p))
    return out


def plan_known(method, n, rng):
    """Plan cho method known (cdf): (src, rel_dst, video) per class."""
    real = collect_side(method, f"{method}_cdf", "Real")
    fake = collect_side(method, f"{method}_cdf", "Fake")
    picks = {}
    for label, pool in (("real", real), ("fake", fake)):
        keep = rng.sample(pool, min(n, len(pool)))
        rows = []
        for vid, p in keep:
            if label == "real":
                src = celeb_real_path(p)      # data/raw/real-root/Celeb-DF-v2/...
            else:
                src = air_path(p, method)     # Air DF40/<method>/cdf/frames/...
            basename = os.path.basename(p)
            rel = f"{method}/{label}/{vid}/{basename}"
            rows.append((src, os.path.join(DST, rel), vid))
        picks[label] = rows
    return picks


def plan_unknown(method, n, rng):
    """Plan cho method unknown (FE): real/fake jpg phẳng trên Air."""
    real = collect_side(method, method, "Real")
    fake = collect_side(method, method, "Fake")
    picks = {}
    for label, pool in (("real", real), ("fake", fake)):
        keep = rng.sample(pool, min(n, len(pool)))
        rows = []
        for vid, p in keep:                   # p = .../DF40/<method>/real|<label>/<id>.jpg
            src = air_path(p, method)
            basename = os.path.basename(p)
            rel = f"{method}/{label}/{basename}"   # flat: mỗi ảnh 1 "video"
            rows.append((src, os.path.join(DST, rel), vid))
        picks[label] = rows
    return picks


def copy_all(todos):
    """Copy (src, dst) song song, trả (ok, missing)."""
    ok = missing = 0
    lock = threading_lock()

    def one(item):
        nonlocal ok, missing
        src, dst = item
        if os.path.exists(dst):              # đã có (re-run) -> bỏ qua
            with lock:
                ok += 1
            return
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            with lock:
                ok += 1
        else:
            with lock:
                missing += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ex.map(one, todos)
    return ok, missing


def threading_lock():
    import threading
    return threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="số ảnh mỗi class mỗi method")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = {**KNOWN, **UNKNOWN}
    all_rows = []          # manifest rows
    summary = []
    todos = []             # (src, dst) to copy

    for method, group in sorted(methods.items()):
        rng = random.Random(method)           # seed cố định theo tên method
        plan = plan_known(method, args.n, rng) if method in KNOWN \
            else plan_unknown(method, args.n, rng)
        n_real = len(plan["real"]); n_fake = len(plan["fake"])
        summary.append((method, group, n_real, n_fake))
        if args.dry_run:
            print(f"  [{group}] {method:10s} real={n_real:5d} fake={n_fake:5d}")
            continue
        for label, rows in plan.items():
            for src, dst, vid in rows:
                mlabel = "real" if label == "real" else method
                all_rows.append([mlabel, vid, os.path.relpath(dst, DST)])
                todos.append((src, dst))

    if args.dry_run:
        print(f"\nDry-run xong. {len(methods)} method, tổng "
              f"{sum(s[2] + s[3] for s in summary):,} ảnh. Bỏ --dry-run để copy.")
        return

    print(f"Copy {len(todos):,} ảnh vào {DST}/ ...")
    ok, missing = copy_all(todos)

    with open(os.path.join(DST, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "video", "path"])
        w.writerows(all_rows)

    print(f"\nCopy xong: ok={ok:,} missing={missing:,}")
    print(f"{'method':12s}{'nhóm':4s}{'real':>7s}{'fake':>7s}")
    for method, group, nr, nf in summary:
        print(f"{method:12s}{group:4s}{nr:7d}{nf:7d}")
    print(f"\nManifest: {DST}/manifest.csv ({len(all_rows):,} dòng)")
    print(f"Tổng: real={sum(s[2] for s in summary):,} fake={sum(s[3] for s in summary):,}")


if __name__ == "__main__":
    main()
