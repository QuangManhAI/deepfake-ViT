#!/usr/bin/env python3
"""Build MAX-expansion finetune dataset cho 12 method yếu.

Fake (~98.9K):
  - df-40-test-full trừ test (exact-frame + video-level): styleclip 1824, stargan 984,
    starganv2 1001, deepfacelab 2342, heygen 838, whichfaceisreal 251, CollabDiff 250, MidJourney 252 (5 png + 247 gif)
  - DF40_train_extracted (train split sạch): faceswap 22.8K, sadtalker 22.8K, wav2lip 22.7K, MRAA 22.8K

Real (mặc định ~70.6K):
  - FF++ replay từ exp02 train splits (27.6K, video-level — user-approved)
  - celeb_df_extracted (25.9K)
  - df-40-test-full/<method>/real (17K — clean source của test split DF40; --no-real-dir để loại)

--no-real-dir: bỏ real/ folders. Dùng khi real/ gây boundary shift (max_v2 FN 33->154).
Balance: dataset giữ imbalance, script finetune dùng WeightedRandomSampler để cân bằng per-step.
Verification: 0 path trùng chính xác test manifest/test_balanced.
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/quangmanh/deepfake")
TEST_FULL = Path("/workspace/data/df-40-test-full")
DF40_TRAIN = Path("/workspace/data/DF40_train_extracted")
CELEB_REAL = Path("/workspace/data/hoangtuan_data/processed/celeb_df_extracted")
MANIFEST = Path("/workspace/data/test_data_v3/manifest.csv")
TEST_BALANCED = ROOT / "experiments/results/error_analysis_lora/test_balanced.csv"
OUT = ROOT / "data/finetune_exp02"
SEED = 42
IMG = (".png", ".jpg", ".jpeg")

# df-40-test-full: method -> max clean to use (full clean pool)
TESTFULL_CAPS = {
    "styleclip": 1824, "stargan": 984, "starganv2": 1001, "deepfacelab": 2342,
    "heygen": 838, "whichfaceisreal": 251, "CollabDiff": 250, "MidJourney": 252,  # 5 png + 247 gif
}
# methods whose fake pool also contains .gif frames (added to IMG exts)
GIF_METHODS = {"MidJourney"}
# DF40_train_extracted: method -> max clean (full pool)
DFTRAIN_CAPS = {"faceswap": 22852, "sadtalker": 22797, "wav2lip": 22682, "MRAA": 22811}
TEST_FULL_DIR = {"heygen": "heygen_new"}
EXP02_SPLITS = ["train_domain_balanced.csv", "train_balanced.csv", "train_v3_clean.csv"]


def walk_images(d, exts=IMG):
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(exts):
                yield Path(root) / fn


def norm_video(v):
    return Path(str(v)).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-real-dir", action="store_true",
                    help="bỏ real/ folders của df-40-test-full (real = FF++ replay + celeb only)")
    ap.add_argument("--out", type=str, default="train_max.csv", help="tên file output trong data/finetune_exp02/")
    args = ap.parse_args()

    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- test exclusion sets ----
    test_paths = set()
    test_videos = defaultdict(set)
    for mf in (MANIFEST, TEST_BALANCED):
        root = Path("/workspace/data/test_data_v3") if mf == MANIFEST else None
        with open(mf, newline="") as f:
            for r in csv.DictReader(f):
                if root is not None:
                    test_paths.add(str(root / r["path"]))
                else:
                    test_paths.add(r["path"])
                if r.get("label") == "1":
                    test_videos[r.get("method", "")].add(norm_video(r.get("video", "")))
    print(f"test paths: {len(test_paths)} | fake source tokens per method")

    rows = []
    report = {}

    # ---- fake: df-40-test-full (exact-frame/video-level exclusion) ----
    for method, cap in TESTFULL_CAPS.items():
        dirname = TEST_FULL_DIR.get(method, method)
        src = TEST_FULL / dirname / "fake"
        if not src.is_dir():
            print(f"!! {method}: missing {src}"); continue
        tokens = test_videos.get(method, set())
        exts = IMG + (".gif",) if method in GIF_METHODS else IMG
        cand = []
        for p in walk_images(src, exts=exts):
            if str(p) in test_paths or p.stem in tokens or p.parent.name in tokens:
                continue
            cand.append(p)
        sel = rng.sample(cand, min(cap, len(cand)))
        for p in sel:
            rows.append((str(p), 1, method))
        report[method] = {"source": f"df-40-test-full/{dirname}", "clean_pool": len(cand), "selected": len(sel)}
        print(f"  {method:<14} clean={len(cand):>6} selected={len(sel):>6}")

    # ---- fake: DF40_train_extracted (clean train split) ----
    for method, cap in DFTRAIN_CAPS.items():
        src = DF40_TRAIN / method
        cand = list(walk_images(src))
        # safety: drop any test-colliding path
        hit = sum(1 for p in cand if str(p) in test_paths)
        if hit:
            print(f"!! {method}: {hit} train frames in test — removing"); cand = [p for p in cand if str(p) not in test_paths]
        sel = rng.sample(cand, min(cap, len(cand)))
        for p in sel:
            rows.append((str(p), 1, method))
        report[method] = {"source": "DF40_train_extracted", "clean_pool": len(cand), "selected": len(sel)}
        print(f"  {method:<14} clean={len(cand):>6} selected={len(sel):>6}")

    # ---- real: FF++ replay (exp02 train splits) ----
    ffpp = set()
    for fn in EXP02_SPLITS:
        p = Path(f"/workspace/hoangtuan/deepfake-ViT/data/splits/{fn}")
        if not p.exists(): continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["label"] == "0" and ("/FaceForensics++/" in r["path"] or "/original_sequences/" in r["path"]):
                    ffpp.add(r["path"])
    ffpp = sorted(p for p in ffpp if p not in test_paths)
    for p in ffpp:
        rows.append((p, 0, "real_ffpp"))
    print(f"  real_ffpp : {len(ffpp)} (replay, video-level)")

    # ---- real: celeb_df_extracted ----
    celeb = sorted(str(p) for p in walk_images(CELEB_REAL) if str(p) not in test_paths)
    for p in celeb:
        rows.append((p, 0, "real_celeb"))
    print(f"  real_celeb: {len(celeb)}")

    # ---- real: df-40-test-full/<method>/real folders (clean, non-test) ----
    # cảnh báo: thêm 17K ảnh kiểu DF40 vào real pool làm boundary dịch về real
    # (max_v2 FN 33->154). Mặc định LOẠI — chỉ dùng khi cần thêm real.
    real_dir = {}
    if not args.no_real_dir:
        for d in sorted(TEST_FULL.iterdir()):
            r = d / "real"
            if not r.is_dir():
                continue
            method = TEST_FULL_DIR.get(d.name, d.name)
            fs = [str(p) for p in walk_images(r) if str(p) not in test_paths]
            for p in fs:
                rows.append((p, 0, f"real_{method}"))
            real_dir[method] = len(fs)
            print(f"  real_{method:<12}: {len(fs)}")
        print(f"  real/ folders total: {sum(real_dir.values())}")
    else:
        print("  real/ folders: SKIPPED (--no-real-dir)")

    # ---- verify ----
    sel_paths = {r[0] for r in rows}
    overlap = sel_paths & test_paths
    assert not overlap, f"LEAK: {len(overlap)} finetune paths in test!\n" + sorted(overlap)[0]
    n0 = sum(1 for r in rows if r[1] == 0)
    n1 = sum(1 for r in rows if r[1] == 1)
    print(f"\nVERIFY: 0 exact-path overlap ✓ | rows {len(rows)} (real {n0} / fake {n1})")

    rng.shuffle(rows)
    out_csv = OUT / args.out
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "method"])
        w.writerows(rows)
    report["total"] = {"rows": len(rows), "real": n0, "fake": n1,
                       "real_breakdown": {"real_ffpp": len(ffpp), "real_celeb": len(celeb), "real_dir": real_dir}}
    report["note"] = ("imbalance fake:real ≈ 1.37:1; finetune dùng WeightedRandomSampler cân bằng per-step. "
                      "MidJourney fake = 5 png + 247 gif. "
                      + ("real/ = folder real của test split DF40 (không nằm trong test_data_v3); sẽ leak nếu chạy protocol test DF40 đầy đủ. "
                         "CẢNH BÁO: thêm real/ làm boundary dịch về real (max_v2 FN 33->154)."
                         if not args.no_real_dir else "real/ folders: BỊ LOẠI (--no-real-dir)."))
    (OUT / "build_max_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_csv} + {OUT / 'build_max_report.json'}")


if __name__ == "__main__":
    main()
