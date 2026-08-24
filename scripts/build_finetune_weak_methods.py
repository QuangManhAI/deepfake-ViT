#!/usr/bin/env python3
"""Build a leak-free finetune dataset for exp02 (Tuấn) targeting the weak methods.

Sources (all under /workspace/data):
  - df-40-test-full  : previously-unseen methods (styleclip, stargan, starganv2,
                       deepfacelab, heygen_new, whichfaceisreal, CollabDiff, MidJourney).
                       NOTE this is the DF40 TEST split — we keep only frames whose source
                       token/video is NOT present in test_data_v3 (exact-file exclusion).
  - DF40_train_extracted : weak-but-seen methods (faceswap, sadtalker, wav2lip, MRAA)
                       from the clean DF40 TRAIN split (disjoint from test by construction).
  - celeb_df_extracted   : real faces, video-disjoint from the Celeb-DF-v2 test list
                       (verified: 463 videos, 0 overlap with the 178 test videos).

Output: data/finetune_exp02/train.csv (path,label,method) + build_report.json.
Verification: asserts zero byte-level path overlap with test_data_v3 manifest.
"""
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/quangmanh/deepfake")
TEST_ROOT = Path("/workspace/data/test_data_v3")
MANIFEST = TEST_ROOT / "manifest.csv"
TEST_FULL = Path("/workspace/data/df-40-test-full")
DF40_TRAIN = Path("/workspace/data/DF40_train_extracted")
CELEB_REAL = Path("/workspace/data/hoangtuan_data/processed/celeb_df_extracted")
OUT = ROOT / "data/finetune_exp02"
SEED = 42

IMG_EXTS = (".png", ".jpg", ".jpeg")

# method -> max frames to keep in the finetune set
FAKE_CAPS = {
    # previously-unseen methods, from df-40-test-full minus test frames/videos
    "styleclip": 1200, "stargan": 984, "starganv2": 1000, "deepfacelab": 1200,
    "heygen": 838, "whichfaceisreal": 251, "CollabDiff": 250, "MidJourney": 5,
    # weak-but-seen methods, from the clean DF40 train split
    "faceswap": 800, "sadtalker": 600, "wav2lip": 600, "MRAA": 400,
}
# df-40-test-full dir name may differ from manifest method name
TEST_FULL_DIR = {"heygen": "heygen_new"}

# df-40-test-full is the DF40 TEST split; methods that store whole test videos here
# (video methods) must be excluded at the VIDEO level -> keep only frames from videos
# NOT used in test_data_v3. Image-static methods are excluded at the file level
# (each image is an independent unit; excluding the exact test files is sufficient).
VIDEO_LEVEL_METHODS = {"deepfacelab", "heygen", "CollabDiff"}


def norm_video(v):
    return Path(str(v)).stem


def walk_images(d):
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield Path(root) / fn


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. test exclusion sets from the manifest ----
    test_paths = set()
    test_videos = defaultdict(set)  # method -> normalized video/source tokens in test
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            test_paths.add(str(TEST_ROOT / r["path"]))
            if r["label"] == "1":
                test_videos[r["method"]].add(norm_video(r["video"]))
    print(f"test manifest: {len(test_paths)} rows | {sum(len(v) for v in test_videos.values())} fake source tokens")

    # ---- 2. fake: previously-unseen methods from df-40-test-full (minus test) ----
    rows, report = [], {}
    excluded_by_test = 0
    for method, cap in FAKE_CAPS.items():
        if method in ("faceswap", "sadtalker", "wav2lip", "MRAA"):
            continue  # handled from DF40_train_extracted below
        src_dir = TEST_FULL / TEST_FULL_DIR.get(method, method) / "fake"
        if not src_dir.is_dir():
            print(f"!! {method}: missing {src_dir}"); continue
        tokens = test_videos.get(method, set())
        cand = []
        for p in walk_images(src_dir):
            base, parent = p.stem, p.parent.name
            is_test = (str(p) in test_paths) or (base in tokens) or (parent in tokens)
            if is_test:
                excluded_by_test += 1
                continue
            cand.append(p)
        sel = rng.sample(cand, min(cap, len(cand))) if cand else []
        for p in sel:
            rows.append((str(p), 1, method))
        report[method] = {"pool_clean": len(cand), "selected": len(sel), "test_excluded": excluded_by_test}
        excluded_by_test = 0
        print(f"  {method:<14} clean={len(cand):>5}  selected={len(sel):>4}")

    # ---- 3. fake: weak-but-seen methods from the clean DF40 train split ----
    for method, cap in FAKE_CAPS.items():
        if method not in ("faceswap", "sadtalker", "wav2lip", "MRAA"):
            continue
        src_dir = DF40_TRAIN / method
        cand = list(walk_images(src_dir))
        # sanity: train split must not contain any test file
        hit_test = sum(1 for p in cand if str(p) in test_paths)
        if hit_test:
            print(f"!! {method}: {hit_test} train frames collide with test — skipping collision")
            cand = [p for p in cand if str(p) not in test_paths]
        sel = rng.sample(cand, min(cap, len(cand))) if cand else []
        for p in sel:
            rows.append((str(p), 1, method))
        report[method] = {"pool": len(cand), "selected": len(sel)}
        print(f"  {method:<14} pool={len(cand):>6}  selected={len(sel):>4}")

    # ---- 4. real from celeb_df_extracted (video-disjoint from Celeb-DF-v2 test list) ----
    real_cand = list(walk_images(CELEB_REAL))
    hit_test = sum(1 for p in real_cand if str(p) in test_paths)
    assert hit_test == 0, f"{hit_test} real frames collide with test manifest!"
    n_real = len(rows)  # balance 1:1
    sel_real = rng.sample(real_cand, min(n_real, len(real_cand)))
    for p in sel_real:
        rows.append((str(p), 0, "real"))
    report["real"] = {"pool": len(real_cand), "selected": len(sel_real), "note": "celeb_df_extracted, video-disjoint from Celeb-DF-v2 test list"}

    # ---- 5. verify + write ----
    sel_paths = {r[0] for r in rows}
    overlap = sel_paths & test_paths
    assert not overlap, f"LEAK: {len(overlap)} finetune paths appear in test manifest!\n" + list(overlap)[:3][0]
    print(f"\nVERIFY: {len(overlap)} path overlaps with test manifest ✓")

    rng.shuffle(rows)
    with open(OUT / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "method"])
        w.writerows(rows)

    n0 = sum(1 for r in rows if r[1] == 0)
    n1 = sum(1 for r in rows if r[1] == 1)
    report["total"] = {"rows": len(rows), "real": n0, "fake": n1}
    with open(OUT / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT / 'train.csv'}: {len(rows)} rows ({n0} real / {n1} fake)")
    print(f"Wrote {OUT / 'build_report.json'}")


if __name__ == "__main__":
    main()
