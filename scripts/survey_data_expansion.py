#!/usr/bin/env python3
"""Khảo sát: mở rộng data finetune tới đâu cho các method yếu.

Nguồn:
  - df-40-test-full  (test split DF40, trừ frame/video có trong test_data_v3)
  - DF40_train_extracted (train split sạch, 31 method)

Output: experiments/results/data_expansion_survey.md + .json
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/quangmanh/deepfake")
TEST_FULL = Path("/workspace/data/df-40-test-full")
DF40_TRAIN = Path("/workspace/data/DF40_train_extracted")
MANIFEST = Path("/workspace/data/test_data_v3/manifest.csv")
OUT = ROOT / "experiments/results/data_expansion_survey.md"
OUT_J = ROOT / "experiments/results/data_expansion_survey.json"
IMG = (".png", ".jpg", ".jpeg")

# current finetune usage (from build report)
CURRENT = {
    "styleclip": 1200, "stargan": 984, "starganv2": 1000, "deepfacelab": 1200,
    "heygen": 838, "whichfaceisreal": 251, "CollabDiff": 250, "MidJourney": 5,
    "faceswap": 800, "sadtalker": 600, "wav2lip": 600, "MRAA": 400,
}
TEST_FULL_DIR = {"heygen": "heygen_new"}  # dir name -> manifest method
WEAK = ["MidJourney", "whichfaceisreal", "faceswap", "styleclip", "CollabDiff",
        "sadtalker", "wav2lip", "heygen", "stargan", "starganv2", "deepfacelab", "MRAA"]


def walk_images(d):
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(IMG):
                yield Path(root) / fn


def main():
    # ---- test exclusion tokens (method -> set of video stems) ----
    test_videos = defaultdict(set)
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] == "1":
                test_videos[r["method"]].add(Path(r["video"]).stem)

    # ---- df-40-test-full clean counts (per method, correct name mapping) ----
    testfull = {}
    for dirname in sorted(os.listdir(TEST_FULL)):
        d = TEST_FULL / dirname
        if not d.is_dir():
            continue
        method = {v: k for k, v in TEST_FULL_DIR.items()}.get(dirname, dirname)
        fake_dir = d / "fake"
        src = fake_dir if fake_dir.is_dir() else d
        total = sum(1 for _ in walk_images(src))
        tokens = test_videos.get(method, set())
        n_test = sum(1 for p in walk_images(src) if p.stem in tokens or p.parent.name in tokens)
        testfull[method] = {"total": total, "test": n_test, "clean": total - n_test}

    # ---- DF40_train_extracted counts ----
    dftrain = {}
    for d in sorted(DF40_TRAIN.iterdir()):
        if d.is_dir():
            dftrain[d.name] = sum(1 for _ in walk_images(d))

    # ---- assemble survey ----
    rows = []
    for m in WEAK:
        cur = CURRENT.get(m, 0)
        tf = testfull.get(m, {"clean": 0})
        dt = dftrain.get(m)
        max_single = max(tf["clean"], dt or 0)
        rows.append({
            "method": m,
            "current_used": cur,
            "df40testfull_clean": tf["clean"],
            "df40testfull_total": tf["total"],
            "df40train_extracted": dt,
            "max_available": max_single,
            "headroom": max_single - cur,
        })

    total_cur = sum(r["current_used"] for r in rows)
    total_max = sum(r["max_available"] for r in rows)
    md = [f"# Khảo sát mở rộng data finetune (method yếu)\n",
          f"| Method | Đang dùng | df-40-test-full sạch | DF40_train_extracted | **Tối đa** | Headroom |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        dt = f"{r['df40train_extracted']:,}" if r["df40train_extracted"] else "—"
        md.append(f"| {r['method']:<14} | {r['current_used']:>6} | {r['df40testfull_clean']:>6} "
                  f"| {dt:>10} | **{r['max_available']:>7,}** | +{r['headroom']:>6,} |")
    md.append(f"| **TOTAL** | {total_cur:,} | | | **{total_max:,}** | +{total_max-total_cur:,} |")
    md.append("\n## Ghi chú")
    md.append("- **MidJourney: hard ceiling 5 ảnh** — chỉ tồn tại trong test split; DF40_train_extracted không có. Không thể mở rộng.")
    md.append("- `df-40-test-full` là test split DF40, nhưng test_data_v3 chỉ sample một phần → frame/video còn lại sạch (loại trừ chính xác).")
    md.append("- `DF40_train_extracted` (train split) sạch hoàn toàn, 31 method × 21-31K frame.")
    md.append("- **faceswap/sadtalker/wav2lip/MRAA** có headroom khổng lồ (~22K/method) từ train split.")
    md.append("- styleclip +624, deepfacelab +1.142 — headroom vừa từ test split.")
    OUT.write_text("\n".join(md) + "\n")
    OUT_J.write_text(json.dumps({
        "weak_methods": rows, "total_current": total_cur, "total_max": total_max,
        "df40_train_extracted_methods": dftrain, "df40_testfull": testfull,
    }, indent=2))
    print("\n".join(md))
    print(f"\nWrote {OUT} + {OUT_J}")


if __name__ == "__main__":
    main()
