#!/usr/bin/env python3
"""Rebuild finetune dataset with FF++ real replay (video-level, user-approved).

Keeps the current targeted-fake rows from data/finetune_exp02/train.csv (clean fakes),
but REPLACES the real pool: a stratified sample of exp02's own train real frames
(FF++ c23 frames from the 999 test videos + celeb_df_extracted), mirroring the
64:36 FF++:celeb ratio exp02 trained on. This preserves FF++-real recognition that a
celeb-only real pool collapsed.

Leak note (explicitly chosen by user): FF++ real frames here come from the same 999
videos that appear in test (video-level). We still assert ZERO exact-path overlap with
both test_data_v3/manifest.csv and test_balanced.csv.

Output: data/finetune_exp02/train_replay.csv (path,label,method) + replay_report.json
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path("/workspace/quangmanh/deepfake")
CUR_TRAIN = ROOT / "data/finetune_exp02/train.csv"
EXP02_TRAIN = Path("/workspace/hoangtuan/deepfake-ViT/data/splits/train_domain_balanced.csv")
CELEB_REAL = Path("/workspace/data/hoangtuan_data/processed/celeb_df_extracted")
MANIFEST = Path("/workspace/data/test_data_v3/manifest.csv")
TEST_BALANCED = ROOT / "experiments/results/error_analysis_lora/test_balanced.csv"
OUT = ROOT / "data/finetune_exp02"
SEED = 42
FFPP_REAL_TARGET = 5200   # ~64% of 8121 real
CELEB_REAL_TARGET = 2900  # ~36% of 8121 real


def walk_images(d):
    from PIL import Image
    for root, _, files in __import__("os").walk(d):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                yield Path(root) / fn


def load_test_paths():
    paths = set()
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            paths.add(str(Path("/workspace/data/test_data_v3") / r["path"]))
    with open(TEST_BALANCED, newline="") as f:
        for r in csv.DictReader(f):
            paths.add(r["path"])
    return paths


def main():
    rng = random.Random(SEED)
    test_paths = load_test_paths()
    print(f"test exclusion set: {len(test_paths)} paths")

    # ---- 1. keep current targeted fakes ----
    fakes = []
    with open(CUR_TRAIN, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["label"]) == 1:
                fakes.append((r["path"], 1, r["method"]))
    print(f"targeted fakes kept: {len(fakes)}")

    # ---- 2. FF++ real replay from exp02 train ----
    ffpp = set()
    with open(EXP02_TRAIN, newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] == "0" and ("/FaceForensics++/" in r["path"] or "/original_sequences/" in r["path"]):
                ffpp.add(r["path"])
    ffpp = sorted(ffpp)
    ffpp_clean = [p for p in ffpp if p not in test_paths]
    hit = len(ffpp) - len(ffpp_clean)
    sel_ffpp = rng.sample(ffpp_clean, min(FFPP_REAL_TARGET, len(ffpp_clean)))
    print(f"FF++ real replay: pool {len(ffpp)} -> clean {len(ffpp_clean)} (test-excluded {hit}) -> select {len(sel_ffpp)}")

    # ---- 3. celeb real ----
    celeb = [str(p) for p in walk_images(CELEB_REAL)]
    celeb_clean = [p for p in celeb if p not in test_paths]
    hit_c = len(celeb) - len(celeb_clean)
    sel_celeb = rng.sample(celeb_clean, min(CELEB_REAL_TARGET, len(celeb_clean)))
    print(f"celeb real: pool {len(celeb)} -> clean {len(celeb_clean)} (test-excluded {hit_c}) -> select {len(sel_celeb)}")

    # ---- 4. assemble + verify ----
    rows = [(p, 1, m) for p, _, m in fakes]
    rows += [(p, 0, "real_ffpp") for p in sel_ffpp]
    rows += [(p, 0, "real_celeb") for p in sel_celeb]
    sel_paths = {r[0] for r in rows}
    overlap = sel_paths & test_paths
    assert not overlap, f"LEAK: {len(overlap)} finetune paths in test!\n" + list(overlap)[0]
    print(f"VERIFY: {len(overlap)} exact-path overlaps with test ✓")

    rng.shuffle(rows)
    out_csv = OUT / "train_replay.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "method"])
        w.writerows(rows)
    n0 = sum(1 for r in rows if r[1] == 0)
    n1 = sum(1 for r in rows if r[1] == 1)
    rep = {
        "rows": len(rows), "real": n0, "fake": n1,
        "real_breakdown": {"real_ffpp": len(sel_ffpp), "real_celeb": len(sel_celeb)},
        "fake_targeted": len(fakes),
        "test_path_excluded": {"ffpp": hit, "celeb": hit_c},
        "note": "FF++ real = video-level replay từ train gốc exp02 (999 video test, không trùng frame). User-approved.",
    }
    (OUT / "replay_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {out_csv}: {len(rows)} rows ({n0} real [{len(sel_ffpp)} ffpp + {len(sel_celeb)} celeb] / {n1} fake)")
    print(f"Wrote {OUT / 'replay_report.json'}")


if __name__ == "__main__":
    main()
