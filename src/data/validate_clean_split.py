"""
Executable validator for a cleaned split.

Fails loudly if:
- exact duplicates exist across splits
- identity overlap exists (for identity-disjoint strategy)
- video overlap exists (for video-disjoint strategy)
- paths are missing
- duplicate paths exist

Reports class/method/identity/video balance.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_detailed(split_dir, split):
    path = split_dir / f"{split}_detailed.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def validate(split_dir, check_identity_disjoint=False, check_video_disjoint=False):
    print("="*80)
    print(f"Validating split: {split_dir}")
    print(f"  identity-disjoint required: {check_identity_disjoint}")
    print(f"  video-disjoint required: {check_video_disjoint}")
    print("="*80)
    
    train = load_detailed(split_dir, "train")
    val = load_detailed(split_dir, "val")
    test = load_detailed(split_dir, "test")
    
    total = len(train) + len(val) + len(test)
    print(f"\nTotal rows: {total:,}")
    print(f"  train: {len(train):,}")
    print(f"  val:   {len(val):,}")
    print(f"  test:  {len(test):,}")
    
    # Missing files
    missing = []
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                missing.append((split_name, r["path"]))
    print(f"\nMissing files: {len(missing)}")
    
    # Duplicate paths within splits
    dup_paths = 0
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        seen = set()
        for r in rows:
            if r["path"] in seen:
                dup_paths += 1
            seen.add(r["path"])
    print(f"Duplicate paths within splits: {dup_paths}")
    
    # Class balance
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        real = sum(1 for r in rows if r["label"] == 0)
        fake = sum(1 for r in rows if r["label"] == 1)
        print(f"{split_name}: real={real:,} fake={fake:,} ratio={fake/max(1,real):.2f}:1")
    
    # Method balance (top 5)
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        methods = Counter(r["method"] for r in rows)
        print(f"{split_name} top methods: {methods.most_common(5)}")
    
    # Exact duplicate leakage
    def md5s(rows):
        return set(md5_file(str(PROJECT_ROOT / r["path"]) if not Path(r["path"]).is_absolute() else r["path"]) for r in rows)
    
    train_md5 = md5s(train)
    val_md5 = md5s(val)
    test_md5 = md5s(test)
    
    exact_leakage = {
        "train_val": len(train_md5 & val_md5),
        "train_test": len(train_md5 & test_md5),
        "val_test": len(val_md5 & test_md5),
    }
    print(f"\nExact duplicate leakage: {exact_leakage}")
    
    # Identity overlap
    identity_overlap = {
        "train_val": len(set(r["identity"] for r in train) & set(r["identity"] for r in val)),
        "train_test": len(set(r["identity"] for r in train) & set(r["identity"] for r in test)),
        "val_test": len(set(r["identity"] for r in val) & set(r["identity"] for r in test)),
    }
    print(f"Identity overlap: {identity_overlap}")
    
    # Video overlap
    video_overlap = {
        "train_val": len(set(r["video"] for r in train) & set(r["video"] for r in val)),
        "train_test": len(set(r["video"] for r in train) & set(r["video"] for r in test)),
        "val_test": len(set(r["video"] for r in val) & set(r["video"] for r in test)),
    }
    print(f"Video overlap: {video_overlap}")
    
    # Identity / video per split
    for split_name, rows, key in [("train", train, "identity"), ("train", train, "video"), ("val", val, "identity"), ("val", val, "video"), ("test", test, "identity"), ("test", test, "video")]:
        uniq = len(set(r[key] for r in rows))
        print(f"{split_name} {key}s: {uniq:,}")
    
    # Fail loudly if required conditions are violated
    failed = False
    
    if any(v > 0 for v in exact_leakage.values()):
        print("\nFAIL: exact duplicate leakage detected")
        failed = True
    
    if check_identity_disjoint and any(v > 0 for v in identity_overlap.values()):
        print("\nFAIL: identity overlap detected but identity-disjoint required")
        failed = True
    
    if check_video_disjoint and any(v > 0 for v in video_overlap.values()):
        print("\nFAIL: video overlap detected but video-disjoint required")
        failed = True
    
    if missing:
        print(f"\nFAIL: {len(missing)} missing files")
        failed = True
    
    if dup_paths:
        print(f"\nFAIL: {dup_paths} duplicate paths within splits")
        failed = True
    
    if not failed:
        print("\nPASS: all leakage checks passed")
    else:
        print("\nVALIDATION FAILED")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits_identity_clean", help="Split directory to validate")
    parser.add_argument("--identity-disjoint", action="store_true")
    parser.add_argument("--video-disjoint", action="store_true")
    args = parser.parse_args()
    
    split_dir = PROJECT_ROOT / args.split_dir
    validate(split_dir, check_identity_disjoint=args.identity_disjoint, check_video_disjoint=args.video_disjoint)


if __name__ == "__main__":
    main()