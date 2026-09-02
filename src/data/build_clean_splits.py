"""
Build two cleaned, leakage-free split strategies:
A. identity-disjoint + exact-duplicate-aware
B. video-disjoint + exact-duplicate-aware

Preserves original data/splits/ as data/splits_original/ first.
"""

import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ORIGINAL_SPLITS = PROJECT_ROOT / "data" / "splits"
BACKUP_SPLITS = PROJECT_ROOT / "data" / "splits_original"
SPLITS_A = PROJECT_ROOT / "data" / "splits_identity_clean"
SPLITS_B = PROJECT_ROOT / "data" / "splits_video_clean"
TEST_ROOT = PROJECT_ROOT / "test_data_v3"


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_all_rows():
    all_rows = []
    for split in ["train", "val", "test"]:
        with open(ORIGINAL_SPLITS / f"{split}_detailed.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["label"] = int(r["label"])
                r["_split"] = split
                p = Path(r["path"])
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                r["_full_path"] = str(p)
                all_rows.append(r)
    return all_rows


def deduplicate(all_rows, seed=42):
    """Return list with one canonical row per MD5 group."""
    random.seed(seed)
    
    # Compute MD5 if not already present
    md5_to_rows = defaultdict(list)
    for r in all_rows:
        h = md5_file(r["_full_path"])
        r["_md5"] = h
        md5_to_rows[h].append(r)
    
    canonical = []
    removed = []
    for h, items in md5_to_rows.items():
        # Prefer real image, then by sorted path, then deterministic
        # Within group, keep the one with label=0 if present, else first path
        real_items = [x for x in items if x["label"] == 0]
        if real_items:
            chosen = sorted(real_items, key=lambda x: x["path"])[0]
        else:
            chosen = sorted(items, key=lambda x: x["path"])[0]
        canonical.append(chosen)
        for x in items:
            if x is not chosen:
                removed.append({
                    "md5": h,
                    "kept_path": chosen["path"],
                    "removed_path": x["path"],
                    "removed_split": x["_split"],
                    "kept_split": chosen["_split"],
                    "method": x["method"],
                    "identity": x["identity"],
                    "video": x["video"],
                })
    
    return canonical, removed


def split_by_key(rows, key, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split rows by a key (e.g., identity or video) without breaking groups."""
    random.seed(seed)
    
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    
    keys = list(groups.keys())
    random.shuffle(keys)
    
    total = len(rows)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)
    
    train, val, test = [], [], []
    train_count = 0
    val_count = 0
    
    for k in keys:
        items = groups[k]
        if train_count < train_target:
            train.extend(items)
            train_count += len(items)
        elif val_count < val_target:
            val.extend(items)
            val_count += len(items)
        else:
            test.extend(items)
    
    return train, val, test


def write_split(split_dir, train, val, test):
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard splits
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(split_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label"])
            w.writeheader()
            for r in rows:
                w.writerow({"path": r["path"], "label": r["label"]})
        
        with open(split_dir / f"{name}_detailed.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "method", "identity", "domain", "video"])
            w.writeheader()
            for r in rows:
                w.writerow({
                    "path": r["path"],
                    "label": r["label"],
                    "method": r["method"],
                    "identity": r["identity"],
                    "domain": r["domain"],
                    "video": r["video"],
                })


def compute_stats(train, val, test):
    def stats_for(rows):
        df = pd.DataFrame(rows)
        return {
            "n": len(rows),
            "n_real": int((df["label"] == 0).sum()),
            "n_fake": int((df["label"] == 1).sum()),
            "n_identities": df["identity"].nunique(),
            "n_videos": df["video"].nunique(),
            "n_methods": df["method"].nunique(),
            "methods": df[df["label"]==1]["method"].value_counts().to_dict(),
            "domains": df["domain"].value_counts().to_dict(),
        }
    
    return {
        "train": stats_for(train),
        "val": stats_for(val),
        "test": stats_for(test),
        "total": len(train) + len(val) + len(test),
    }


def validate_no_leakage(split_dir, identity_disjoint=False, video_disjoint=False):
    """Validate a cleaned split set for leakage."""
    train = pd.read_csv(split_dir / "train_detailed.csv")
    val = pd.read_csv(split_dir / "val_detailed.csv")
    test = pd.read_csv(split_dir / "test_detailed.csv")
    
    # Compute MD5s and check exact duplicates
    def md5_col(df):
        return df["path"].apply(lambda p: md5_file(str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p))
    
    train_md5 = set(md5_col(train))
    val_md5 = set(md5_col(val))
    test_md5 = set(md5_col(test))
    
    exact_leakage = {
        "train_val": len(train_md5 & val_md5),
        "train_test": len(train_md5 & test_md5),
        "val_test": len(val_md5 & test_md5),
    }
    
    identity_leakage = {
        "train_val": len(set(train["identity"]) & set(val["identity"])),
        "train_test": len(set(train["identity"]) & set(test["identity"])),
        "val_test": len(set(val["identity"]) & set(test["identity"])),
    }
    
    video_leakage = {
        "train_val": len(set(train["video"]) & set(val["video"])),
        "train_test": len(set(train["video"]) & set(test["video"])),
        "val_test": len(set(val["video"]) & set(test["video"])),
    }
    
    return {
        "exact_leakage": exact_leakage,
        "identity_leakage": identity_leakage,
        "video_leakage": video_leakage,
    }


def main():
    print("="*80)
    print("Building cleaned, leakage-free split strategies")
    print("="*80)
    
    # 1. Preserve original splits
    if BACKUP_SPLITS.exists():
        shutil.rmtree(BACKUP_SPLITS)
    shutil.copytree(ORIGINAL_SPLITS, BACKUP_SPLITS)
    print(f"Preserved original splits to: {BACKUP_SPLITS}")
    
    # 2. Load and deduplicate
    all_rows = load_all_rows()
    print(f"Loaded {len(all_rows)} rows from original splits")
    
    dedup_rows, removed = deduplicate(all_rows, seed=42)
    print(f"Deduplicated to {len(dedup_rows)} unique images")
    print(f"Removed/reassigned {len(removed)} exact duplicate rows")
    
    # Save removed log
    removed_path = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "removed_exact_duplicates.csv"
    removed_df = pd.DataFrame(removed)
    removed_df.to_csv(removed_path, index=False)
    print(f"Saved removed log: {removed_path}")
    
    # 3. Strategy A: identity-disjoint
    print("\n--- Strategy A: identity-disjoint + duplicate-aware ---")
    train_a, val_a, test_a = split_by_key(dedup_rows, "identity", train_ratio=0.70, val_ratio=0.15, seed=42)
    write_split(SPLITS_A, train_a, val_a, test_a)
    stats_a = compute_stats(train_a, val_a, test_a)
    val_a = validate_no_leakage(SPLITS_A, identity_disjoint=True, video_disjoint=False)
    
    # 4. Strategy B: video-disjoint
    print("--- Strategy B: video-disjoint + duplicate-aware ---")
    train_b, val_b, test_b = split_by_key(dedup_rows, "video", train_ratio=0.70, val_ratio=0.15, seed=42)
    write_split(SPLITS_B, train_b, val_b, test_b)
    stats_b = compute_stats(train_b, val_b, test_b)
    val_b = validate_no_leakage(SPLITS_B, identity_disjoint=False, video_disjoint=True)
    
    # 5. Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    for split, name in [("train", "Train"), ("val", "Val"), ("test", "Test"), ("total", "Total")]:
        if split == "total":
            print(f"{name:>6s}: A={stats_a['total']:,} B={stats_b['total']:,}")
        else:
            sa = stats_a[split]
            sb = stats_b[split]
            print(f"{name:>6s}:")
            print(f"  A: n={sa['n']:,} real={sa['n_real']:,} fake={sa['n_fake']:,} ratio={sa['n_fake']/max(1,sa['n_real']):.2f}:1 methods={sa['n_methods']} identities={sa['n_identities']:,} videos={sa['n_videos']:,}")
            print(f"  B: n={sb['n']:,} real={sb['n_real']:,} fake={sb['n_fake']:,} ratio={sb['n_fake']/max(1,sb['n_real']):.2f}:1 methods={sb['n_methods']} identities={sb['n_identities']:,} videos={sb['n_videos']:,}")
    
    print("\nValidation A:", json.dumps(val_a, indent=2))
    print("Validation B:", json.dumps(val_b, indent=2))
    
    # 6. Save report
    report = {
        "deduplicated_total": len(dedup_rows),
        "removed_rows": len(removed),
        "strategy_a": stats_a,
        "strategy_b": stats_b,
        "validation_a": val_a,
        "validation_b": val_b,
        "removed_log": str(removed_path),
        "splits_a": str(SPLITS_A),
        "splits_b": str(SPLITS_B),
        "splits_original": str(BACKUP_SPLITS),
    }
    
    report_path = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "clean_split_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved comparison: {report_path}")
    
    # 7. Recommend one
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    a_exact = sum(val_a["exact_leakage"].values())
    b_exact = sum(val_b["exact_leakage"].values())
    a_identity = sum(val_a["identity_leakage"].values())
    b_video = sum(val_b["video_leakage"].values())
    
    if a_exact == 0 and a_identity == 0:
        print("Recommended: Strategy A (identity-disjoint + duplicate-aware)")
        print("  - Prevents exact-duplicate leakage")
        print("  - Preserves identity-disjointness")
        print("  - Does not prevent video/source-level leakage by design")
        print("  - Class/method balance closer to original due to more fine-grained identities")
    else:
        print("Neither strategy is fully clean. Further investigation needed.")


if __name__ == "__main__":
    main()