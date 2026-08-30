"""
Validate identity and video protocol splits, compare them,
freeze the primary protocol, and produce a reproducible package.
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IDENTITY_SPLITS = PROJECT_ROOT / "data" / "splits_identity_clean"
VIDEO_SPLITS = PROJECT_ROOT / "data" / "splits_video_clean"
PROTOCOL_DIR = PROJECT_ROOT / "data" / "protocol"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "dataset_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_detailed(split_dir, split):
    with open(split_dir / f"{split}_detailed.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r["label"] = int(r["label"])
            r["split"] = split
            rows.append(r)
    return rows


def validate_split(split_dir, name, require_identity=False, require_video=False):
    print(f"\n--- Validating {name}: {split_dir} ---")
    
    # Existence
    for split in ["train", "val", "test"]:
        for ext in ["", "_detailed"]:
            p = split_dir / f"{split}{ext}.csv"
            if not p.exists():
                return False, {"error": f"missing {p}"}
    
    all_rows = []
    all_paths = set()
    split_rows = {}
    md5_sets = {}
    id_sets = {}
    video_sets = {}
    method_counts = Counter()
    class_counts = Counter()
    
    for split in ["train", "val", "test"]:
        rows = load_detailed(split_dir, split)
        split_rows[split] = rows
        all_rows.extend(rows)
        
        # File existence and path uniqueness
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                return False, {"error": f"missing file: {r['path']}"}
            if r["path"] in all_paths:
                return False, {"error": f"duplicate path: {r['path']}"}
            all_paths.add(r["path"])
            
            # label validity
            if r["label"] not in (0, 1):
                return False, {"error": f"invalid label {r['label']} for {r['path']}"}
            
            method_counts[r["method"]] += 1
            class_counts[r["label"]] += 1
        
        # md5 sets
        md5s = set()
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            md5s.add(md5_file(str(p)))
        md5_sets[split] = md5s
        id_sets[split] = set(r["identity"] for r in rows)
        video_sets[split] = set(r["video"] for r in rows)
    
    # Exact duplicate overlap
    exact_leakage = {
        "train_val": len(md5_sets["train"] & md5_sets["val"]),
        "train_test": len(md5_sets["train"] & md5_sets["test"]),
        "val_test": len(md5_sets["val"] & md5_sets["test"]),
    }
    
    # Identity overlap
    identity_leakage = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    
    # Video overlap
    video_leakage = {
        "train_val": len(video_sets["train"] & video_sets["val"]),
        "train_test": len(video_sets["train"] & video_sets["test"]),
        "val_test": len(video_sets["val"] & video_sets["test"]),
    }
    
    # Constraints
    failed = False
    if any(v > 0 for v in exact_leakage.values()):
        failed = True
    if require_identity and any(v > 0 for v in identity_leakage.values()):
        failed = True
    if require_video and any(v > 0 for v in video_leakage.values()):
        failed = True
    
    stats = {
        "name": name,
        "path": str(split_dir),
        "train_size": len(split_rows["train"]),
        "val_size": len(split_rows["val"]),
        "test_size": len(split_rows["test"]),
        "total_size": len(all_rows),
        "n_real": class_counts[0],
        "n_fake": class_counts[1],
        "n_methods": len(method_counts),
        "n_identities": len(id_sets["train"] | id_sets["val"] | id_sets["test"]),
        "n_videos": len(video_sets["train"] | video_sets["val"] | video_sets["test"]),
        "exact_leakage": exact_leakage,
        "identity_leakage": identity_leakage,
        "video_leakage": video_leakage,
        "failed": failed,
        "method_counts": dict(method_counts),
        "class_counts": dict(class_counts),
    }
    
    status = "PASS" if not failed else "FAIL"
    print(f"  {name}: {status}")
    print(f"  train={stats['train_size']}, val={stats['val_size']}, test={stats['test_size']}, total={stats['total_size']}")
    print(f"  real={stats['n_real']}, fake={stats['n_fake']}, methods={stats['n_methods']}")
    print(f"  exact: {exact_leakage}")
    print(f"  identity: {identity_leakage}")
    print(f"  video: {video_leakage}")
    
    return not failed, stats


def near_duplicate_overlap(split_dir):
    ndf = pd.read_csv(PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicates_table.csv")
    return int((ndf["cross_split"].astype(str) == "True").sum())


def build_comparison(identity_stats, video_stats):
    rows = []
    for key, label in [
        ("train_size", "Train samples"),
        ("val_size", "Val samples"),
        ("test_size", "Test samples"),
        ("n_real", "Real"),
        ("n_fake", "Fake"),
        ("n_identities", "# identities"),
        ("n_videos", "# videos"),
        ("n_methods", "# methods"),
    ]:
        rows.append({"Metric": label, "Identity Clean": identity_stats[key], "Video Clean": video_stats[key]})
    
    # Leakage counts
    rows.append({"Metric": "Exact duplicate leakage", "Identity Clean": sum(identity_stats["exact_leakage"].values()), "Video Clean": sum(video_stats["exact_leakage"].values())})
    rows.append({"Metric": "Identity overlap", "Identity Clean": sum(identity_stats["identity_leakage"].values()), "Video Clean": sum(video_stats["identity_leakage"].values())})
    rows.append({"Metric": "Video overlap", "Identity Clean": sum(identity_stats["video_leakage"].values()), "Video Clean": sum(video_stats["video_leakage"].values())})
    rows.append({"Metric": "Near-duplicate overlap groups", "Identity Clean": near_duplicate_overlap(IDENTITY_SPLITS), "Video Clean": near_duplicate_overlap(VIDEO_SPLITS)})
    
    # Weak method coverage (methods with < 500 samples)
    identity_low = sum(1 for v in identity_stats["method_counts"].values() if v < 500)
    video_low = sum(1 for v in video_stats["method_counts"].values() if v < 500)
    rows.append({"Metric": "Weak-method coverage (< 500 samples)", "Identity Clean": identity_low, "Video Clean": video_low})
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "protocol_comparison.csv", index=False)
    print(f"\nSaved protocol comparison: {OUT_DIR / 'protocol_comparison.csv'}")
    return df


def freeze_protocol(primary_stats, df):
    print("\n--- Freezing primary protocol ---")
    
    if PROTOCOL_DIR.exists():
        shutil.rmtree(PROTOCOL_DIR)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        shutil.copy(IDENTITY_SPLITS / f"{split}.csv", PROTOCOL_DIR / f"{split}.csv")
        shutil.copy(IDENTITY_SPLITS / f"{split}_detailed.csv", PROTOCOL_DIR / f"{split}_detailed.csv")
    
    metadata = {
        "dataset_version": "identity_clean_v1",
        "dataset_source": "test_data_v3",
        "dataset_size": 30691,
        "clean_size": primary_stats["total_size"],
        "split_strategy": "identity-disjoint",
        "random_seed": 42,
        "train_size": primary_stats["train_size"],
        "val_size": primary_stats["val_size"],
        "test_size": primary_stats["test_size"],
        "class_distribution": primary_stats["class_counts"],
        "method_distribution": primary_stats["method_counts"],
        "identity_constraint": "train/val/test identity-disjoint",
        "video_constraint": "KNOWN LIMITATION: video overlap exists across splits",
        "duplicate_constraint": "exact duplicates removed; one canonical per MD5",
        "primary_protocol_directory": str(PROTOCOL_DIR),
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    with open(PROTOCOL_DIR / "protocol_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    readme = f"""# Primary Dataset Protocol

**Version:** identity_clean_v1
**Source:** test_data_v3
**Strategy:** identity-disjoint + exact-duplicate-aware
**Random seed:** 42

## Files

- `train.csv` / `train_detailed.csv`
- `val.csv` / `val_detailed.csv`
- `test.csv` / `test_detailed.csv`
- `protocol_metadata.json`

## Sizes

- Train: {metadata['train_size']}
- Val:   {metadata['val_size']}
- Test:  {metadata['test_size']}
- Total: {metadata['clean_size']}

## Constraints

- identity(train) ∩ identity(val) = ∅
- identity(train) ∩ identity(test) = ∅
- identity(val) ∩ identity(test) = ∅
- exact-duplicate overlap across splits = 0

## Known Limitations

- **Video/source overlap** remains across splits (1,509 train↔val, 1,542 train↔test, 930 val↔test).
- **Near-duplicate overlap** is documented but not removed (4,215 cross-split groups).
- **Class imbalance** is ~24.5:1 fake:real.

## Reproducibility

This protocol was generated from:
- `src/data/build_clean_splits.py`
- `src/data/validate_protocol.py`
- `data/splits_identity_clean/`
"""
    (PROTOCOL_DIR / "README.md").write_text(readme)
    
    print(f"Frozen primary protocol to: {PROTOCOL_DIR}")
    return metadata


def create_config():
    config = {
        "DATA_PROTOCOL": "identity_clean_v1",
        "protocol_dir": str(PROTOCOL_DIR),
        "train_csv": str(PROTOCOL_DIR / "train.csv"),
        "val_csv": str(PROTOCOL_DIR / "val.csv"),
        "test_csv": str(PROTOCOL_DIR / "test.csv"),
    }
    with open(PROTOCOL_DIR / "protocol_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved protocol config: {PROTOCOL_DIR / 'protocol_config.json'}")
    return config


def main():
    print("="*80)
    print("DATASET PROTOCOL VALIDATION & FREEZE")
    print("="*80)
    
    # 1. Validate both protocols
    id_ok, id_stats = validate_split(IDENTITY_SPLITS, "identity_clean", require_identity=True, require_video=False)
    video_ok, video_stats = validate_split(VIDEO_SPLITS, "video_clean", require_identity=False, require_video=True)
    
    if not id_ok:
        print("\nFATAL: identity_clean split validation failed")
        sys.exit(1)
    if not video_ok:
        print("\nFATAL: video_clean split validation failed")
        sys.exit(1)
    
    # 2. Compare
    df = build_comparison(id_stats, video_stats)
    
    # 3. Select primary and freeze
    primary_stats = id_stats
    metadata = freeze_protocol(primary_stats, df)
    config = create_config()
    
    # 4. Save validation report
    validation_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocols": {
            "identity_clean": id_stats,
            "video_clean": video_stats,
        },
        "primary_protocol": "identity_clean",
        "primary_metadata": metadata,
        "protocol_config": config,
        "status": "VALIDATED",
    }
    
    with open(OUT_DIR / "protocol_validation.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print("\n" + "="*80)
    print("DATASET PROTOCOL PHASE COMPLETE")
    print("="*80)
    print(f"Primary protocol: identity-disjoint at {PROTOCOL_DIR}")
    print(f"Comparison: {OUT_DIR / 'protocol_comparison.csv'}")
    print(f"Validation: {OUT_DIR / 'protocol_validation.json'}")


if __name__ == "__main__":
    main()