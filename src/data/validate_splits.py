"""Validate generated split CSVs."""

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_csv_simple(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_csv_detailed(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def validate_split(name, path, detailed_path=None):
    print(f"\n--- Validating {name} ---")
    rows = load_csv_simple(path)
    print(f"  {len(rows):,} rows (excluding header)")
    
    # Check file existence
    missing = [r for r in rows if not Path(r["path"]).exists()]
    print(f"  Missing files: {len(missing)}")
    if missing[:3]:
        for r in missing[:3]:
            print(f"    {r['path']}")
    
    # Label distribution
    labels = Counter(int(r["label"]) for r in rows)
    print(f"  Labels: {dict(labels)}")
    
    if detailed_path and detailed_path.exists():
        drows = load_csv_detailed(detailed_path)
        methods = Counter(r["method"] for r in drows)
        identities = Counter(r["identity"] for r in drows)
        domains = Counter(r["domain"] for r in drows)
        videos = Counter(r["video"] for r in drows)
        
        print(f"  Methods: {len(methods)}")
        for m, c in methods.most_common(5):
            print(f"    {m}: {c}")
        print(f"  Identities: {len(identities)}")
        print(f"  Domains: {len(domains)} -> {dict(domains)}")
        print(f"  Videos: {len(videos)}")
        
        return {
            "name": name,
            "rows": len(rows),
            "missing_files": len(missing),
            "labels": dict(labels),
            "methods_count": len(methods),
            "identities_count": len(identities),
            "domains_count": len(domains),
            "videos_count": len(videos),
        }
    
    return {
        "name": name,
        "rows": len(rows),
        "missing_files": len(missing),
        "labels": dict(labels),
    }


def main():
    print("="*80)
    print("SPLIT VALIDATION")
    print("="*80)
    
    results = {}
    for split in ["train", "val", "test"]:
        results[split] = validate_split(
            split,
            SPLITS_DIR / f"{split}.csv",
            SPLITS_DIR / f"{split}_detailed.csv",
        )
    
    # Check no identity overlap between splits
    print("\n--- Identity overlap check ---")
    train_ids = set(r["identity"] for r in load_csv_detailed(SPLITS_DIR / "train_detailed.csv"))
    val_ids = set(r["identity"] for r in load_csv_detailed(SPLITS_DIR / "val_detailed.csv"))
    test_ids = set(r["identity"] for r in load_csv_detailed(SPLITS_DIR / "test_detailed.csv"))
    
    print(f"  Train identities: {len(train_ids)}")
    print(f"  Val identities:   {len(val_ids)}")
    print(f"  Test identities:  {len(test_ids)}")
    print(f"  Train ∩ Val: {len(train_ids & val_ids)}")
    print(f"  Train ∩ Test: {len(train_ids & test_ids)}")
    print(f"  Val ∩ Test: {len(val_ids & test_ids)}")
    
    # Check no path overlap
    print("\n--- Path overlap check ---")
    train_paths = set(r["path"] for r in load_csv_simple(SPLITS_DIR / "train.csv"))
    val_paths = set(r["path"] for r in load_csv_simple(SPLITS_DIR / "val.csv"))
    test_paths = set(r["path"] for r in load_csv_simple(SPLITS_DIR / "test.csv"))
    print(f"  Train ∩ Val paths: {len(train_paths & val_paths)}")
    print(f"  Train ∩ Test paths: {len(train_paths & test_paths)}")
    print(f"  Val ∩ Test paths: {len(val_paths & test_paths)}")
    
    results["identity_overlap"] = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    results["path_overlap"] = {
        "train_val": len(train_paths & val_paths),
        "train_test": len(train_paths & test_paths),
        "val_test": len(val_paths & test_paths),
    }
    
    out = PROJECT_ROOT / "src" / "data" / "splits_validation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation report: {out}")


if __name__ == "__main__":
    main()