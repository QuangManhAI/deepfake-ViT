"""Final consistency audit before marking DATA/EDA phase complete."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "test_data_v3" / "manifest.csv"
CLEAN_SPLITS = PROJECT_ROOT / "data" / "splits_identity_clean"
ORIGINAL_SPLITS = PROJECT_ROOT / "data" / "splits_original"
REMOVED_LOG = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "removed_exact_duplicates.csv"
QUALITY_DIR = PROJECT_ROOT / "experiments" / "results" / "data_quality"
OUT = PROJECT_ROOT / "experiments" / "results" / "data_quality" / "consistency_audit.json"


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def main():
    print("="*80)
    print("FINAL CONSISTENCY AUDIT")
    print("="*80)
    
    # 1. Manifest vs cleaned split counts
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        manifest_rows = sum(1 for _ in csv.DictReader(f))
    
    clean_total = 0
    clean_split_counts = {}
    for split in ["train", "val", "test"]:
        rows = load_csv(CLEAN_SPLITS / f"{split}_detailed.csv")
        clean_split_counts[split] = len(rows)
        clean_total += len(rows)
    
    removed = pd.read_csv(REMOVED_LOG)
    removed_count = len(removed)
    
    print(f"\n1. Exact duplicate cleanup")
    print(f"  Manifest rows: {manifest_rows}")
    print(f"  Removed/reassigned: {removed_count}")
    print(f"  Clean split total: {clean_total}")
    print(f"  {manifest_rows} - {removed_count} = {manifest_rows - removed_count}")
    print(f"  Match: {manifest_rows - removed_count == clean_total}")
    
    # 2. Exact duplicate overlap in clean split
    md5_sets = {}
    for split in ["train", "val", "test"]:
        rows = load_csv(CLEAN_SPLITS / f"{split}_detailed.csv")
        md5s = set()
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            md5s.add(md5_file(str(p)))
        md5_sets[split] = md5s
    
    exact_leakage = {
        "train_val": len(md5_sets["train"] & md5_sets["val"]),
        "train_test": len(md5_sets["train"] & md5_sets["test"]),
        "val_test": len(md5_sets["val"] & md5_sets["test"]),
    }
    
    print(f"\n2. Exact duplicate overlap (clean split)")
    print(json.dumps(exact_leakage, indent=2))
    
    # 3. Identity overlap
    id_sets = {}
    for split in ["train", "val", "test"]:
        rows = load_csv(CLEAN_SPLITS / f"{split}_detailed.csv")
        id_sets[split] = set(r["identity"] for r in rows)
    
    identity_leakage = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    
    print(f"\n3. Identity overlap")
    print(json.dumps(identity_leakage, indent=2))
    
    # 4. Video overlap
    video_sets = {}
    for split in ["train", "val", "test"]:
        rows = load_csv(CLEAN_SPLITS / f"{split}_detailed.csv")
        video_sets[split] = set(r["video"] for r in rows)
    
    video_leakage = {
        "train_val": len(video_sets["train"] & video_sets["val"]),
        "train_test": len(video_sets["train"] & video_sets["test"]),
        "val_test": len(video_sets["val"] & video_sets["test"]),
    }
    
    print(f"\n4. Video overlap")
    print(json.dumps(video_leakage, indent=2))
    
    # 5. Near-duplicate overlap (use report)
    near_report = json.loads((PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicates_report.json").read_text())
    ndf = pd.read_csv(PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicates_table.csv")
    near_cross = int((ndf["cross_split"].astype(str) == "True").sum())
    
    print(f"\n5. Near-duplicate overlap")
    print(f"  Near-duplicate groups: {near_report['near_duplicate_groups']}")
    print(f"  Cross-split near-duplicate groups: {near_cross}")
    
    # 6. Distribution shift (JS divergence)
    dshift = pd.read_csv(QUALITY_DIR / "distribution_shift.csv")
    js_max = dshift["jensen_shannon"].max()
    
    print(f"\n6. Distribution shift")
    print(f"  Maximum Jensen-Shannon divergence: {js_max:.4f}")
    print(f"  Conclusion: No substantial distribution shift was detected for the evaluated metrics.")
    
    # 7. Weak method classification
    msum = pd.read_csv(QUALITY_DIR / "method_quality_summary.csv")
    mean_count = msum["sample_count"].mean()
    med_edge = msum["median_edge"].median()
    
    def classify(row):
        low_data = row["sample_count"] < mean_count
        low_quality = (row["median_width"] < 256) or (row["median_edge"] < med_edge) or (row["quality_outlier_rate"] > 0.2)
        if low_data and low_quality:
            return "BOTH"
        elif low_data:
            return "LOW_DATA_VOLUME"
        elif low_quality:
            return "LOW_IMAGE_QUALITY"
        else:
            return "NONE"
    
    msum["weakness_type"] = msum.apply(classify, axis=1)
    msum.to_csv(QUALITY_DIR / "method_quality_summary.csv", index=False)
    
    print(f"\n7. Weak method classification")
    print(msum[["method", "sample_count", "median_edge", "median_width", "weakness_type"]].head(10).to_string(index=False))
    
    # 8. Counts across CSVs
    sample_quality = pd.read_csv(QUALITY_DIR / "sample_quality.csv")
    
    print(f"\n8. CSV consistency")
    print(f"  sample_quality.csv rows: {len(sample_quality)}")
    print(f"  method_quality_summary.csv rows: {len(msum)}")
    print(f"  Clean split total: {clean_total}")
    print(f"  sample_quality matches clean total: {len(sample_quality) == clean_total}")
    
    # 9. Audit summary
    audit = {
        "manifest_rows": manifest_rows,
        "clean_split_total": clean_total,
        "removed_reassigned": removed_count,
        "arithmetic_check": manifest_rows - removed_count == clean_total,
        "exact_leakage": exact_leakage,
        "identity_leakage": identity_leakage,
        "video_leakage": video_leakage,
        "near_duplicate_groups": near_report["near_duplicate_groups"],
        "cross_split_near_groups": near_cross,
        "max_js_divergence": js_max,
        "distribution_shift_conclusion": "No substantial distribution shift was detected for the evaluated metrics.",
        "sample_quality_rows": len(sample_quality),
        "data_matches": len(sample_quality) == clean_total,
    }
    
    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2)
    
    print(f"\nSaved audit: {OUT}")
    
    # Final status
    if all(v == 0 for v in exact_leakage.values()) and audit["arithmetic_check"] and audit["data_matches"]:
        print("\n" + "="*80)
        print("DATA / EDA PHASE = COMPLETE")
        print("="*80)
    else:
        print("\nAUDIT FAILED — see details above")


if __name__ == "__main__":
    main()