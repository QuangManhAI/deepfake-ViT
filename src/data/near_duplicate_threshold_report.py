"""Classify near-duplicate groups by strength using max_hamming as proxy."""

import csv
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
TABLE = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicates_table.csv"
OUT = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicate_threshold_report.json"


def main():
    rows = []
    with open(TABLE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["max_hamming"] = int(r["max_hamming"])
            rows.append(r)
    
    # NOTE: max_hamming is the largest pairwise Hamming distance within a connected component.
    # Because components are built with pairwise threshold 8, some max_hamming > 8 means
    # the group is a transitive chain. Pairs within the group may still be within threshold 8.
    #
    # We classify:
    #   strong:   max_hamming <= 5   (very high confidence near-duplicate)
    #   moderate: max_hamming 6-8    (connected under threshold 8)
    #   weak:     max_hamming 9-12   (transitive chain, possible hash collision / false positive)
    #   fp:       max_hamming > 12   (likely false positive or hash collision)
    
    strong = [r for r in rows if r["max_hamming"] <= 5]
    moderate = [r for r in rows if 6 <= r["max_hamming"] <= 8]
    weak = [r for r in rows if 9 <= r["max_hamming"] <= 12]
    fp = [r for r in rows if r["max_hamming"] > 12]
    
    def count_images(group):
        return sum(int(r["num_images"]) for r in group)
    
    def cross_split(group):
        return sum(1 for r in group if r["cross_split"] == "True")
    
    report = {
        "total_groups": len(rows),
        "total_images_in_groups": sum(int(r["num_images"]) for r in rows),
        "total_cross_split_groups": sum(1 for r in rows if r["cross_split"] == "True"),
        "threshold_definitions": {
            "strong": "max_hamming <= 5",
            "moderate": "6 <= max_hamming <= 8",
            "weak": "9 <= max_hamming <= 12",
            "false_positive": "max_hamming > 12",
        },
        "strong": {
            "groups": len(strong),
            "images": count_images(strong),
            "cross_split_groups": cross_split(strong),
        },
        "moderate": {
            "groups": len(moderate),
            "images": count_images(moderate),
            "cross_split_groups": cross_split(moderate),
        },
        "weak": {
            "groups": len(weak),
            "images": count_images(weak),
            "cross_split_groups": cross_split(weak),
        },
        "false_positive": {
            "groups": len(fp),
            "images": count_images(fp),
            "cross_split_groups": cross_split(fp),
        },
        "recommended_threshold": "5 for strong near-duplicate detection; 8 for broad recall; above 10 is likely false positive",
    }
    
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    
    print("="*80)
    print("Near-duplicate threshold classification")
    print("="*80)
    print(f"Total near-duplicate groups: {report['total_groups']}")
    print(f"Cross-split near-duplicate groups: {report['total_cross_split_groups']}")
    print()
    for cls in ["strong", "moderate", "weak", "false_positive"]:
        r = report[cls]
        print(f"{cls:15s}: {r['groups']:>5,} groups, {r['images']:>6,} images, {r['cross_split_groups']:>5,} cross-split")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()