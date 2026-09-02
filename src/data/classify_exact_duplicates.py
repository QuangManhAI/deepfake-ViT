"""Classify the 247 cross-split exact duplicate groups."""

import csv
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
TABLE = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "exact_duplicates_table.csv"
OUT = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "exact_duplicates_classification.json"


def classify(row):
    """Classify a cross-split exact duplicate group."""
    methods = set(row["methods"].split(","))
    labels = set(int(x) for x in row["labels"].split(","))
    splits = set(row["splits"].split(","))
    
    num_images = int(row["num_images"])
    num_identities = int(row["num_identities"])
    num_videos = int(row["num_videos"])
    num_methods = int(row["num_methods"])
    
    # Determine if real image is reused
    has_real = "real" in methods
    
    # Determine if same physical image or regenerated/processed
    # If num_identities == num_images and num_videos == num_images:
    #   same physical image reused for different identity/video metadata
    # If num_identities == 1:
    #   same physical image within the same identity, likely regeneration/processing artifact
    # If num_methods > 1:
    #   real image reused across fake methods (intentional DF40 construction?)
    
    # Intentional dataset construction: real image reused across methods
    if has_real and num_methods > 1:
        # The real image is the source and is reused. If cross-split, still a problem
        # because the same physical image appears in train and val/test.
        return "LEAKAGE", "real_image_reused_across_methods"
    
    if num_identities == 1 and num_videos == 1:
        # Same identity, same video, same method. Possibly a regeneration or processing
        # artifact that produced an exact duplicate. Still leakage if cross split.
        return "LEAKAGE", "same_identity_same_video_duplicates"
    
    if num_identities == num_images and num_videos == num_images:
        # Same physical image, but different identity/video metadata.
        # This means the manifest reused the same image for different records.
        return "LEAKAGE", "same_image_reused_for_different_identities_videos"
    
    # Otherwise, ambiguous
    return "AMBIGUOUS", "mixed_identity_or_video"


def main():
    cross_groups = []
    with open(TABLE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cross_split"] == "True":
                cls, reason = classify(row)
                row["classification"] = cls
                row["reason"] = reason
                cross_groups.append(row)
    
    # Count classifications
    counts = Counter(r["classification"] for r in cross_groups)
    reasons = Counter(r["reason"] for r in cross_groups)
    
    # Count images to remove/reassign
    total_images = sum(int(r["num_images"]) for r in cross_groups)
    extra_images = total_images - len(cross_groups)  # one per group is kept
    
    # Save classification
    report = {
        "cross_split_groups": len(cross_groups),
        "classifications": dict(counts),
        "reasons": dict(reasons),
        "total_images_in_cross_groups": total_images,
        "extra_images_to_remove_or_reassign": extra_images,
        "groups": cross_groups,
    }
    
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    
    print("="*80)
    print("Cross-split exact duplicate classification")
    print("="*80)
    print(f"Cross-split groups: {len(cross_groups)}")
    print(f"Total images: {total_images}")
    print(f"Extra images (beyond one per group): {extra_images}")
    print("\nClassifications:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("\nReasons:")
    for k, v in reasons.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()