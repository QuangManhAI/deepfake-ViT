"""Run near-duplicate detection on a sample of the dataset."""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.duplicate_detection import find_near_duplicates, analyze_duplicate_statistics


def main():
    random.seed(42)
    
    # Load test split (smallest) for speed
    rows = []
    with open(PROJECT_ROOT / "data" / "splits" / "test_detailed.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    # Sample 1000 images
    sample = random.sample(rows, min(1000, len(rows)))
    paths = [r["path"] for r in sample]
    
    print(f"Analyzing near-duplicates on {len(paths)} test images...")
    
    near = find_near_duplicates(paths, hash_function='average_hash', threshold=8)
    stats = analyze_duplicate_statistics(near)
    
    print(f"Near-duplicate groups: {stats.get('num_groups', 0)}")
    print(f"Affected images: {stats.get('total_affected_images', 0)}")
    if 'largest_group' in stats:
        print(f"Largest group: {stats['largest_group']}")
    
    # Save
    out = PROJECT_ROOT / "experiments" / "results" / "eda_real_data" / "near_duplicates_sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, "w") as f:
        json.dump({
            "sample_size": len(paths),
            "groups": {k: v for k, v in near.items()},
            "stats": stats,
        }, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()