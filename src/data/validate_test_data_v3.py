"""Validate test_data_v3 manifest and structure."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_ROOT = PROJECT_ROOT / "test_data_v3"
MANIFEST = TEST_ROOT / "manifest.csv"


def main():
    print("="*80)
    print("test_data_v3 VALIDATION")
    print("="*80)
    print(f"Test root: {TEST_ROOT}")
    print(f"Manifest:  {MANIFEST}")
    print()
    
    # Check manifest exists
    if not MANIFEST.exists():
        print(f"✗ Manifest not found: {MANIFEST}")
        sys.exit(1)
    
    # Load manifest
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append(r)
    
    print(f"Manifest rows: {len(rows):,}")
    
    # Check columns
    if rows:
        print(f"Columns: {list(rows[0].keys())}")
    
    # Validate required columns
    required = ["path", "label", "method", "identity", "domain", "video"]
    missing_cols = [c for c in required if c not in rows[0]]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
    else:
        print("✓ All required columns present")
    
    # Validate each row
    missing_files = []
    invalid_labels = []
    empty_identities = []
    duplicate_rows = []
    seen_paths = set()
    
    methods = Counter()
    identities = Counter()
    domains = Counter()
    videos = Counter()
    labels = Counter()
    
    for i, r in enumerate(rows):
        p = TEST_ROOT / r["path"]
        if not p.exists():
            missing_files.append((i, r["path"]))
        
        try:
            lbl = int(r["label"])
            if lbl not in (0, 1):
                invalid_labels.append((i, r["label"]))
        except ValueError:
            invalid_labels.append((i, r["label"]))
        
        if not r.get("identity") or r.get("identity") == "unknown":
            empty_identities.append((i, r))
        
        if r["path"] in seen_paths:
            duplicate_rows.append(r["path"])
        seen_paths.add(r["path"])
        
        methods[r.get("method", "unknown")] += 1
        identities[r.get("identity", "unknown")] += 1
        domains[r.get("domain", "unknown")] += 1
        videos[r.get("video", "unknown")] += 1
        labels[lbl] += 1
    
    print(f"\nValidation Results:")
    print(f"  Missing files: {len(missing_files)}")
    if missing_files[:5]:
        for i, p in missing_files[:5]:
            print(f"    row {i}: {p}")
    print(f"  Invalid labels: {len(invalid_labels)}")
    print(f"  Empty/unknown identities: {len(empty_identities)}")
    print(f"  Duplicate path rows: {len(duplicate_rows)}")
    
    print(f"\nClass distribution:")
    for lbl, cnt in sorted(labels.items()):
        name = "Real" if lbl == 0 else "Fake"
        print(f"  {name} (label={lbl}): {cnt:,}")
    
    print(f"\nMethod distribution: {len(methods)} methods")
    for m, cnt in methods.most_common(15):
        print(f"  {m:20s}: {cnt:>6,}")
    
    print(f"\nTop identities:")
    for id, cnt in identities.most_common(10):
        print(f"  {id:20s}: {cnt:>6,}")
    
    print(f"\nDomain distribution: {len(domains)} domains")
    for d, cnt in domains.most_common():
        print(f"  {d:20s}: {cnt:>6,}")
    
    print(f"\nVideo distribution: {len(videos)} unique video IDs")
    print(f"  Top videos:")
    for v, cnt in videos.most_common(5):
        print(f"    {v:20s}: {cnt:>6,}")
    
    print(f"\nTotal actual image files under test_data_v3/:")
    img_count = sum(1 for _ in TEST_ROOT.rglob("*.jpg"))
    print(f"  {img_count:,} .jpg files")
    
    # Save validation report
    report = {
        "manifest_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "missing_files": len(missing_files),
        "invalid_labels": len(invalid_labels),
        "empty_identities": len(empty_identities),
        "duplicate_paths": len(duplicate_rows),
        "class_distribution": {str(k): v for k, v in labels.items()},
        "method_count": len(methods),
        "methods": dict(methods.most_common()),
        "identity_count": len(identities),
        "domain_count": len(domains),
        "video_count": len(videos),
        "actual_images": img_count,
    }
    
    out_path = PROJECT_ROOT / "src" / "data" / "test_data_v3_validation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report: {out_path}")


if __name__ == "__main__":
    main()