"""
Full-dataset exact and near-duplicate analysis for test_data_v3 splits.

Produces:
- exact_duplicates_table.csv
- exact_duplicates_report.json
- near_duplicates_table.csv
- near_duplicates_report.json
"""

import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "eda_real_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_detailed(split: str) -> List[dict]:
    path = SPLITS_DIR / f"{split}_detailed.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            r["split"] = split
            rows.append(r)
    return rows


def md5_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def average_hash_8x8(path: str) -> str:
    """Compute 8x8 average hash as 16-hex-character string."""
    with Image.open(path) as img:
        img = img.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = np.array(img)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    bits_str = "".join("1" if b else "0" for b in bits)
    return hex(int(bits_str, 2))[2:].zfill(16)


def hamming(hex1: str, hex2: str) -> int:
    x = int(hex1, 16) ^ int(hex2, 16)
    return bin(x).count("1")


def split_hash_quadrants(ahash: str) -> List[str]:
    """Return 4 16-bit quadrants of the 8x8 ahash for LSH bucketing."""
    # 8x8 bits = 64 bits = 16 hex chars
    # quadrant 0: bits (0:4,0:4) = first 2 hex chars and 5th-6th? easier: use a 4x4 downsampling
    # simpler: take the first 4, next 4, next 4, last 4 hex chars? That's 16-bit buckets but not spatial.
    # Better: 4x4 block: top-left chars 0-3, top-right 4-7, bottom-left 8-11, bottom-right 12-15
    return [ahash[0:4], ahash[4:8], ahash[8:12], ahash[12:16]]


def build_full_dataset():
    all_rows = []
    for split in ["train", "val", "test"]:
        all_rows.extend(load_detailed(split))
    return all_rows


def analyze_exact_duplicates(rows: List[dict]) -> Tuple[Dict, List[dict], List[dict]]:
    print("Computing MD5 for all images...")
    t0 = time.time()
    hash_to_rows = defaultdict(list)
    for i, r in enumerate(rows):
        p = Path(r["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        h = md5_file(str(p))
        r["md5"] = h
        hash_to_rows[h].append(r)
        if i % 5000 == 0:
            print(f"  {i:,} processed")
    print(f"  MD5 done in {time.time()-t0:.1f}s")
    
    # Build table
    table = []
    cross_split_groups = []
    for h, items in hash_to_rows.items():
        if len(items) > 1:
            methods = sorted(set(r["method"] for r in items))
            labels = sorted(set(r["label"] for r in items))
            identities = sorted(set(r["identity"] for r in items))
            videos = sorted(set(r["video"] for r in items))
            splits = sorted(set(r["split"] for r in items))
            
            row = {
                "md5": h,
                "num_images": len(items),
                "num_methods": len(methods),
                "methods": ",".join(methods),
                "labels": ",".join(str(x) for x in labels),
                "num_identities": len(identities),
                "num_videos": len(videos),
                "splits": ",".join(splits),
                "cross_split": len(splits) > 1,
                "cross_method": len(methods) > 1,
            }
            table.append(row)
            if len(splits) > 1:
                cross_split_groups.append(row)
    
    return hash_to_rows, table, cross_split_groups


def analyze_near_duplicates(rows: List[dict], threshold: int = 8) -> Tuple[List[dict], List[dict]]:
    print("Computing 8x8 average hashes...")
    t0 = time.time()
    hashes = []
    for i, r in enumerate(rows):
        p = Path(r["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        ahash = average_hash_8x8(str(p))
        r["ahash"] = ahash
        hashes.append((r, ahash))
        if i % 5000 == 0:
            print(f"  {i:,} processed")
    print(f"  Hashing done in {time.time()-t0:.1f}s")
    
    # LSH: bucket by 4 quadrants
    buckets = defaultdict(list)
    for r, ahash in hashes:
        for q in split_hash_quadrants(ahash):
            buckets[q].append((r, ahash))
    
    # Compare candidates within each bucket
    print("Comparing hash candidates...")
    t0 = time.time()
    seen_pairs = set()
    near_groups = []  # list of (canonical_path, [dup_paths...])
    canonical_map = {}  # path -> list of near duplicates
    
    for bucket, candidates in buckets.items():
        if len(candidates) < 2:
            continue
        # Compare pairs in this bucket
        for (r1, h1), (r2, h2) in combinations(candidates, 2):
            if h1 == h2:
                continue  # exact duplicate, skip
            key = tuple(sorted([r1["path"], r2["path"]]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            d = hamming(h1, h2)
            if d <= threshold:
                canonical_map.setdefault(r1["path"], []).append(r2)
                canonical_map.setdefault(r2["path"], []).append(r1)
    
    # Build groups
    processed = set()
    table = []
    cross_split_groups = []
    for r, ahash in hashes:
        if r["path"] in processed:
            continue
        group = [r] + canonical_map.get(r["path"], [])
        if len(group) < 2:
            continue
        processed.update(x["path"] for x in group)
        
        methods = sorted(set(x["method"] for x in group))
        labels = sorted(set(x["label"] for x in group))
        identities = sorted(set(x["identity"] for x in group))
        videos = sorted(set(x["video"] for x in group))
        splits = sorted(set(x["split"] for x in group))
        
        # Min/max hamming within group
        hashes_g = [x["ahash"] for x in group]
        max_hamming = max(hamming(hashes_g[i], hashes_g[j]) for i, j in combinations(range(len(hashes_g)), 2)) if len(hashes_g) > 1 else 0
        
        row = {
            "canonical_path": group[0]["path"],
            "num_images": len(group),
            "max_hamming": max_hamming,
            "num_methods": len(methods),
            "methods": ",".join(methods),
            "labels": ",".join(str(x) for x in labels),
            "num_identities": len(identities),
            "num_videos": len(videos),
            "splits": ",".join(splits),
            "cross_split": len(splits) > 1,
            "cross_method": len(methods) > 1,
        }
        table.append(row)
        if len(splits) > 1:
            cross_split_groups.append(row)
    
    print(f"  Near-duplicate comparison done in {time.time()-t0:.1f}s")
    return table, cross_split_groups


def main():
    print("="*80)
    print("FULL-DATASET DUPLICATE & NEAR-DUPLICATE ANALYSIS")
    print("="*80)
    
    rows = build_full_dataset()
    print(f"Loaded {len(rows)} rows")
    
    # Exact duplicates
    hash_to_rows, exact_table, exact_cross = analyze_exact_duplicates(rows)
    
    # Save exact table
    if exact_table:
        with open(OUT_DIR / "exact_duplicates_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=exact_table[0].keys())
            w.writeheader()
            w.writerows(exact_table)
    
    total_extra = sum(r["num_images"] - 1 for r in exact_table)
    cross_split_count = len(exact_cross)
    
    exact_report = {
        "total_images": len(rows),
        "duplicate_groups": len(exact_table),
        "extra_images": total_extra,
        "cross_split_groups": cross_split_count,
        "table_path": str(OUT_DIR / "exact_duplicates_table.csv"),
    }
    with open(OUT_DIR / "exact_duplicates_report.json", "w") as f:
        json.dump(exact_report, f, indent=2)
    
    print(f"\nExact duplicates: {len(exact_table)} groups, {total_extra} extra images")
    print(f"Cross-split exact duplicate groups: {cross_split_count}")
    if exact_cross:
        for r in exact_cross[:5]:
            print(f"  CROSS-SPLIT: {r}")
    
    # Near duplicates
    near_table, near_cross = analyze_near_duplicates(rows, threshold=8)
    
    if near_table:
        with open(OUT_DIR / "near_duplicates_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=near_table[0].keys())
            w.writeheader()
            w.writerows(near_table)
    
    near_report = {
        "total_images": len(rows),
        "near_duplicate_groups": len(near_table),
        "cross_split_groups": len(near_cross),
        "table_path": str(OUT_DIR / "near_duplicates_table.csv"),
    }
    with open(OUT_DIR / "near_duplicates_report.json", "w") as f:
        json.dump(near_report, f, indent=2)
    
    print(f"\nNear duplicates: {len(near_table)} groups")
    print(f"Cross-split near duplicate groups: {len(near_cross)}")
    if near_cross:
        for r in near_cross[:5]:
            print(f"  CROSS-SPLIT NEAR: {r}")


if __name__ == "__main__":
    main()