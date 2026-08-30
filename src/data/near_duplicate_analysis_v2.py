"""Recompute near-duplicate groups using union-find (disjoint set) for correctness."""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "eda_real_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_detailed(split):
    with open(SPLITS_DIR / f"{split}_detailed.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            r["split"] = split
            yield r


def average_hash_8x8(path):
    with Image.open(path) as img:
        img = img.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = np.array(img)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    bits_str = "".join("1" if b else "0" for b in bits)
    return hex(int(bits_str, 2))[2:].zfill(16)


def hamming(hex1, hex2):
    return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")


def split_quadrants(ahash):
    return [ahash[0:4], ahash[4:8], ahash[8:12], ahash[12:16]]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def main():
    print("="*80)
    print("Near-duplicate analysis v2 (union-find)")
    print("="*80)
    
    # Load all rows
    all_rows = []
    for split in ["train", "val", "test"]:
        all_rows.extend(list(load_detailed(split)))
    print(f"Loaded {len(all_rows)} rows")
    
    # Compute hashes
    print("Computing hashes...")
    t0 = time.time()
    hashes = {}
    for i, r in enumerate(all_rows):
        p = Path(r["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        hashes[r["path"]] = average_hash_8x8(str(p))
        if i % 5000 == 0:
            print(f"  {i:,}")
    print(f"  done in {time.time()-t0:.1f}s")
    
    # LSH bucketing
    print("Building LSH buckets...")
    buckets = defaultdict(list)
    for r in all_rows:
        h = hashes[r["path"]]
        for q in split_quadrants(h):
            buckets[q].append((r["path"], h))
    
    # Union-find for near-duplicates (threshold 8)
    print("Running union-find...")
    t0 = time.time()
    uf = UnionFind([r["path"] for r in all_rows])
    seen_pairs = set()
    
    for bucket, candidates in buckets.items():
        if len(candidates) < 2:
            continue
        for (p1, h1), (p2, h2) in combinations(candidates, 2):
            if h1 == h2:
                continue
            key = tuple(sorted([p1, p2]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if hamming(h1, h2) <= 8:
                uf.union(p1, p2)
    
    print(f"  union-find done in {time.time()-t0:.1f}s")
    
    # Build groups
    print("Building groups...")
    groups = defaultdict(list)
    for r in all_rows:
        groups[uf.find(r["path"])].append(r)
    
    # Filter and build table
    table = []
    cross = []
    for root, items in groups.items():
        if len(items) < 2:
            continue
        
        methods = sorted(set(r["method"] for r in items))
        labels = sorted(set(r["label"] for r in items))
        identities = sorted(set(r["identity"] for r in items))
        videos = sorted(set(r["video"] for r in items))
        splits = sorted(set(r["split"] for r in items))
        
        # Compute max hamming within group
        hs = [hashes[r["path"]] for r in items]
        max_h = 0
        for i, j in combinations(range(len(hs)), 2):
            d = hamming(hs[i], hs[j])
            if d > max_h:
                max_h = d
        
        row = {
            "canonical_path": root,
            "num_images": len(items),
            "max_hamming": max_h,
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
            cross.append(row)
    
    # Save
    with open(OUT_DIR / "near_duplicates_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)
    
    report = {
        "total_groups": len(table),
        "cross_split_groups": len(cross),
        "total_images_in_groups": sum(int(r["num_images"]) for r in table),
        "table_path": str(OUT_DIR / "near_duplicates_table.csv"),
    }
    with open(OUT_DIR / "near_duplicates_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nNear-duplicate groups: {len(table)}")
    print(f"Cross-split groups: {len(cross)}")
    print(f"Total images in groups: {report['total_images_in_groups']}")
    print(f"Saved table: {report['table_path']}")


if __name__ == "__main__":
    main()