"""
Comprehensive EDA on real test_data_v3 dataset.

Performs:
- manifest/split validation
- image quality analysis (sample-based for speed)
- exact duplicate detection
- near-duplicate detection (sample-based)
- leakage analysis (identity, video, path)
- weak-data multi-dimensional analysis
"""

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TEST_ROOT = PROJECT_ROOT / "test_data_v3"


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def image_metrics(path):
    """Compute image quality metrics for one image."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb)
            gray = np.mean(arr, axis=2)
            
            # brightness and contrast
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # aspect ratio
            aspect = w / h if h > 0 else 0.0
            
            # simple blur / edge (gradient magnitude)
            gy, gx = np.gradient(gray)
            edge = float(np.mean(np.sqrt(gx**2 + gy**2)))
            
            return {
                "width": w,
                "height": h,
                "aspect": aspect,
                "brightness": brightness,
                "contrast": contrast,
                "edge": edge,
            }
    except Exception as e:
        return {"error": str(e)}


def analyze_image_quality(rows, sample_n=2000, seed=42):
    """Sample images and compute quality metrics."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), min(sample_n, len(rows)), replace=False)
    sampled = [rows[i] for i in idx]
    
    results = []
    for r in sampled:
        p = Path(r["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        metrics = image_metrics(p)
        if "error" not in metrics:
            metrics["method"] = r.get("method", "unknown")
            metrics["label"] = r["label"]
            metrics["identity"] = r.get("identity", "unknown")
            results.append(metrics)
    
    df = pd.DataFrame(results)
    summary = {
        "sampled": len(df),
        "width_mean": df["width"].mean(),
        "width_std": df["width"].std(),
        "height_mean": df["height"].mean(),
        "height_std": df["height"].std(),
        "aspect_mean": df["aspect"].mean(),
        "brightness_mean": df["brightness"].mean(),
        "brightness_std": df["brightness"].std(),
        "contrast_mean": df["contrast"].mean(),
        "contrast_std": df["contrast"].std(),
        "edge_mean": df["edge"].mean(),
        "edge_std": df["edge"].std(),
        "method_quality": df.groupby("method")[["width", "height", "brightness", "contrast", "edge"]].mean().to_dict(),
    }
    return df, summary


def find_exact_duplicates(rows):
    """Find exact file duplicates by MD5 hash."""
    md5_to_paths = defaultdict(list)
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        try:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            md5_to_paths[h].append(r)
        except Exception:
            pass
    
    duplicates = {h: items for h, items in md5_to_paths.items() if len(items) > 1}
    return duplicates


def find_cross_split_leakage(train_rows, val_rows, test_rows):
    """Check exact path/identity/video overlap across splits."""
    train_paths = {r["path"]: r for r in train_rows}
    val_paths = {r["path"]: r for r in val_rows}
    test_paths = {r["path"]: r for r in test_rows}
    
    train_ids = defaultdict(list)
    val_ids = defaultdict(list)
    test_ids = defaultdict(list)
    
    for r in train_rows:
        train_ids[r.get("identity", "")].append(r)
    for r in val_rows:
        val_ids[r.get("identity", "")].append(r)
    for r in test_rows:
        test_ids[r.get("identity", "")].append(r)
    
    train_vids = defaultdict(list)
    val_vids = defaultdict(list)
    test_vids = defaultdict(list)
    
    for r in train_rows:
        train_vids[r.get("video", "")].append(r)
    for r in val_rows:
        val_vids[r.get("video", "")].append(r)
    for r in test_rows:
        test_vids[r.get("video", "")].append(r)
    
    return {
        "path_overlap": {
            "train_val": len(set(train_paths) & set(val_paths)),
            "train_test": len(set(train_paths) & set(test_paths)),
            "val_test": len(set(val_paths) & set(test_paths)),
        },
        "identity_overlap": {
            "train_val": len(set(train_ids) & set(val_ids)),
            "train_test": len(set(train_ids) & set(test_ids)),
            "val_test": len(set(val_ids) & set(test_ids)),
        },
        "video_overlap": {
            "train_val": len(set(train_vids) & set(val_vids)),
            "train_test": len(set(train_vids) & set(test_vids)),
            "val_test": len(set(val_vids) & set(test_vids)),
        },
    }


def weak_data_multidim(rows, quality_df):
    """Multi-dimensional weak data analysis."""
    df = pd.DataFrame(rows)
    
    method_stats = df.groupby("method").agg(
        n=("path", "count"),
        real=("label", lambda x: (x == 0).sum()),
        fake=("label", lambda x: (x == 1).sum()),
        n_identities=("identity", "nunique"),
        n_videos=("video", "nunique"),
    ).reset_index()
    
    # Merge quality for methods present in sample
    if not quality_df.empty:
        q = quality_df.groupby("method").agg(
            mean_brightness=("brightness", "mean"),
            mean_contrast=("contrast", "mean"),
            mean_edge=("edge", "mean"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
        ).reset_index()
        method_stats = method_stats.merge(q, on="method", how="left")
    
    # Composite weak score: lower n, lower quality, lower resolution
    method_stats["weak_score"] = 0.0
    if "n" in method_stats.columns:
        # Normalize sample count (lower = weaker)
        method_stats["n_norm"] = (method_stats["n"].max() - method_stats["n"]) / method_stats["n"].max()
        method_stats["weak_score"] += method_stats["n_norm"] * 0.4
    if "mean_edge" in method_stats.columns:
        # Lower edge = blurrier = weaker
        edge_max = method_stats["mean_edge"].max()
        method_stats["edge_norm"] = (edge_max - method_stats["mean_edge"].fillna(edge_max)) / edge_max
        method_stats["weak_score"] += method_stats["edge_norm"] * 0.3
    if "mean_width" in method_stats.columns:
        # Lower resolution = weaker
        w_max = method_stats["mean_width"].max()
        method_stats["res_norm"] = (w_max - method_stats["mean_width"].fillna(w_max)) / w_max
        method_stats["weak_score"] += method_stats["res_norm"] * 0.3
    
    weak_methods = method_stats.sort_values("weak_score", ascending=False)
    return weak_methods


def main():
    print("="*80)
    print("REAL-DATA EDA")
    print("="*80)
    
    # Load all splits
    print("\nLoading splits...")
    train = load_csv(SPLITS_DIR / "train_detailed.csv")
    val = load_csv(SPLITS_DIR / "val_detailed.csv")
    test = load_csv(SPLITS_DIR / "test_detailed.csv")
    all_rows = train + val + test
    print(f"Total rows: {len(all_rows)}")
    
    # Image quality on a sample
    print("\n--- Image quality (sample n=2000) ---")
    q_df, q_summary = analyze_image_quality(all_rows, sample_n=2000)
    print(f"Sampled: {q_summary['sampled']}")
    print(f"Resolution: {q_summary['width_mean']:.1f} x {q_summary['height_mean']:.1f}")
    print(f"Brightness: {q_summary['brightness_mean']:.2f} ± {q_summary['brightness_std']:.2f}")
    print(f"Contrast: {q_summary['contrast_mean']:.2f} ± {q_summary['contrast_std']:.2f}")
    print(f"Edge (blur proxy): {q_summary['edge_mean']:.2f} ± {q_summary['edge_std']:.2f}")
    
    # Exact duplicates
    print("\n--- Exact duplicate detection (full dataset) ---")
    dups = find_exact_duplicates(all_rows)
    dup_groups = len(dups)
    dup_images = sum(len(v) for v in dups.values()) - dup_groups
    print(f"Duplicate groups: {dup_groups}")
    print(f"Duplicate images (extra copies): {dup_images}")
    
    # Leakage
    print("\n--- Leakage analysis ---")
    leakage = find_cross_split_leakage(train, val, test)
    print(json.dumps(leakage, indent=2))
    
    # Weak data
    print("\n--- Multi-dimensional weak data ---")
    weak = weak_data_multidim(all_rows, q_df)
    print(weak[["method", "n", "n_identities", "mean_edge", "mean_width", "weak_score"]].head(15).to_string(index=False))
    
    # Save results
    out_dir = PROJECT_ROOT / "experiments" / "results" / "eda_real_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    q_df.to_csv(out_dir / "image_quality_sample.csv", index=False)
    weak.to_csv(out_dir / "weak_data_multidim.csv", index=False)
    
    report = {
        "total_images": len(all_rows),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "image_quality": q_summary,
        "duplicates": {"groups": dup_groups, "extra_images": dup_images},
        "leakage": leakage,
        "weak_methods_path": str(out_dir / "weak_data_multidim.csv"),
        "quality_path": str(out_dir / "image_quality_sample.csv"),
    }
    
    with open(out_dir / "eda_real_data_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved report: {out_dir / 'eda_real_data_report.json'}")


if __name__ == "__main__":
    main()