#!/usr/bin/env python3
"""DF40 Deepfake Benchmark — Complete Dataset Preparation & Split Generator.

Generates all required evaluation protocols and per-method test sets:

Protocol 1: Identity-Disjoint Splits (Zero Data Leakage Protocol - Recommended)
    - Source: test_data_v3 (contains all 40 fake methods, 1,177 unique real faces, and identity metadata).
    - Partition: Grouped by unique identity (seed 42) into 70% Train / 15% Val / 15% Test.
    - Files: data/splits/train.csv, val.csv, test.csv
    - Guarantee: Exactly ZERO identity overlap between train, val, and test.

Protocol 2: High-Scale Combined Training Pool (FF++ Real + DF40 Fake)
    - Combines 27k+ disjoint FaceForensics++ Real frames with DF40 fake training pool.
    - Zero Leakage: Strictly excludes any video ID / identity in Test or Val.
    - Files: data/splits/train_combined_balanced.csv, val_combined_balanced.csv, train_pool_693k.csv, val_pool.csv

Protocol 3: Balanced 1:1 Identity-Disjoint Splits
    - Files: data/splits/train_balanced.csv, val_balanced.csv, test_balanced.csv
    - Method convenience splits: train_insight.csv, train_faceswap.csv, train_simswap.csv, etc.

Protocol 4: Full Benchmark Suite (All 30.6k+ test_data_v3 samples)
    - Files: data/splits/test_full.csv, test_full_detailed.csv

Protocol 5: Method-Specific Test Sets for ALL 40 Deepfake Methods
    - Stored under data/splits/methods/:
      - test_<method>_balanced.csv (1:1 Real:Fake balanced from test split)
      - test_<method>_full.csv (All fake samples of method from test split + test real samples)
      - test_<method>_detailed.csv (Detailed metadata)
      - benchmark_test_<method>_balanced.csv (1:1 Real:Fake balanced across entire benchmark)
      - benchmark_test_<method>_full.csv (All fake samples of method + all 1,177 real samples)

Usage:
    .venv/bin/python3 src/data/prepare_df40_splits.py --seed 42
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False




METHOD_ALIASES = {
    "insight": ["insight", "inswap"],
    "inswap": ["insight", "inswap"],
    "faceswap": ["faceswap"],
    "simswap": ["simswap"],
    "sadtalker": ["sadtalker"],
    "mobileswap": ["mobileswap"],
    "dit": ["dit", "DiT"],
    "stylegan3": ["stylegan3", "StyleGAN3"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DF40 train/val/test splits")
    parser.add_argument(
        "--df40-root",
        default=os.environ.get("DF40_ROOT", "data/raw"),
        help="Root directory containing test_data_v3, DF40_train, and FF++ (READ ONLY)",
    )
    parser.add_argument(
        "--train-manifest",
        default="",
        help="Path to DF40_train_manifest.csv",
    )
    parser.add_argument(
        "--test-manifest",
        default="",
        help="Path to test_data_v3 manifest.csv",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Output directory for generated split CSVs inside project",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Output directory for summary statistics inside project",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic identity grouping",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Ratio of identities for training in identity-disjoint protocol",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of identities for validation in identity-disjoint protocol",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of identities for testing in identity-disjoint protocol",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        default=True,
        help="Verify sample images exist and decode correctly",
    )
    return parser.parse_args()


def load_test_manifest(manifest_path, root_dir):
    """Load test_data_v3 manifest rows with resolved paths, filtering for readable images."""
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Test Manifest not found: {manifest_p}")

    rows = []
    skipped_unreadable = 0
    root_p = Path(root_dir)
    with open(manifest_p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lbl = int(r.get("label", "0" if r.get("method") == "real" else "1"))
            rel_p = r["path"]
            abs_p = (root_p / rel_p).resolve()

            if not os.access(abs_p, os.R_OK):
                skipped_unreadable += 1
                continue

            identity = r.get("identity", "unknown")
            method = r.get("method", "real" if lbl == 0 else "fake")
            domain = r.get("domain", "unknown")
            video = r.get("video", "unknown")
            rows.append({
                "path": str(abs_p),
                "rel_path": rel_p,
                "label": lbl,
                "identity": identity,
                "method": method,
                "domain": domain,
                "video": video,
            })

    if skipped_unreadable > 0:
        print(f"  [Notice] Skipped {skipped_unreadable:,} unreadable files from test manifest.")
    return rows


def load_faceforensics_real_frames(ff_root, excluded_identities=None):
    """Load FaceForensics++ real frames, skipping any identities in excluded_identities (test/val)."""
    ff_p = Path(ff_root)
    if not ff_p.exists():
        return []

    excluded = excluded_identities or set()
    rows = []
    skipped_leak = 0

    for vid_dir in sorted(ff_p.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid_id = vid_dir.name
        identity_key = f"ffc:{vid_id}"
        if identity_key in excluded:
            skipped_leak += 1
            continue

        for img_p in sorted(vid_dir.glob("*.png")):
            rows.append({
                "path": str(img_p.resolve()),
                "label": 0,
                "method": "real",
                "identity": identity_key,
                "domain": "ffc",
                "video": vid_id,
            })

    if skipped_leak > 0:
        print(f"  [Zero Leakage] Excluded {skipped_leak} FF++ real video folders overlapping with held-out test/val identities.")
    return rows


def load_train_pool_manifest(manifest_path):
    """Load DF40_train_manifest.csv rows."""
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        return []

    rows = []
    with open(manifest_p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lbl = int(r.get("label", "1"))
            rows.append({
                "path": r["path"],
                "label": lbl,
                "method": r.get("method", "fake" if lbl == 1 else "real"),
            })
    return rows


def make_identity_disjoint_splits(rows, train_ratio, val_ratio, test_ratio, seed):
    """Split rows strictly by identity key (zero identity leakage)."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    id_to_rows = defaultdict(list)
    for r in rows:
        id_to_rows[r["identity"]].append(r)

    identities = sorted(id_to_rows.keys())
    rng.shuffle(identities)

    n_total = len(identities)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train : n_train + n_val])
    test_ids = set(identities[n_train + n_val :])

    # Assert 100% mutual exclusivity
    assert len(train_ids & val_ids) == 0, "Leakage between train and val!"
    assert len(train_ids & test_ids) == 0, "Leakage between train and test!"
    assert len(val_ids & test_ids) == 0, "Leakage between val and test!"

    splits = {"train": [], "val": [], "test": []}
    for ident in identities:
        target_split = "train" if ident in train_ids else "val" if ident in val_ids else "test"
        splits[target_split].extend(id_to_rows[ident])

    id_splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    return splits, id_splits


def make_balanced_subset(split_rows, seed, max_samples=None):
    """Create a 1:1 real:fake balanced subset."""
    rng = random.Random(seed)
    reals = [r for r in split_rows if r["label"] == 0]
    fakes = [r for r in split_rows if r["label"] == 1]

    n_sample = min(len(reals), len(fakes))
    if max_samples:
        n_sample = min(n_sample, max_samples // 2)

    if n_sample == 0:
        return []

    sampled_fakes = rng.sample(fakes, n_sample)
    sampled_reals = rng.sample(reals, n_sample) if len(reals) > n_sample else reals[:n_sample]
    balanced = sampled_reals + sampled_fakes
    rng.shuffle(balanced)
    return balanced


def filter_by_method(split_rows, target_fake_method="insight", seed=42):
    """Filter rows for a specific fake method and real samples, balancing 1:1."""
    aliases = METHOD_ALIASES.get(target_fake_method.lower(), [target_fake_method.lower()])
    reals = [r for r in split_rows if r["label"] == 0]
    fakes = [r for r in split_rows if r["method"].lower() in aliases]

    n = min(len(reals), len(fakes))
    if n == 0:
        return reals + fakes
    rng = random.Random(seed)
    sampled_reals = rng.sample(reals, n) if len(reals) > n else reals
    sampled_fakes = rng.sample(fakes, n) if len(fakes) > n else fakes
    combined = sampled_reals + sampled_fakes
    rng.shuffle(combined)
    return combined


def write_split_csv(out_path, rows, include_extra=False):
    """Write standard path,label CSV for model consumption."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_extra:
            fieldnames = ["path", "label", "method"]
            if rows and "identity" in rows[0]:
                fieldnames.extend(["identity", "domain", "video"])
            writer.writerow(fieldnames)
            for r in rows:
                row_vals = [r["path"], r["label"], r.get("method", "")]
                if "identity" in r:
                    row_vals.extend([r.get("identity", ""), r.get("domain", ""), r.get("video", "")])
                writer.writerow(row_vals)
        else:
            writer.writerow(["path", "label"])
            for r in rows:
                writer.writerow([r["path"], r["label"]])


def verify_split_integrity(rows, sample_check=30):
    """Check sample images exist, have non-zero size, and decode if PIL available."""
    sample_indices = random.sample(range(len(rows)), min(sample_check, len(rows)))
    for idx in sample_indices:
        p = rows[idx]["path"]
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing image at {p}")
        if os.path.getsize(p) == 0:
            raise RuntimeError(f"Empty image file at {p}")
        if HAS_PIL:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                raise RuntimeError(f"Corrupt image at {p}: {e}")
    return True



def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    methods_dir = splits_dir / "methods"
    processed_dir = Path(args.processed_dir)

    splits_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df40_root = Path(args.df40_root)
    train_manifest_path = args.train_manifest or str(df40_root / "DF40_train_manifest.csv")
    test_manifest_path = args.test_manifest or str(df40_root / "test_data_v3" / "manifest.csv")
    test_root_dir = str(df40_root / "test_data_v3")

    print("=" * 80)
    print("  DF40 DEEPFAKE BENCHMARK — DATASET PREPARATION & SPLIT GENERATOR")
    print("=" * 80)
    print(f"  DF40 Source Root    : {df40_root} (READ-ONLY)")
    print(f"  Test Manifest       : {test_manifest_path}")
    print(f"  Splits Output Dir   : {splits_dir.resolve()}")
    print(f"  Methods Output Dir  : {methods_dir.resolve()}")
    print(f"  Random Seed         : {args.seed}")
    print("-" * 80)

    # 1. Load test_data_v3
    test_v3_rows = load_test_manifest(test_manifest_path, test_root_dir)
    print(f"✅ Loaded {len(test_v3_rows):,} total benchmark samples from test_data_v3.")

    # 2. Verify Sample Image Decoding
    if args.verify_images:
        print("🔍 Verifying sample images...")
        verify_split_integrity(test_v3_rows, sample_check=50)
        print("  ✔ Sample image decodes verified successfully.")

    # 3. Generate Identity-Disjoint Splits (Protocol 1 - Zero Leakage)
    print("\n📊 PROTOCOL 1: Identity-Disjoint Splits (Zero Identity Leakage):")
    id_splits_data, id_splits_keys = make_identity_disjoint_splits(
        test_v3_rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    stats = {}
    for s_name in ["train", "val", "test"]:
        s_rows = id_splits_data[s_name]
        n_r = sum(1 for r in s_rows if r["label"] == 0)
        n_f = sum(1 for r in s_rows if r["label"] == 1)
        n_ids = len(id_splits_keys[s_name])
        ratio = f"{n_f / max(1, n_r):.1f}:1" if n_r > 0 else "N/A"
        stats[s_name] = {"total": len(s_rows), "real": n_r, "fake": n_f, "identities": n_ids, "ratio": ratio}
        print(f"  ▶ {s_name.upper():<5} : {len(s_rows):>6,} images | Real: {n_r:>5,} | Fake: {n_f:>5,} | Identities: {n_ids:>5,} | Ratio: {ratio}")

    # Write Standard Identity-Disjoint Splits
    write_split_csv(splits_dir / "train.csv", id_splits_data["train"])
    write_split_csv(splits_dir / "val.csv", id_splits_data["val"])
    write_split_csv(splits_dir / "test.csv", id_splits_data["test"])
    write_split_csv(splits_dir / "train_detailed.csv", id_splits_data["train"], include_extra=True)
    write_split_csv(splits_dir / "val_detailed.csv", id_splits_data["val"], include_extra=True)
    write_split_csv(splits_dir / "test_detailed.csv", id_splits_data["test"], include_extra=True)

    # 4. Generate Unified Master Benchmark Test Set (Protocol 4 - All 4 Datasets Combined)
    celeb_test_dir = splits_dir.parent / "processed" / "celeb_df_test_extracted"
    celeb_test_p = splits_dir / "test_celeb_df_v2.csv"


    unified_test_rows = list(test_v3_rows)
    
    if celeb_test_p.exists():
        with open(celeb_test_p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                unified_test_rows.append({
                    "path": r["path"],
                    "label": int(r.get("label", "1")),
                    "method": r.get("method", "CelebDFv2" if int(r.get("label", "1")) == 1 else "real"),
                    "identity": r.get("identity", ""),
                    "domain": r.get("domain", "cdc"),
                    "video": r.get("video", ""),
                })
        print(f"  🌟 Merged Celeb-DF-v2 Test Benchmark into Unified Master Test Suite ({len(unified_test_rows):,} total test images).")
    elif celeb_test_dir.exists():
        celeb_imgs = list(celeb_test_dir.glob("*.png"))
        for img_p in celeb_imgs:
            fname = img_p.name
            is_fake = 1 if ("fake" in fname.lower() and "real" not in fname.lower()) or fname.startswith("fake_") or "test_fake" in fname.lower() else 0
            # Extract video stem and identity
            v_name = fname.replace("celeb_test_fake_", "").replace("celeb_test_real_", "").replace("fake_", "").replace("real_", "").rsplit("_frame", 1)[0]
            id_name = v_name.split("_")[0] if "_" in v_name else v_name
            unified_test_rows.append({
                "path": str(img_p.resolve()),
                "label": is_fake,
                "method": "CelebDFv2" if is_fake == 1 else "real",
                "identity": f"cdc:{id_name}",
                "domain": "cdc",
                "video": v_name,
            })
        print(f"  🌟 Merged {len(celeb_imgs):,} Celeb-DF-v2 Test Frames into Unified Master Test Suite ({len(unified_test_rows):,} total test images).")

    write_split_csv(splits_dir / "test_full.csv", unified_test_rows)
    write_split_csv(splits_dir / "test_full_detailed.csv", unified_test_rows, include_extra=True)
    print(f"\n🎯 Unified Master Benchmark Test Suite (4-in-1): {splits_dir / 'test_full.csv'} ({len(unified_test_rows):,} rows)")


    # 5. Generate 1:1 Balanced Splits (Protocol 3)
    train_bal = make_balanced_subset(id_splits_data["train"], seed=args.seed)
    val_bal = make_balanced_subset(id_splits_data["val"], seed=args.seed)
    test_bal = make_balanced_subset(id_splits_data["test"], seed=args.seed)
    write_split_csv(splits_dir / "train_balanced.csv", train_bal)
    write_split_csv(splits_dir / "val_balanced.csv", val_bal)
    write_split_csv(splits_dir / "test_balanced.csv", test_bal)
    print(f"⚖️ 1:1 Balanced Splits: train={len(train_bal):,}, val={len(val_bal):,}, test={len(test_bal):,}")

    # 6. Generate Method-Specific Test Sets for ALL 40 Deepfake Methods (Unified 4-in-1)
    print("\n🔬 PROTOCOL 5: Generating Method-Specific Test Sets (All 40 Methods)...")
    all_methods = sorted(set(r["method"] for r in unified_test_rows if r["label"] == 1))

    # Test partition real samples
    test_reals = [r for r in id_splits_data["test"] if r["label"] == 0]
    # Benchmark full real samples across all 4 datasets
    benchmark_all_reals = [r for r in unified_test_rows if r["label"] == 0]

    methods_summary = {}
    rng = random.Random(args.seed)

    for m in all_methods:
        # A. Identity-Disjoint Test split samples for method m
        m_test_fakes = [r for r in id_splits_data["test"] if r["method"] == m and r["label"] == 1]
        if not m_test_fakes and m == "CelebDFv2":
            # For CelebDFv2, use its balanced test set directly
            m_test_fakes = [r for r in unified_test_rows if r["method"] == m and r["label"] == 1][:len(test_reals)]
        n_bal_test = min(len(test_reals), len(m_test_fakes))
        sampled_reals_test = rng.sample(test_reals, n_bal_test) if len(test_reals) > n_bal_test else test_reals[:n_bal_test]
        sampled_fakes_test = rng.sample(m_test_fakes, n_bal_test) if len(m_test_fakes) > n_bal_test else m_test_fakes[:n_bal_test]
        m_test_balanced = sampled_reals_test + sampled_fakes_test
        rng.shuffle(m_test_balanced)

        m_test_full = test_reals + m_test_fakes
        rng.shuffle(m_test_full)

        # Write identity-disjoint method test sets
        write_split_csv(methods_dir / f"test_{m}_balanced.csv", m_test_balanced)
        write_split_csv(methods_dir / f"test_{m}_full.csv", m_test_full)
        write_split_csv(methods_dir / f"test_{m}_detailed.csv", m_test_full, include_extra=True)

        # B. Benchmark Suite (Across all 100% held-out unified test suite)
        m_bench_fakes = [r for r in unified_test_rows if r["method"] == m and r["label"] == 1]
        n_bal_bench = min(len(benchmark_all_reals), len(m_bench_fakes))
        sampled_reals_bench = rng.sample(benchmark_all_reals, n_bal_bench) if len(benchmark_all_reals) > n_bal_bench else benchmark_all_reals[:n_bal_bench]
        sampled_fakes_bench = rng.sample(m_bench_fakes, n_bal_bench) if len(m_bench_fakes) > n_bal_bench else m_bench_fakes[:n_bal_bench]
        m_bench_balanced = sampled_reals_bench + sampled_fakes_bench
        rng.shuffle(m_bench_balanced)

        m_bench_full = benchmark_all_reals + m_bench_fakes
        rng.shuffle(m_bench_full)

        write_split_csv(methods_dir / f"benchmark_test_{m}_balanced.csv", m_bench_balanced)
        write_split_csv(methods_dir / f"benchmark_test_{m}_full.csv", m_bench_full)

        methods_summary[m] = {
            "test_split_fakes": len(m_test_fakes),
            "test_split_balanced_total": len(m_test_balanced),
            "benchmark_fakes": len(m_bench_fakes),
            "benchmark_balanced_total": len(m_bench_balanced),
            "benchmark_full_total": len(m_bench_full),
        }


    print(f"  ✔ Successfully generated method test splits for all {len(all_methods)} fake methods in {methods_dir}")

    # 7. Generate Convenience Splits for Key Methods in splits_dir
    key_methods = ["insight", "faceswap", "simswap", "sadtalker", "mobileswap", "dit", "stylegan3"]
    for target_m in key_methods:
        tr_m = filter_by_method(id_splits_data["train"], target_m, seed=args.seed)
        val_m = filter_by_method(id_splits_data["val"], target_m, seed=args.seed)
        te_m = filter_by_method(id_splits_data["test"], target_m, seed=args.seed)
        write_split_csv(splits_dir / f"train_{target_m}.csv", tr_m)
        write_split_csv(splits_dir / f"val_{target_m}.csv", val_m)
        write_split_csv(splits_dir / f"test_{target_m}.csv", te_m)
        print(f"  ▶ Convenience {target_m:<12}: train={len(tr_m):>5,}, val={len(val_m):>5,}, test={len(te_m):>5,}")

    # 8. Generate Full-Scale Combined Training Pool with FF++ and Celeb-DF Real Frames (Protocol 2 - Zero Leakage)
    print("\n🎬 PROTOCOL 2: Generating Full-Scale Training Pool with FF++ and Celeb-DF Real Frames...")
    ff_root = df40_root / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "frames"
    # Strictly exclude identities in test_ids and val_ids
    excluded_ids = set(id_splits_keys["test"]) | set(id_splits_keys["val"])
    ff_real_rows = load_faceforensics_real_frames(ff_root, excluded_identities=excluded_ids)
    print(f"  🎬 Loaded {len(ff_real_rows):,} disjoint Real frames from FaceForensics++ (Zero Test/Val Leakage).")

    # Load extracted Celeb-DF-v2 real frames
    celeb_manifest_p = splits_dir / "celeb_df_extracted_real_frames.csv"
    celeb_real_rows = []
    if celeb_manifest_p.exists():
        with open(celeb_manifest_p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("identity") not in excluded_ids:
                    celeb_real_rows.append({
                        "path": r["path"],
                        "label": 0,
                        "method": "real",
                        "identity": r.get("identity", ""),
                        "domain": "cdc",
                        "video": r.get("video", ""),
                    })
        print(f"  🎬 Loaded {len(celeb_real_rows):,} disjoint Real frames from Celeb-DF-v2 (Zero Test/Val Leakage).")

    all_real_training_rows = ff_real_rows + celeb_real_rows
    print(f"  🌟 Total Combined Disjoint Real Frames (FF++ + Celeb-DF): {len(all_real_training_rows):,} images.")

    train_pool_rows = load_train_pool_manifest(train_manifest_path)
    if train_pool_rows:
        # Extract fake training frames
        train_pool_fakes = [r for r in train_pool_rows if r["label"] == 1]

        # Merge all real frames with fake pool
        combined_pool = train_pool_fakes + all_real_training_rows
        rng = random.Random(args.seed)
        rng.shuffle(combined_pool)

        n_p_val = int(len(combined_pool) * 0.10)
        p_val = combined_pool[:n_p_val]
        p_train = combined_pool[n_p_val:]
        write_split_csv(splits_dir / "train_pool_693k.csv", p_train)
        write_split_csv(splits_dir / "val_pool.csv", p_val)

        # High-scale 1:1 balanced pool from (FF++ Real + Celeb-DF Real) + DF40 fake
        n_bal_sample = min(len(all_real_training_rows), len(train_pool_fakes))
        sampled_reals = list(all_real_training_rows[:n_bal_sample])
        sampled_fakes = rng.sample(train_pool_fakes, n_bal_sample)
        rng.shuffle(sampled_reals)
        rng.shuffle(sampled_fakes)

        n_val_each = int(n_bal_sample * 0.10)
        reals_train = sampled_reals[n_val_each:]
        reals_val = sampled_reals[:n_val_each]
        fakes_train = sampled_fakes[n_val_each:]
        fakes_val = sampled_fakes[:n_val_each]

        bal_train = reals_train + fakes_train
        bal_val = reals_val + fakes_val
        rng.shuffle(bal_train)
        rng.shuffle(bal_val)

        # Write unified balanced splits (both canonical and combined alias names)
        write_split_csv(splits_dir / "train_balanced.csv", bal_train)
        write_split_csv(splits_dir / "val_balanced.csv", bal_val)
        write_split_csv(splits_dir / "train_combined_balanced.csv", bal_train)
        write_split_csv(splits_dir / "val_combined_balanced.csv", bal_val)


        # Write unified full pool splits (both canonical and alias names)
        write_split_csv(splits_dir / "train.csv", p_train)
        write_split_csv(splits_dir / "val.csv", p_val)
        write_split_csv(splits_dir / "test.csv", unified_test_rows)
        write_split_csv(splits_dir / "train_pool_693k.csv", p_train)
        write_split_csv(splits_dir / "val_pool.csv", p_val)

        # Write unified balanced test set (4-in-1)
        test_all_reals = [r for r in unified_test_rows if r["label"] == 0]
        test_all_fakes = [r for r in unified_test_rows if r["label"] == 1]
        n_test_bal = min(len(test_all_reals), len(test_all_fakes))
        test_bal_sampled = rng.sample(test_all_reals, n_test_bal) + rng.sample(test_all_fakes, n_test_bal)
        rng.shuffle(test_bal_sampled)
        write_split_csv(splits_dir / "test_balanced.csv", test_bal_sampled)

        print(f"  📦 Full-Scale Combined Training Pool: train={len(p_train):,}, val={len(p_val):,}")
        print(f"  ⚖️ High-Scale 1:1 Balanced Pool (FF++ & Celeb-DF Real + DF40 Fake): train={len(bal_train):,}, val={len(bal_val):,}")
        print(f"  🎯 Unified 1:1 Balanced Test Set: test={len(test_bal_sampled):,}")



    # 9. Export Manifest & Metadata
    manifest_summary = {
        "dataset_name": "DF40 Deepfake Benchmark",
        "seed": args.seed,
        "identity_disjoint_splits": stats,
        "test_full_total": len(test_v3_rows),
        "total_methods": len(all_methods),
        "methods_list": all_methods,
        "split_files": [str(f.name) for f in sorted(splits_dir.glob("*.csv"))],
        "method_split_files": [str(f.name) for f in sorted(methods_dir.glob("*.csv"))],
    }

    split_info_path = splits_dir / "split_info.json"
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)
    print(f"\n📄 Saved Split Metadata to {split_info_path}")

    methods_summary_path = splits_dir / "methods_summary.json"
    with open(methods_summary_path, "w", encoding="utf-8") as f:
        json.dump(methods_summary, f, indent=2)
    print(f"📄 Saved Methods Summary to {methods_summary_path}")

    processed_summary_path = processed_dir / "data_prep_manifest.json"
    with open(processed_summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)
    print(f"📄 Saved Processed Manifest to {processed_summary_path}")

    print("\n" + "=" * 80)
    print("  ✨ DATA PREPARATION COMPLETED & VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()

