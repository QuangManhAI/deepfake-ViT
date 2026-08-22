#!/usr/bin/env python3
"""
Extract official Celeb-DF-v2 Test Benchmark frames (518 videos from List_of_testing_videos.txt).
Creates test_CelebDFv2_balanced.csv, test_CelebDFv2_full.csv, and test_celeb_df_v2.csv.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _process_test_video_worker(item):
    rel_path_str, label_str, celeb_root_str, output_dir_str, frames_per_vid, resolution = item
    celeb_root = Path(celeb_root_str)
    output_dir = Path(output_dir_str)

    vid_path = celeb_root / rel_path_str
    if not vid_path.exists():
        return []

    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    label = int(label_str)
    # Note: in List_of_testing_videos.txt, '1' is Real, '0' is Fake
    # In standard convention: label 0 is Real, label 1 is Fake
    binary_label = 0 if label == 1 else 1
    method_name = "real" if binary_label == 0 else "CelebDFv2"

    positions = np.linspace(0.15, 0.85, frames_per_vid)
    records = []

    for f_idx, pos in enumerate(positions):
        frame_num = int(total_frames * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w, _ = frame.shape
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        square = frame[top : top + min_dim, left : left + min_dim]
        resized = cv2.resize(square, (resolution, resolution), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)

        prefix = "real" if binary_label == 0 else "fake"
        frame_filename = f"celeb_test_{prefix}_{vid_path.stem}_f{f_idx:02d}.png"
        save_path = output_dir / frame_filename
        im.save(save_path, "PNG", optimize=True)

        records.append(
            {
                "path": str(save_path),
                "label": binary_label,
                "method": method_name,
                "identity": f"cdc:{vid_path.stem}",
                "domain": "cdc",
                "video": vid_path.name,
                "source": "Celeb-DF-v2-Test",
            }
        )

    cap.release()
    return records


def extract_celeb_df_test_suite(
    celeb_df_root: Path,
    output_dir: Path,
    splits_dir: Path,
    frames_per_video: int = 5,
    resolution: int = 256,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = splits_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)

    test_list_txt = celeb_df_root / "List_of_testing_videos.txt"
    if not test_list_txt.exists():
        print(f"Error: {test_list_txt} not found!")
        return

    with open(test_list_txt, "r") as f:
        entries = [line.strip().split() for line in f if line.strip()]

    print(f"Found {len(entries)} official test video entries in {test_list_txt.name}.")

    items = [
        (parts[1], parts[0], str(celeb_df_root), str(output_dir), frames_per_video, resolution)
        for parts in entries
        if len(parts) >= 2
    ]

    extracted_records = []
    num_workers = min(16, os.cpu_count() or 8)
    print(f"Extracting test frames with {num_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_test_video_worker, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Celeb-DF-v2 Test Benchmark"):
            res = future.result()
            if res:
                extracted_records.extend(res)

    df_test = pd.DataFrame(extracted_records)
    print(f"\nExtracted {len(df_test):,} total test benchmark frames.")
    n_reals = (df_test["label"] == 0).sum()
    n_fakes = (df_test["label"] == 1).sum()
    print(f"  • Real Test Frames: {n_reals:,} (from 178 YouTube-real test videos)")
    print(f"  • Fake Test Frames: {n_fakes:,} (from 340 Celeb-synthesis test videos)")

    # 1. Full Celeb-DF-v2 Test Set
    df_test.to_csv(splits_dir / "test_celeb_df_v2.csv", index=False)
    df_test.to_csv(methods_dir / "test_CelebDFv2_full.csv", index=False)
    df_test.to_csv(methods_dir / "benchmark_test_CelebDFv2_full.csv", index=False)

    # 2. Balanced 1:1 Celeb-DF-v2 Test Set
    reals = df_test[df_test["label"] == 0]
    fakes = df_test[df_test["label"] == 1]
    n_bal = min(len(reals), len(fakes))
    df_bal = pd.concat([reals.sample(n_bal, random_state=42), fakes.sample(n_bal, random_state=42)]).sample(frac=1.0, random_state=42)

    df_bal.to_csv(splits_dir / "test_celeb_df_v2_balanced.csv", index=False)
    df_bal.to_csv(methods_dir / "test_CelebDFv2_balanced.csv", index=False)
    df_bal.to_csv(methods_dir / "benchmark_test_CelebDFv2_balanced.csv", index=False)

    print("=" * 80)
    print("CELEB-DF-V2 TEST BENCHMARK CREATED")
    print("=" * 80)
    print(f"   • Full Test CSV     : {splits_dir / 'test_celeb_df_v2.csv'} ({len(df_test):,} imgs)")
    print(f"   • Balanced Test CSV : {splits_dir / 'test_celeb_df_v2_balanced.csv'} ({len(df_bal):,} imgs, 1:1)")
    print(f"   • Method Test CSVs  : {methods_dir / 'test_CelebDFv2_balanced.csv'}, {methods_dir / 'test_CelebDFv2_full.csv'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb-root", type=str, default="/workspace/data/Celeb-DF-v2")
    parser.add_argument("--output-dir", type=str, default="/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_test_extracted")
    parser.add_argument("--splits-dir", type=str, default="/workspace/hoangtuan/deepfake-ViT/data/splits")
    parser.add_argument("--frames-per-video", type=int, default=5)
    args = parser.parse_args()

    extract_celeb_df_test_suite(
        Path(args.celeb_root),
        Path(args.output_dir),
        Path(args.splits_dir),
        frames_per_video=args.frames_per_video,
    )


if __name__ == "__main__":
    main()
