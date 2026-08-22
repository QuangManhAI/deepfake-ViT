#!/usr/bin/env python3
"""
Extract Real Face Frames from Celeb-DF-v2 with strict Zero-Leakage isolation.
Extracts 15 frames per video from Celeb-real and YouTube-real (excluding Test and Val videos).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _process_single_video_worker(item):
    vid_path, ident_key, src_type, output_dir_str, frames_per_video, resolution = item
    output_dir = Path(output_dir_str)
    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    positions = np.linspace(0.1, 0.9, frames_per_video)
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

        frame_filename = f"{vid_path.stem}_f{f_idx:02d}.png"
        save_path = output_dir / frame_filename
        im.save(save_path, "PNG", optimize=True)

        records.append(
            {
                "path": str(save_path),
                "label": 0,
                "method": "real",
                "identity": ident_key,
                "domain": "cdc",
                "video": vid_path.name,
                "source": src_type,
            }
        )

    cap.release()
    return records


def extract_celeb_df_real_frames(
    celeb_df_root: Path,
    output_dir: Path,
    splits_dir: Path,
    frames_per_video: int = 15,
    resolution: int = 256,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify Test and Val exclusions
    test_exclusions = set()

    # Load List_of_testing_videos.txt if exists
    test_list_txt = celeb_df_root / "List_of_testing_videos.txt"
    if test_list_txt.exists():
        with open(test_list_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_exclusions.add(Path(parts[1]).stem)

    # Load test_detailed.csv and val_detailed.csv exclusions
    for split_name in ["test_detailed.csv", "val_detailed.csv"]:
        split_p = splits_dir / split_name
        if split_p.exists():
            df_split = pd.read_csv(split_p)
            for _, row in df_split.iterrows():
                if str(row.get("domain", "")) == "cdc":
                    ident = str(row.get("identity", ""))
                    if ident.startswith("cdc:"):
                        test_exclusions.add(ident.replace("cdc:", ""))
                    vid = str(row.get("video", ""))
                    if vid and vid != "nan":
                        test_exclusions.add(Path(vid).stem)

    print(f"Loaded {len(test_exclusions)} Celeb-DF video stems to exclude (0% leakage guarantee).")

    # 2. Gather eligible videos
    video_sources = []

    # Celeb-real
    celeb_real_dir = celeb_df_root / "Celeb-real"
    if celeb_real_dir.exists():
        for vid in sorted(celeb_real_dir.glob("*.mp4")):
            if vid.stem not in test_exclusions:
                video_sources.append((vid, f"cdc:{vid.stem}", "Celeb-real", str(output_dir), frames_per_video, resolution))

    # YouTube-real
    yt_real_dir = celeb_df_root / "YouTube-real"
    if yt_real_dir.exists():
        for vid in sorted(yt_real_dir.glob("*.mp4")):
            if vid.stem not in test_exclusions:
                video_sources.append((vid, f"cdc:{vid.stem}", "YouTube-real", str(output_dir), frames_per_video, resolution))

    print(f"Found {len(video_sources)} eligible Real videos for training extraction.")

    # 3. Fast Parallel Extraction
    extracted_records = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = min(16, os.cpu_count() or 8)
    print(f"Running multi-threaded frame extraction with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_video_worker, item): item for item in video_sources}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Celeb-DF frames (Parallel)"):
            res = future.result()
            if res:
                extracted_records.extend(res)

    # 4. Save Manifest
    df_manifest = pd.DataFrame(extracted_records)
    manifest_out = splits_dir / "celeb_df_extracted_real_frames.csv"
    df_manifest.to_csv(manifest_out, index=False)



    print("=" * 80)
    print("CELEB-DF-V2 REAL FRAME EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"   • Total Real Frames Extracted : {len(df_manifest):,} images ($256 \\times 256$)")
    print(f"   • Unique Identities Processed : {df_manifest['identity'].nunique():,} unique IDs")
    print(f"   • Output Directory            : {output_dir}")
    print(f"   • Manifest Saved to           : {manifest_out}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb-root", type=str,
                        default=os.environ.get("CELEB_DF_ROOT", "data/raw/Celeb-DF-v2"),
                        help="Celeb-DF-v2 source root (env CELEB_DF_ROOT)")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed/celeb_df_extracted")
    parser.add_argument("--splits-dir", type=str,
                        default="data/splits")
    parser.add_argument("--frames-per-video", type=int, default=15)
    args = parser.parse_args()

    extract_celeb_df_real_frames(
        Path(args.celeb_root),
        Path(args.output_dir),
        Path(args.splits_dir),
        frames_per_video=args.frames_per_video,
    )


if __name__ == "__main__":
    main()
