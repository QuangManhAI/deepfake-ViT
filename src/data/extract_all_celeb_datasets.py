#!/usr/bin/env python3
"""
High-Speed Multithreaded Frame Extractor for Celeb-DF-v2 and Celeb-DF (v1).
Extracts BOTH Real and Fake (Celeb-synthesis) face frames for training and testing with 0% data leakage.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _process_video_worker(item):
    vid_path_str, ident_key, src_type, is_fake, method_name, out_dir_str, n_frames, res = item
    out_dir = Path(out_dir_str)
    cap = cv2.VideoCapture(vid_path_str)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_f <= 0:
        cap.release()
        return []

    step = max(1, total_f // n_frames)
    records = []
    f_idx = 0
    saved = 0
    vid_stem = Path(vid_path_str).stem

    while cap.isOpened() and saved < n_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if f_idx % step == 0:
            h, w, _ = frame.shape
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            sq = frame[top : top + min_dim, left : left + min_dim]
            resized = cv2.resize(sq, (res, res), interpolation=cv2.INTER_AREA)

            prefix = "fake" if is_fake == 1 else "real"
            out_fname = f"{prefix}_{vid_stem}_f{saved:02d}.jpg"
            out_p = out_dir / out_fname
            cv2.imwrite(str(out_p), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            records.append({
                "path": str(out_p),
                "label": is_fake,
                "method": method_name,
                "identity": ident_key,
                "domain": "cdc",
                "video": Path(vid_path_str).name,
                "source": src_type,
            })
            saved += 1
        f_idx += 1

    cap.release()
    return records


def extract_all(
    shared_data_root: Path,
    output_dir: Path,
    splits_dir: Path,
    real_fps: int = 15,
    fake_fps: int = 4,
    res: int = 256,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather Test Exclusions from List_of_testing_videos.txt
    test_exclusions = set()
    for c_dir in [shared_data_root / "Celeb-DF-v2", shared_data_root / "Celeb-DF"]:
        t_list = c_dir / "List_of_testing_videos.txt"
        if t_list.exists():
            with open(t_list, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_exclusions.add(Path(parts[1]).stem)

    print(f"🔒 Identified {len(test_exclusions)} official test videos to strictly isolate (0% leakage).")

    # 2. Gather All Training Videos across Celeb-DF-v2 and Celeb-DF v1
    tasks = []

    # A. Celeb-DF-v2 (Real & Fake)
    c_v2 = shared_data_root / "Celeb-DF-v2"
    if c_v2.exists():
        for v in (c_v2 / "Celeb-real").glob("*.mp4"):
            if v.stem not in test_exclusions:
                tasks.append((str(v), f"cdc:{v.stem}", "Celeb-DF-v2-Celeb-real", 0, "real", str(output_dir), real_fps, res))
        for v in (c_v2 / "YouTube-real").glob("*.mp4"):
            if v.stem not in test_exclusions:
                tasks.append((str(v), f"cdc:{v.stem}", "Celeb-DF-v2-YouTube-real", 0, "real", str(output_dir), real_fps, res))
        for v in (c_v2 / "Celeb-synthesis").glob("*.mp4"):
            if v.stem not in test_exclusions:
                id_stem = v.stem.split("_")[0] if "_" in v.stem else v.stem
                tasks.append((str(v), f"cdc:{id_stem}", "Celeb-DF-v2-Synthesis", 1, "CelebDFv2", str(output_dir), fake_fps, res))

    # B. Celeb-DF v1 (Real & Fake)
    c_v1 = shared_data_root / "Celeb-DF"
    if c_v1.exists():
        for v in (c_v1 / "Celeb-real").glob("*.mp4"):
            if v.stem not in test_exclusions:
                tasks.append((str(v), f"cdc:{v.stem}", "Celeb-DF-v1-Celeb-real", 0, "real", str(output_dir), real_fps, res))
        for v in (c_v1 / "YouTube-real").glob("*.mp4"):
            if v.stem not in test_exclusions:
                tasks.append((str(v), f"cdc:{v.stem}", "Celeb-DF-v1-YouTube-real", 0, "real", str(output_dir), real_fps, res))
        for v in (c_v1 / "Celeb-synthesis").glob("*.mp4"):
            if v.stem not in test_exclusions:
                id_stem = v.stem.split("_")[0] if "_" in v.stem else v.stem
                tasks.append((str(v), f"cdc:{id_stem}", "Celeb-DF-v1-Synthesis", 1, "CelebDFv1", str(output_dir), fake_fps, res))

    n_real = sum(1 for t in tasks if t[3] == 0)
    n_fake = sum(1 for t in tasks if t[3] == 1)
    print(f"🎬 Total Eligible Training Videos: {len(tasks):,} ({n_real:,} Real + {n_fake:,} Fake)")

    # 3. Fast Multiprocess Extraction
    num_workers = min(16, os.cpu_count() or 8)
    print(f"⚡ Launching high-speed extraction with {num_workers} parallel workers...")

    extracted = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_video_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting Celeb-DF frames (Real + Fake)"):
            r = fut.result()
            if r:
                extracted.extend(r)

    # 4. Save Master Extracted Manifest
    df_manifest = pd.DataFrame(extracted)
    manifest_out = splits_dir / "celeb_df_extracted_real_frames.csv"
    df_manifest.to_csv(manifest_out, index=False)

    n_ext_real = (df_manifest["label"] == 0).sum()
    n_ext_fake = (df_manifest["label"] == 1).sum()

    print("=" * 85)
    print("ALL CELEB-DF DATASETS (V1 & V2, REAL & FAKE) FULLY EXTRACTED")
    print("=" * 85)
    print(f"   • Total Frames Extracted : {len(df_manifest):,} images ($256 \times 256$)")
    print(f"   • Real Human Faces       : {n_ext_real:,} frames")
    print(f"   • Fake Deepfake Frames   : {n_ext_fake:,} frames")
    print(f"   • Manifest Saved to      : {manifest_out}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=str,
                        default=os.environ.get("DF40_ROOT", "data/raw"),
                        help="Shared read-only data root (env DF40_ROOT)")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed/celeb_df_extracted")
    parser.add_argument("--splits-dir", type=str,
                        default="data/splits")
    parser.add_argument("--real-fps", type=int, default=15)
    parser.add_argument("--fake-fps", type=int, default=4)
    args = parser.parse_args()

    extract_all(
        Path(args.shared_data_root),
        Path(args.output_dir),
        Path(args.splits_dir),
        real_fps=args.real_fps,
        fake_fps=args.fake_fps,
    )
