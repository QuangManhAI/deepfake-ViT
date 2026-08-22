"""Trích frame REAL Celeb-DF theo đúng protocol DF40 (từ dataset_json).

Đọc toàn bộ data/dataset_json/*.json, gom các frame index mà mỗi video real
(Celeb-DF-v2/YouTube-real, Celeb-DF-v2/Celeb-real) được tham chiếu, rồi trích
CHÍNH XÁC các index đó từ video mp4 bằng ffmpeg (select) + resize 256x256.

Output: data/raw/real-root/Celeb-DF-v2/<sub>/frames/<video_id>/<frame>.png
  (khớp cấu trúc đường dẫn trong JSON: deepfakes_detection_datasets/ -> data/raw/real-root/)

Chạy:
  .venv/bin/python src/data/extract_celeb_frames.py            # trích tất cả
  .venv/bin/python src/data/extract_celeb_frames.py --dry-run  # chỉ báo kế hoạch
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

CE = "data/raw/real-root/Celeb-DF-v2"
JSON_DIR = "data/dataset_json"
SIZE = 256  # khớp frame FF++ đã có
FFMPEG = "ffmpeg"


def collect_needed():
    """(sub, video) -> set(frame_idx) cho tất cả real Celeb-DF trong JSON."""
    need = defaultdict(set)
    for fp in sorted(glob.glob(os.path.join(JSON_DIR, "*.json"))):
        if fp.endswith("DF40_all.json"):
            continue
        with open(fp) as f:
            d = json.load(f)

        def walk(o):
            if isinstance(o, dict):
                if "frames" in o and o["frames"]:
                    for p in o["frames"]:
                        if "/Celeb-DF-v2/" in p and "/frames/" in p:
                            rel = p.split("Celeb-DF-v2/")[1]
                            parts = rel.split("/")
                            if len(parts) >= 4 and parts[1] == "frames":
                                need[(parts[0], parts[2])].add(
                                    int(parts[3].split(".")[0]))
                for v in o.values():
                    walk(v)

        walk(d)
    return need


def video_path(sub, vid):
    return os.path.join(CE, sub, f"{vid}.mp4")


def extract_video(sub, vid, idxs):
    """Trích đúng các index bằng select + scale, rename theo index gốc."""
    src = video_path(sub, vid)
    if not os.path.isfile(src):
        print(f"  [thiếu video] {src}")
        return 0
    out_dir = os.path.join(CE, sub, "frames", vid)
    os.makedirs(out_dir, exist_ok=True)

    idxs = sorted(idxs)
    # biểu thức select: eq(n,0)+eq(n,9)+...
    sel = "+".join(f"eq(n\\,{i})" for i in idxs)
    tmp = os.path.join(out_dir, ".tmp_%04d.png")
    r = subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", src,
         "-vf", f"select='{sel}',scale={SIZE}:{SIZE}:flags=bicubic",
         "-vsync", "0", tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ffmpeg lỗi] {src}: {r.stderr.strip()[:200]}")
        return 0

    frames = sorted(f for f in os.listdir(out_dir) if f.startswith(".tmp_"))
    got = len(frames)
    for k, f in enumerate(frames):
        os.rename(os.path.join(out_dir, f),
                  os.path.join(out_dir, f"{idxs[k]:03d}.png"))
    if got != len(idxs):
        print(f"  [thiếu {len(idxs)-got}] {sub}/{vid}: cần {len(idxs)}, được {got} "
              f"(video có thể ngắn hơn index tham chiếu)")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sub", default=None, help="chỉ trích 1 nhóm: YouTube-real / Celeb-real")
    args = ap.parse_args()

    need = collect_needed()
    if args.sub:
        need = {k: v for k, v in need.items() if k[0] == args.sub}
    print(f"Video real cần trích: {len(need)} "
          f"(YouTube-real={sum(1 for (s,_) in need if s=='YouTube-real')}, "
          f"Celeb-real={sum(1 for (s,_) in need if s=='Celeb-real')})")
    print(f"Tổng frame: {sum(len(v) for v in need.values()):,}\n")

    if args.dry_run:
        for (sub, vid), idxs in sorted(need.items()):
            src = video_path(sub, vid)
            ok = "✅" if os.path.isfile(src) else "❌ thiếu"
            print(f"  {ok} {sub}/{vid}: {len(idxs)} frame (max idx {max(idxs)})")
        print("\nDry-run xong. Bỏ --dry-run để trích thật.")
        return

    total = 0
    for (sub, vid), idxs in sorted(need.items()):
        got = extract_video(sub, vid, idxs)
        total += got
    print(f"\nĐã trích {total:,} frame (dự kiến {sum(len(v) for v in need.values()):,})")


if __name__ == "__main__":
    main()
