"""Trích frame từ video REAL (FF++ original / Celeb-DF real) về ảnh 256x256,
theo đúng protocol của DF40: lấy mỗi frame thứ 6 (select='not(mod(n,6))'),
resize 256x256, tên file giữ số frame gốc (000.png, 006.png, 012.png, ...).

Đích: tạo phần "real" ghép với fake cdf/ff của các method chỉ-có-fake trong
DF40, cho benchmark cân bằng real/fake.

Chạy:
  .venv/bin/python scripts/extract_real_frames.py \
      --videos data/real-root/original_sequences/youtube/c23/videos/585.mp4 \
      --out data/real-root/frames/ff++/585 \
      --step 6 --size 256
  # nhiều video:
  find data/real-root/original_sequences/youtube/c23/videos -name '*.mp4' | \
    head -50 | xargs -I{} .venv/bin/python scripts/extract_real_frames.py --videos {} --out data/real-root/frames/ff++
"""
import argparse
import os
import subprocess
import sys

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def frame_count(video):
    """Số frame video (dùng để biết rename đúng)."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", video],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return None


def extract(video, out_dir, step, size):
    """Trích frame mỗi `step` frame, resize `size`x`size`, giữ số frame gốc."""
    os.makedirs(out_dir, exist_ok=True)
    n_total = frame_count(video)
    if n_total is None:
        print(f"  [skip] không đọc được frame count: {video}")
        return 0

    tmp = os.path.join(out_dir, ".tmp_%04d.png")
    vf = f"select='not(mod(n\\,{step}))',scale={size}:{size}:flags=bicubic"
    r = subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", video,
         "-vf", vf, "-vsync", "0", tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [skip] ffmpeg lỗi {video}: {r.stderr.strip()[:200]}")
        return 0

    # rename .tmp_XXXX.png -> <index gốc>.png
    frames = sorted(f for f in os.listdir(out_dir) if f.startswith(".tmp_"))
    for i, f in enumerate(frames):
        src_idx = i * step  # frame gốc (0, step, 2*step, ...)
        os.rename(os.path.join(out_dir, f),
                  os.path.join(out_dir, f"{src_idx:03d}.png"))
    print(f"  {os.path.basename(video)}: {len(frames)} frames -> {out_dir}")
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="video(s) mp4, phân cách bằng dấu phẩy hoặc lặp lại --videos")
    ap.add_argument("--out", required=True, help="thư mục xuất (mỗi video 1 thư mục con)")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    vids = [v for v in args.videos.split(",") if v]
    total = 0
    for v in vids:
        if not os.path.isfile(v):
            print(f"  [thiếu] {v}")
            continue
        # out/<tên video gốc>/<frame>.png
        vid_dir = os.path.join(args.out, os.path.splitext(os.path.basename(v))[0])
        if os.path.exists(vid_dir) and len(os.listdir(vid_dir)):
            print(f"  [skip đã có] {vid_dir}")
            continue
        total += extract(v, vid_dir, args.step, args.size)
    print(f"\nTổng frame trích: {total:,}")


if __name__ == "__main__":
    main()
