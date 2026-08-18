#!/usr/bin/env python
"""Tải toàn bộ data DF40 từ Google Drive (có retry + log).

⚠️ KHÔNG còn là nguồn khuyến nghị: các link Google Drive của DF40 thường
không truy cập được (gdown rc=1) và `gdown` không tải được folder lồng sâu.
Nguồn ưu tiên hiện tại là Hugging Face Hub:
    hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40
Script này giữ lại cho trường hợp Drive vẫn hoạt động (real files + JSON).

- Lưu vào data/DF40/
- Log từng bước ra data/DF40/download.log (timestamp, retry quota, summary)
- Tự retry khi Google Drive báo "Too many users" (hết quota tạm thời)
- Resume được (gdown --continue / --remaining-ok)

Cách chạy:
  .venv/bin/python src/data/download_df40.py                         # tất cả
  .venv/bin/python src/data/download_df40.py --only real_ffpp,real_celebdf,dataset_json
  .venv/bin/python src/data/download_df40.py --skip test_data,train_data

LƯU Ý QUAN TRỌNG:
  gdown liệt kê TỐI ĐA 50 mục mỗi folder (do parse HTML). Folder test_data/train_data
  của DF40 lồng sâu (hàng trăm subfolder) nên gdown sẽ tải THIẾU.
  -> Để tải đủ 93GB/50GB lồng sâu, dùng rclone (khuyến nghị) thay vì gdown.
  Các mục file (real) + JSON (folder nhỏ) thì gdown tải OK.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

BASE = os.path.join("data", "DF40")
LOG_PATH = os.path.join(BASE, "download.log")
PY = sys.executable

# (key, kind, google-drive-id, mô tả)
ITEMS = [
    ("real_ffpp",    "file",   "1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb",      "FF++ real images (file)"),
    ("real_celebdf", "file",   "1P9Ep4-nxGpBX8LZGq2UyxqoCL6sBDN8Z",      "Celeb-DF real images (file)"),
    ("dataset_json", "folder", "19VhAL4aDJOKvhl9stEq_ymFeHiXo6_j-",      "JSON path files (small folder)"),
    ("train_data",   "folder", "1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_",      "train fake (~50GB, NESTED)"),
    ("test_data",    "folder", "1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD",      "test fake (~93GB, NESTED)"),
]

QUOTA_MARKERS = ["too many users", "quota", "cannot be downloaded", "exceeded",
                 "can't download", "please try"]


def log(msg):
    line = "[{}] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run_gdown(key, item):
    kind, gid, _note = item
    outdir = os.path.join(BASE, key)
    os.makedirs(outdir, exist_ok=True)
    if kind == "file":
        cmd = [PY, "-m", "gdown", "--id", gid, "--continue", "--quiet"]
    else:
        cmd = [PY, "-m", "gdown", "--folder",
               "https://drive.google.com/drive/folders/" + gid,
               "--continue", "--remaining-ok", "--quiet"]
    return subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)


def download_one(key, item, max_attempts, backoff):
    log("== START {} : {} ==".format(key, item[2]))
    for attempt in range(1, max_attempts + 1):
        log("[{}] attempt {}/{}".format(key, attempt, max_attempts))
        try:
            r = run_gdown(key, item)
        except Exception as e:  # noqa: BLE001
            log("[{}] exception: {} — retry in 30s".format(key, e))
            time.sleep(30)
            continue

        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        for line in combined.strip().splitlines():
            s = line.strip()
            if s and ("error" in s.lower() or "fail" in s.lower() or "download" in s.lower()):
                log("[{}]   {}".format(key, s[-160:]))

        if any(m in combined.lower() for m in QUOTA_MARKERS):
            wait = min(backoff * (2 ** (attempt - 1)), 3600)
            log("[{}] QUOTA/LIMIT — chờ {}s rồi retry".format(key, int(wait)))
            time.sleep(wait)
            continue
        if r.returncode != 0:
            log("[{}] gdown rc={} — retry in 30s".format(key, r.returncode))
            time.sleep(30)
            continue
        log("[{}] DONE".format(key))
        return True

    log("[{}] FAILED sau {} lần thử".format(key, max_attempts))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="chỉ tải các key này (phẩy ngăn cách)")
    ap.add_argument("--skip", help="bỏ qua các key này (phẩy ngăn cách)")
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument("--backoff", type=float, default=120.0, help="giây chờ đầu tiên khi gặp quota")
    args = ap.parse_args()

    os.makedirs(BASE, exist_ok=True)
    log("======== DF40 download ========")
    log("items: {}".format(", ".join(k for k, *_ in ITEMS)))

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    results = {}
    for key, kind, gid, note in ITEMS:
        item = (kind, gid, note)
        if only is not None and key not in only:
            continue
        if key in skip:
            log("[{}] SKIPPED".format(key))
            continue
        results[key] = download_one(key, item, args.max_attempts, args.backoff)

    log("======== SUMMARY ========")
    for key, ok in results.items():
        log("  {:<14} {}".format(key, "OK" if ok else "FAILED"))
    log("log: {}".format(LOG_PATH))


if __name__ == "__main__":
    main()
