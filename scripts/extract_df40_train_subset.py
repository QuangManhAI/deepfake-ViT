#!/usr/bin/env python3
"""Trích xuất subset tối thiểu từ DF40_train (HF Hub) bằng HTTP Range + gộp theo cặp.

Thay vì tải cả zip (~2GB/method), chỉ tải:
  - central directory (đuôi file) → để liệt kê member + offset
  - VỚI MỖI CẶP được chọn: 1 dải byte duy nhất phủ toàn bộ ảnh của cặp → parse local header
Vậy mỗi cặp ~2-3 request HTTP thay vì ~10 request/file. Tổng tải ~250-300MB.

Cách dùng:
  .venv/bin/python scripts/extract_df40_train_subset.py \
      --out data_train_local --pairs 8 --seed 42 \
      --methods faceswap facedancer inswap blendface fsgan mobileswap
"""
import argparse
import http.client
import json
import os
import random
import re
import socket
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from urllib.parse import urlparse

REPO = "ManhQuangAI/DF40_train"
TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
MARGIN = 16 * 1024  # 16KB đệm cho extra_len / data descriptor


def load_token():
    with open(TOKEN_PATH) as f:
        return f.read().strip()


def resolve_cdn_url(token, filename):
    req = urllib.request.Request(
        f"https://huggingface.co/datasets/{REPO}/resolve/main/{filename}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.geturl()


class RangeFile:
    """File-like đọc từ HTTP Range, dùng cho zipfile.ZipFile (central dir)."""

    def __init__(self, url):
        p = urlparse(url)
        self.host = p.netloc
        self.path = p.path + (("?" + p.query) if p.query else "")
        self.conn = http.client.HTTPSConnection(self.host, timeout=300)
        self._pos = 0
        self._len = None

    def _read_retry(self, start, end, n_retry=3):
        """Đọc 1 dải byte, retry với connection mới khi timeout/đứt."""
        for attempt in range(n_retry):
            try:
                self.conn.request("GET", self.path,
                                  headers={"Range": f"bytes={start}-{end}"})
                resp = self.conn.getresponse()
                return resp, resp.read()
            except (socket.timeout, http.client.IncompleteRead,
                    BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"  !! fetch [{start}-{end}] lỗi {type(e).__name__}: "
                      f"{e} (thử {attempt + 1}/{n_retry})", flush=True)
                time.sleep(2 * (attempt + 1))
                self.conn = http.client.HTTPSConnection(self.host, timeout=300)
        raise RuntimeError(f"fetch [{start}-{end}] thất bại sau {n_retry} lần")

    def _fetch(self, start, end):
        resp, data = self._read_retry(start, end)
        if resp.status == 206:
            cr = resp.getheader("Content-Range")
            if self._len is None and cr and "/" in cr:
                self._len = int(cr.split("/")[1])
            return data
        if self._len is None:
            self._len = len(data)
        return data[start:end + 1]

    def seek(self, off, whence=0):
        if whence == 0:
            self._pos = off
        elif whence == 1:
            self._pos += off
        elif whence == 2:
            if self._len is None:
                self._fetch(0, 0)
            self._pos = self._len + off
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if self._len is not None and self._pos >= self._len:
            return b""
        if n is None or n < 0:
            n = min(16 * 1024 * 1024, (self._len or 16 * 1024 * 1024) - self._pos)
        end = self._pos + n - 1
        data = self._fetch(self._pos, end)
        self._pos += len(data)
        return data

    def seekable(self):
        return True

    def fetch(self, start, end):
        return self._fetch(start, end)


FRAME_RE = re.compile(r"\.(png|jpg|jpeg)$", re.IGNORECASE)


def inflate(raw, method):
    if method == 0:
        return raw
    if method == 8:
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        return d.decompress(raw) + d.flush()
    raise ValueError(f"unsupported compress method {method}")


def extract_pair_span(rf, cdn_len, zip_name, members, out_dir):
    """Gộp toàn bộ member của 1 cặp thành 1 dải byte, parse + ghi file."""
    members = sorted(members, key=lambda zi: zi.header_offset)
    lo = members[0].header_offset
    # end: header_offset + 30 + name_len + extra_len + comp_size, thêm margin
    hi = max(zi.header_offset + 30 + len(zi.filename) + MARGIN + zi.compress_size
             for zi in members)
    hi = min(hi, cdn_len - 1)
    block = rf.fetch(lo, hi)

    n = 0
    for zi in members:
        local_off = zi.header_offset - lo
        if local_off + 30 > len(block):
            raise RuntimeError("span quá ngắn — cần xử lý thêm")
        sig, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", block[local_off:local_off + 30])
        if sig != 0x04034B50:
            raise RuntimeError(f"lệch local header tại {zi.filename}: sig={sig:#x}")
        data_start = local_off + 30 + name_len + extra_len
        data_end = data_start + zi.compress_size
        if data_end > len(block):
            # thiếu → fetch thêm phần đuôi
            extra = rf.fetch(lo + len(block), lo + data_end - 1)
            block += extra
        raw = block[data_start:data_end]
        try:
            content = inflate(raw, zi.compress_type)
        except zlib.error as e:
            raise RuntimeError(f"inflate lỗi {zi.filename}: {e}")
        if len(content) != zi.file_size:
            raise RuntimeError(f"{zi.filename}: size {len(content)} != {zi.file_size}")
        # verify CRC (nếu descriptor thì CRC nằm trong central dir — đúng)
        actual = zlib.crc32(content) & 0xFFFFFFFF
        if actual != zi.CRC:
            raise RuntimeError(f"{zi.filename}: CRC {actual:#x} != {zi.CRC:#x}")
        with open(os.path.join(out_dir, os.path.basename(zi.filename)), "wb") as f:
            f.write(content)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_train_local")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+",
                    default=["faceswap", "facedancer", "inswap",
                             "blendface", "fsgan", "mobileswap"])
    args = ap.parse_args()

    token = load_token()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {}

    for mth in args.methods:
        fn = f"{mth}.zip"
        print(f"[{mth}] resolve CDN URL...", flush=True)
        url = resolve_cdn_url(token, fn)
        rf = RangeFile(url)
        zf = zipfile.ZipFile(rf)
        infos = zf.infolist()
        cdn_len = rf._len
        zf.close()
        frames = [zi for zi in infos if FRAME_RE.search(zi.filename)]
        prefix = f"{mth}/frames/"
        pairs = sorted({zi.filename[len(prefix):].split("/")[0] for zi in frames
                        if zi.filename.startswith(prefix)})
        if not pairs:
            print(f"  !! không có pair nào trong {fn}", flush=True)
            continue
        chosen = rng.sample(pairs, min(args.pairs, len(pairs)))
        n_imgs = 0
        n_done = 0
        for pair in chosen:
            out_pair = os.path.join(args.out, mth, "fake", pair)
            if os.path.isdir(out_pair) and os.listdir(out_pair):
                # đã extract (resume) — đếm lại số ảnh, không tải lại
                n_done += len(os.listdir(out_pair))
                continue
            os.makedirs(out_pair, exist_ok=True)
            members = [zi for zi in frames
                       if zi.filename.startswith(f"{prefix}{pair}/")]
            n_imgs += extract_pair_span(rf, cdn_len, fn, members, out_pair)
        n_imgs += n_done
        manifest[mth] = {
            "pairs": chosen,
            "n_images": n_imgs,
            "n_total_pairs": len(pairs),
        }
        print(f"[{mth}] xong: {len(chosen)} cặp, {n_imgs} ảnh "
              f"(tổng {len(pairs)} cặp)", flush=True)

    with open(os.path.join(args.out, "_extract_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(v["n_images"] for v in manifest.values())
    print(f"\nXONG: {total} ảnh fake trong {args.out}")


if __name__ == "__main__":
    sys.exit(main())
