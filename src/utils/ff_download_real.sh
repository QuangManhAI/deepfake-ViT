#!/bin/bash
# Tải song song FF++ real (original_sequences/youtube, c23) từ server kaldir (TU Munich).
# Server chậm (~35KB/s/connection) nên dùng nhiều luồng song song + resume.
#
# Cách chạy:
#   src/utils/ff_download_real.sh [path_list_video] [số_luồng]
#     path_list_video mặc định /tmp/ff_vids.txt (1000 video)
#     số_luồng mặc định 24
set -u

VIDS="${1:-/tmp/ff_vids.txt}"
PARALLEL="${2:-12}"
OUT="data/raw/real-root/original_sequences/youtube/c23/videos"
BASE="https://kaldir.vc.in.tum.de/faceforensics/v3/original_sequences/youtube/c23/videos"

mkdir -p "$OUT"

worker() {
  local v="$1"
  local dst="$OUT/$v"
  local ok="$OUT/.$v.ok"
  # đã tải xong (có marker .ok) thì bỏ qua — resume
  [ -f "$ok" ] && return 0
  # bỏ partial cũ (killed giữa chừng) rồi tải lại từ đầu
  rm -f "$dst"
  curl -s --retry 5 --retry-delay 2 --connect-timeout 20 \
       --max-time 600 -o "$dst" "$BASE/$v" \
    && touch "$ok" \
    || { rm -f "$dst"; return 1; }
}
export OUT BASE
export -f worker

total=$(wc -l < "$VIDS")
echo "Bắt đầu tải $total video vào $OUT (P=$PARALLEL)"
cat "$VIDS" | xargs -P "$PARALLEL" -n 1 bash -c 'worker "$1"' _
echo "DONE: $(ls "$OUT" 2>/dev/null | grep -c '\.mp4$') files"
