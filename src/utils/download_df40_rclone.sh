#!/bin/bash
# Tải DF40 test/train data (folder lồng sâu) bằng rclone.
# Chạy sau khi đã `rclone config` tạo remote tên "gdrive".
# Có thể Ctrl-C rồi chạy lại — rclone tự resume (không tải lại file đã có).
set -e
BASE="data/DF40"
mkdir -p "$BASE"

TEST_ID="1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD"
TRAIN_ID="1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_"

echo "== Kiểm tra: liệt kê 40 method trong test_data =="
rclone lsd gdrive: --drive-root-folder-id "$TEST_ID"

echo ""
echo "== Tải test_data (~93GB) =="
rclone copy gdrive: "$BASE/test_data" \
  --drive-root-folder-id "$TEST_ID" \
  -P --transfers 4 --drive-chunk-size 32M \
  --retries 10 --low-level-retries 20 --timeout 30s

echo ""
echo "== Tải train_data (~50GB) =="
rclone copy gdrive: "$BASE/train_data" \
  --drive-root-folder-id "$TRAIN_ID" \
  -P --transfers 4 --drive-chunk-size 32M \
  --retries 10 --low-level-retries 20 --timeout 30s

echo "DONE"
