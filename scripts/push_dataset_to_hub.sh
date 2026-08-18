#!/usr/bin/env bash
# ============================================================
# Push dataset test_data_v3.zip lên Hugging Face Hub.
#
# Trước tiên tạo dataset repo 1 lần:
#   hf repo create df40-test-data-v3 --type dataset --private
#
# Cách dùng:
#   bash scripts/push_dataset_to_hub.sh               # repo mặc định
#   bash scripts/push_dataset_to_hub.sh <repo_id>
# ============================================================
# LƯU Ý: bắt buộc --type dataset. `hf upload` MẶC ĐỊNH --type model
# → thiếu flag sẽ upload nhầm vào MODEL repo cùng tên (đã từng bị).
# ============================================================
set -euo pipefail

REPO="${1:-ManhQuangAI/df40-test-data-v3}"

if [ ! -f test_data_v3.zip ]; then
    echo "LỖI: chưa có test_data_v3.zip ở thư mục gốc (4.2GB). Chạy lệnh zip trước."
    exit 1
fi

echo "Đang upload test_data_v3.zip (~4.2GB) lên $REPO (dataset) — có thể mất vài phút..."
hf upload "$REPO" test_data_v3.zip test_data_v3.zip --type dataset

echo ""
echo "OK: test_data_v3.zip đã push lên https://huggingface.co/datasets/$REPO"
