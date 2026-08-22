#!/usr/bin/env bash
# ============================================================
# Push code + weights lên Hugging Face Hub (chạy TRÊN MÁY LOCAL,
# cần đã đăng nhập:  hf login  — token quyền WRITE).
#
# Cách dùng:
#   bash src/utils/push_to_hub.sh                      # repo mặc định
#   bash src/utils/push_to_hub.sh <repo_id>            # repo tuỳ chỉnh
#
# Dùng ALLOWLIST để KHÔNG bao giờ upload nhầm data lớn (test_data*/data*/experiments*).
# ============================================================
set -euo pipefail

REPO="${1:-ManhQuangAI/dinov3-deepfake-detection}"

hf upload "$REPO" . . \
  --include "src/**" \
  --include "experiments/checkpoints/weights/**" \
  --include "requirements.txt" \
  --include "RUNPOD.md" \
  --include "README.md" \
  --exclude ".git/*" \
  --exclude ".venv/*"

echo ""
echo "OK: code + models đã push lên $REPO"
echo "Kiểm tra tại: https://huggingface.co/$REPO"
