#!/usr/bin/env bash
# ============================================================
# Push code + models lên Hugging Face Hub (chạy TRÊN MÁY LOCAL,
# cần đã đăng nhập:  hf login  — token quyền WRITE).
#
# Cách dùng:
#   bash scripts/push_to_hub.sh                      # repo mặc định
#   bash scripts/push_to_hub.sh <repo_id>            # repo tuỳ chỉnh
#
# Dùng ALLOWLIST để KHÔNG bao giờ upload nhầm data lớn (test_data*/data*/outputs*).
# ============================================================
set -euo pipefail

REPO="${1:-ManhQuangAI/dinov3-deepfake-detection}"

hf upload "$REPO" . . \
  --include "src/**" \
  --include "scripts/**" \
  --include "models/**" \
  --include "requirements.txt" \
  --include "RUNPOD.md" \
  --include "README.md" \
  --exclude ".git/*" \
  --exclude ".venv/*"

echo ""
echo "OK: code + models đã push lên $REPO"
echo "Kiểm tra tại: https://huggingface.co/$REPO"
