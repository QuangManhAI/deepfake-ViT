#!/usr/bin/env bash
# ============================================================
# Cài môi trường cho dự án anti-deepfake trên RunPod Ubuntu (GPU).
#
# Chạy trong thư mục repo (sau khi đã pull code từ HF Hub):
#   bash scripts/setup_ubuntu.sh
#
# Kết quả: venv .venv có đủ PyTorch CUDA + toàn bộ deps.
# ============================================================
set -euo pipefail

# Chạy với quyền root hay sudo tuỳ môi trường
SUDO=""
if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

echo "==> [1/4] Cài system packages (python3.11, unzip, git)"
$SUDO apt-get update -y
$SUDO apt-get install -y python3.11 python3.11-venv python3-pip unzip git

echo "==> [2/4] Tạo venv .venv"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> [3/4] Cài PyTorch + CUDA (cu124)"
# Kiểm tra driver GPU trước. Nếu nvidia-smi không chạy -> driver thiếu.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi không thấy. Kiểm tra driver GPU trên pod."
fi
# Nếu driver quá cũ (CUDA <= 12.1) thì đổi cu124 -> cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> [4/4] Cài các deps còn lại"
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " XONG. Kiểm tra GPU:"
python -c "
import torch
print('torch', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG có GPU — kiểm tra lại driver')
"
echo "============================================================"
